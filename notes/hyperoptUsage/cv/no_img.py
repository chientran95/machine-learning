from abc import abstractmethod, ABC
from typing import Type

import numpy as np
from hyperopt import STATUS_OK
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from cv.ensemble import bagging
from cv.hpt import resume_trials


class ObjectiveCV(ABC):
    _X_pseudo = None
    _y_pseudo = None

    def __init__(self, X, y, pipe=None, seed=1, eps=1e-15):
        self._X = X
        self._y = y
        self._pipe = pipe
        self._seed = seed
        self._eps = eps

    @abstractmethod
    def make_predictive(self, sp, X_tr, y_tr, X_val, y_val):
        pass

    def pre_fit(self, space):
        pass

    def _pipe_fit(self, X, y):
        if self._pipe:
            self._pipe.fit(X, y)
            # print(self._pipe.named_steps["selection"].scores_)

    def _pipe_transform(self, X):
        if self._pipe:
            return self._pipe.transform(X)
        return X

    def set_pseudo(self, X, y):
        self._X_pseudo = X
        self._y_pseudo = y

    # noinspection PyCallingNonCallable
    # noinspection PyTupleAssignmentBalance
    def cv(self, space, X_test=None):
        X, y = self._X, self._y

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self._seed)

        scores_tr = []
        scores_val = []
        blended = np.zeros_like(y).astype('f')
        test_preds = []

        for train_idx, val_idx in skf.split(np.zeros_like(y), y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Add pseudo labeled data into training data
            if self._X_pseudo is not None:
                X_tr = np.concatenate((X_tr, self._X_pseudo))
                y_tr = np.concatenate((y_tr, self._y_pseudo))

            # Preprocess
            self.pre_fit(space)
            self._pipe_fit(X_tr, y_tr)
            X_tr = self._pipe_transform(X_tr)
            X_val = self._pipe_transform(X_val)

            # Train
            pred_fn = self.make_predictive(space, X_tr, y_tr, X_val, y_val)

            # Predict
            probs_tr = pred_fn(X_tr)
            probs_val = pred_fn(X_val)
            if X_test is not None:
                test_preds.append(pred_fn(self._pipe_transform(X_test)))

            scores_tr.append(log_loss(y_tr, probs_tr, eps=self._eps))
            scores_val.append(log_loss(y_val, probs_val, eps=self._eps))

            blended[val_idx] = probs_val.reshape(-1)

        test_preds = np.array(test_preds).T

        return np.mean(scores_tr), np.mean(scores_val), blended, test_preds

    def objective(self, sp):
        score_tr, score_val, _, _ = self.cv(sp)
        print("---", score_tr, score_val, sp)

        return {
            'loss': score_val,
            'status': STATUS_OK,
            'score_tr': score_tr,
            'score_val': score_val,
        }


class CmdFactory:
    def __init__(self, cls: Type[ObjectiveCV]):
        self._cls = cls

    def hpt(self, train_data, space, trials_file, max_iter, steps, seed, pipe=None):
        X, y = train_data
        cv = self._cls(X, y, seed=seed, pipe=pipe)

        resume_trials(cv, space, trials_file, max_iter, trials_step=steps)

    def pred(self, train_data, test_data, params, out_tr, out_test, n_bags, seed, pipe=None):
        X, y = train_data
        X_test = test_data

        def pred_nth_bagging(n):
            cv = self._cls(X, y, seed=seed + n, pipe=pipe)
            score_tr, score_val, train_preds, test_preds = cv.cv(params, X_test=X_test)
            print('Score: ', score_tr, score_val)
            return train_preds, test_preds

        train_bag_preds, test_bag_preds = bagging(n_bags=n_bags,
                                                  pred_fn=pred_nth_bagging)
        np.save(out_tr, train_bag_preds)
        np.save(out_test, test_bag_preds)
