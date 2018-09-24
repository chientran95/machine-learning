import pickle

import numpy as np
from hyperopt import fmin, tpe, Trials


def resume_trials(cv, space, trials_file, max_iter, trials_step=1):
    # noinspection PyBroadException
    def run_trials():
        try:
            trials = pickle.load(open(trials_file, 'rb'))
        except:
            trials = Trials()

        cur_trials = len(trials.trials)
        max_trials = len(trials.trials) + trials_step
        print("Run from {} trials to {} (+{}) trials".format(cur_trials, max_trials, trials_step))

        best = fmin(fn=cv.objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=max_trials,
                    trials=trials)
        print(np.argmin(trials.losses()), trials.best_trial['result'])
        print("Best :", best)

        with open(trials_file, "wb") as f:
            pickle.dump(trials, f)

    print("Start %s more iterations..." % max_iter)
    for n in range(max_iter):
        run_trials()
