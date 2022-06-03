import os
import numpy as np


class Metric():
    def __init__(self, y_true, y_pred):
        assert len(y_true.shape) == 3 and y_true.shape[-1] == 1
        assert len(y_pred.shape) == 3 and y_pred.shape[-1] == 1

        self.y_true = np.array(y_true[:, :, 0])
        self.y_pred = np.array(y_pred[:, :, 0])

    def mean_absrel_err(self):
        # Calculate Mean Absolute Relative Error
        pixel_errs = np.abs(self.y_true - self.y_pred)/self.y_true
        return pixel_errs.mean()

    def mean_sqrel_err(self):
        # Calculate Mean Squared Relative Error
        pixel_errs = ((self.y_true - self.y_pred)**2)/self.y_true
        return pixel_errs.mean()

    def rms_err(self):
        # Calculate Root Mean Squared Error
        pixel_errs = (self.y_true - self.y_pred)**2
        return np.sqrt(pixel_errs.mean())

    def rms_log_err(self):
        # Calculate Root Mean Squared log Error
        mod_y_true = np.clip(self.y_true, 0.000001, self.y_true.max())
        mod_y_pred = np.clip(self.y_pred, 0.000001, self.y_pred.max())
        pixel_errs = (np.log(mod_y_true) - np.log(mod_y_pred))**2
        return np.sqrt(pixel_errs.mean())

    def percentage_of_pixel_acc(self, theshold=1.25):
        # Calculate percentage of pixel accuracy w.r.t threshold
        thresh = np.maximum((self.y_pred/self.y_true), (self.y_true/self.y_pred))
        return {'delta1': (thresh < theshold).astype(float).mean(),
                'delta2': (thresh < (theshold**2)).astype(float).mean(),
                'delta3': (thresh < (theshold**3)).astype(float).mean()}
