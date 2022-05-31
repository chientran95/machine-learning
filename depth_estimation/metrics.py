import os
import numpy as np
import pandas as pd


class Metric():
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

        assert len(y_true.shape) == 2
        assert len(y_pred.shape) == 2

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
        pixel_errs = (np.log(self.y_true) - np.log(self.y_pred))**2
        return np.sqrt(pixel_errs.mean())

    def percentage_of_pixel_acc(self, theshold=1.25):
        thresh = np.maximum((self.y_pred/self.y_true), (self.y_true/self.y_pred))
        return {'delta1': (thresh < 1.25).astype(float).mean(),
                'delta2': (thresh < (1.25**2)).astype(float).mean(),
                'delta3': (thresh < (1.25**3)).astype(float).mean()}
