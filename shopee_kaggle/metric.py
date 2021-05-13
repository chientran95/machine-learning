import numpy as np
import pandas as pd

def getMetric(col):
    def f1score(row):
        n = len(np.intersect1d(row.target,row[col]))
        return 2*n / (len(row.target)+len(row[col]))
    return f1score

# Usage:
# train['f1'] = train.apply(getMetric('oof'),axis=1)
# print('CV score =', train.f1.mean())