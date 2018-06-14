import numpy as np


def longest_streak(series):
    res = 0
    t = 0
    for v in series:
        if v != 0 and v != np.nan:
            t += 1
        else:
            res = max(res, t)
            t = 0
    return res
