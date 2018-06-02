import numpy as np
import pandas  as pd
from scipy.special import erfinv
from scipy.stats import rankdata
from tqdm import tqdm


def gauss_rank(X_train, X_test, cols, eps=0.01):
    df: pd.DataFrame = pd.concat([X_train, X_test], axis=0)
    for f in tqdm(cols):
        null_row = df[f].isnull()
        df[f] = rankdata(df[f]) - 1
        df.loc[null_row, f] = np.nan
        df[f] = erfinv(df[f] / df[f].max() * 2 * (1 - eps) - (1 - eps))
    return df.iloc[:len(X_train)].copy(), df.iloc[len(X_train):]
