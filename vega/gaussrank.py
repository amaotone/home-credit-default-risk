import numpy as np
import pandas  as pd
from scipy.special import erfinv
from scipy.stats import rankdata
from tqdm import tqdm


def gauss_rank(dfs, cols, fill_value=np.nan, eps=0.01):
    df: pd.DataFrame = pd.concat(dfs, axis=0)
    for f in tqdm(cols):
        mask = df[f].isnull()
        df[f] = rankdata(df[f]) - 1
        df.loc[mask, f] = np.nan
        df[f] = erfinv(df[f] / df[f].max() * 2 * (1 - eps) - (1 - eps))
        df.loc[mask, f] = fill_value
        
    res = []
    prev = 0
    for d in dfs:
        res.append(df[d.columns][prev:len(d)].copy())
        prev += len(d)
    return res