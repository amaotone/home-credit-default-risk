import numpy as np
import pandas as pd
from tqdm import tqdm


class WeightOfEvidence:
    def __init__(self, cols=None, unknown=-1, prefix=None, suffix=None):
        self.unknown = unknown
        self.prefix = prefix + '_' if prefix else ''
        self.suffix = '_' + suffix if suffix else ''
        self.cols = cols
        self.maps = {}
    
    def fit(self, X, y):
        if self.cols is None:
            self.cols = X.select_dtypes(['object']).columns.tolist()
        for f in tqdm(self.cols):
            df = pd.DataFrame({f: X[f].fillna('NaN'), 'TARGET': y})
            piv = df.groupby(f).TARGET.value_counts().unstack().fillna(0).astype(int)
            self.maps[f] = ((piv + 0.5) / piv.sum(axis=0)).apply(lambda x: np.log(x.iloc[0] / x.iloc[1]), axis=1)
    
    def transform(self, X):
        r = pd.DataFrame()
        for f in tqdm(self.cols):
            ref = X[f].fillna('NaN')
            mask = ~ref.isin(self.maps.keys())
            r[f] = ref.replace(self.maps[f])
            r[mask] = self.unknown
        r.columns = self.prefix + r.columns + self.suffix
        return r
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
