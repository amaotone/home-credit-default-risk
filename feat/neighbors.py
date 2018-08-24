import os
import sys

import annoy
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../../spica')
from spica.features.base import Feature, generate_features, get_arguments

from utils import timer
from config import *

Feature.dir = '../working'
Feature.prefix = 'neighbor'


def get_neightbor_feat(X_trn, X_tst, k=5, n_trees=10):
    u = annoy.AnnoyIndex(X_trn.shape[1])
    print('add items')
    for i, v in enumerate(pd.DataFrame(X_trn).drop_duplicates(keep='false').values):
        u.add_item(i, v)
    print('build')
    u.build(n_trees)
    print('get neighbors')
    res = np.empty(X_tst.shape[0])
    for i, v in enumerate(X_tst):
        res[i] = np.mean(u.get_nns_by_vector(v, n=k, include_distances=True)[1])
    return res


def build(X, n_trees=10):
    u = annoy.AnnoyIndex(X.shape[1])
    for i, v in enumerate(pd.DataFrame(X).drop_duplicates(keep=False).values):
        u.add_item(i, v)
    u.build(n_trees)
    return u


def get_feat(X, u, k=5):
    res = np.empty(X.shape[0])
    for i, v in enumerate(X.values):
        res[i] = np.mean(u.get_nns_by_vector(v, n=k, include_distances=True)[1])
    return res


def neighbors(train, test, target, cv: PredefinedSplit, k=5, n_trees=10):
    res_train = np.zeros((train.shape[0], 2))
    res_test = np.zeros((test.shape[0], 2))
    for i, (trn_idx, val_idx) in tqdm(enumerate(cv.split(train)), total=cv.get_n_splits()):
        target_trn = target.iloc[trn_idx]
        X_trn = train.iloc[trn_idx]
        X_val = train.iloc[val_idx]
        n = X_trn[target_trn == 0]
        p = X_trn[target_trn == 1]
        for j, X in enumerate([n, p]):
            u = build(X, n_trees)
            res_train[val_idx, j] = get_feat(X_val, u, k=k)
            res_test[:, j] += get_feat(test, u, k)
    res_test /= cv.get_n_splits()
    return res_train, res_test


class NeighborBuro(Feature):
    prefix = 'neighbor_buro'
    
    def create_features(self):
        global X_train, X_test
        sc = StandardScaler()
        df_train = pd.DataFrame(sc.fit_transform(X_train))
        df_test = pd.DataFrame(sc.transform(X_test))
        res_train, res_test = neighbors(df_train, df_test, train.TARGET, cv, k=5, n_trees=10)
        self.train = pd.DataFrame(res_train, columns=['neg', 'pos'])
        self.test = pd.DataFrame(res_test, columns=['neg', 'pos'])


if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)[['TARGET']]
        test = pd.read_feather(TEST)
        cv_id = pd.read_feather('../input/cv_id.ftr')
        cv = PredefinedSplit(cv_id)
        
        dfs = [pd.read_feather(str(f)) for f in sorted(Path('../working/').glob('buro_*_train.ftr'))]
        X_train = pd.concat(dfs, axis=1)  # type: pd.DataFrame
        dfs = [pd.read_feather(str(f)) for f in sorted(Path('../working/').glob('buro_*_test.ftr'))]
        X_test = pd.concat(dfs, axis=1)  # type: pd.DataFrame
    
    with timer('preprocessing'):
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_train.fillna(X_train.mean(), inplace=True)
        X_test.fillna(X_train.mean(), inplace=True)
    
    with timer('create dataset'):
        generate_features(globals(), args.force)
