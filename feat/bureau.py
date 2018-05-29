import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat import Feature, get_arguments, generate_features
from utils import timer
from config import *

CREDIT_CAT_COLS = ['NAME_CONTRACT_STATUS']


class BureauActiveCount(Feature):
    def create_features(self):
        df = buro.groupby('SK_ID_CURR').CREDIT_ACTIVE.value_counts().unstack().fillna(0).astype(int)
        df.columns = 'bureau_' + df.columns + '_count'
        sum_ = df.sum(axis=1)
        for f in df.columns:
            df[f.replace('_count', '_ratio')] = df[f] / sum_
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class BureauActiveAndTypeProduct(Feature):
    def create_features(self):
        tmp = buro.copy()
        tmp['TMP'] = tmp.CREDIT_ACTIVE + '_' + tmp.CREDIT_TYPE
        df = tmp.groupby('SK_ID_CURR').TMP.value_counts().unstack().fillna(0).astype(int)
        df.columns = 'bureau_' + df.columns + '_count'
        sum_ = df.sum(axis=1)
        for f in df.columns:
            df[f.replace('_count', '_ratio')] = df[f] / sum_
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class BureauInterval(Feature):
    # https://www.kaggle.com/shanth84/home-credit-bureau-data-feature-engineering
    def create_features(self):
        df = pd.DataFrame()
        group = buro.groupby('SK_ID_CURR').DAYS_CREDIT
        df['bureau_interval_latest'] = group.apply(lambda x: x.diff().iloc[-1]).fillna(0)
        df['bureau_interval_min'] = group.apply(lambda x: x.diff().min()).fillna(0)
        df['bureau_interval_mean'] = group.apply(lambda x: x.diff().mean()).fillna(0)
        df['bureau_interval_max'] = group.apply(lambda x: x.diff().max()).fillna(0)
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class BureauEnddate(Feature):
    # https://www.kaggle.com/shanth84/home-credit-bureau-data-feature-engineering
    def create_features(self):
        tmp = buro.copy()
        tmp['TMP'] = tmp.DAYS_CREDIT_ENDDATE > 0
        df = pd.DataFrame()
        df['bureau_future_expire_count'] = tmp.groupby('SK_ID_CURR').TMP.sum().fillna(0)
        df['bureau_future_expire_ratio'] = tmp.groupby('SK_ID_CURR').TMP.mean().fillna(0)
        future = buro.query('DAYS_CREDIT_ENDDATE > 0').groupby('SK_ID_CURR').DAYS_CREDIT_ENDDATE
        df['bureau_future_expire_mean'] = future.mean().fillna(0)
        df['bureau_future_expire_min'] = future.min().fillna(0)
        df['bureau_future_expire_interval_first'] = future.apply(lambda x: x.diff().iloc[0]).fillna(0)
        df['bureau_future_expire_interval_min'] = future.apply(lambda x: x.diff().min()).fillna(0)
        df['bureau_future_expire_interval_mean'] = future.apply(lambda x: x.diff().mean()).fillna(0)
        df['bureau_future_expire_interval_max'] = future.apply(lambda x: x.diff().max()).fillna(0)
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class BureauAmountPairwise(Feature):
    def create_features(self):
        df = pd.DataFrame()
        amt_cols = buro.filter(regex='^AMT_').columns.tolist()
        for i, j in itertools.combinations(amt_cols, 2):
            df[f'bureau_{i}_minus_{j}'] = buro[i] - buro[j]
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class BureauProlonged(Feature):
    def create_features(self):
        df = pd.DataFrame()
        df['bureau_prolonged_count'] = buro.groupby('SK_ID_CURR').CNT_CREDIT_PROLONG.fillna(0).sum()
        df['bureau_prolonged_mean'] = buro.groupby('SK_ID_CURR').CNT_CREDIT_PROLONG.fillna(0).mean()
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


if __name__ == '__main__':
    args = get_arguments('POS CASH')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)[['SK_ID_CURR']]
        test = pd.read_feather(TEST)[['SK_ID_CURR']]
        buro = pd.read_feather(BURO)
    
    with timer('preprocessing'):
        buro.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
        buro = buro.sort_values(['SK_ID_CURR', 'DAYS_CREDIT']).reset_index(drop=True)
        buro.loc[:, buro.columns.str.startswith('AMT_')] = np.log1p(buro.filter(regex='^(AMT_)'))
        buro.loc[:, buro.columns.str.startswith('DAYS_')] = buro.filter(regex='^DAYS_').replace({365243: np.nan})
        buro.CREDIT_TYPE = buro.CREDIT_TYPE.str.replace(' ', '_')
    
    with timer('create dataset'):
        generate_features([
            BureauActiveCount(), BureauActiveAndTypeProduct(), BureauInterval(),
            BureauEnddate(), BureauAmountPairwise(), BureauProlonged()
        ], args.force)
