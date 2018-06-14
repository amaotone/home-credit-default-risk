import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat import Feature, get_arguments, generate_features, SubfileFeature
from utils import timer
from config import *


class BureauActiveCount(Feature):
    def create_features(self):
        df = buro.groupby('SK_ID_CURR').CREDIT_ACTIVE.value_counts().unstack().fillna(0).astype(int)
        df.columns = 'bureau_' + df.columns + '_count'
        sum_ = df.sum(axis=1)
        for f in df.columns:
            df[f.replace('_count', '_ratio')] = df[f] / sum_
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class BureauBasic(SubfileFeature):
    def create_features(self):
        num_cols = [f for f in buro.columns if buro[f].dtype != 'object']
        cat_cols = [f for f in buro.columns if buro[f].dtype == 'object']
        self.df = pd.DataFrame()
        for f in num_cols:
            self.df[f'{f}_min'] = buro.groupby('SK_ID_CURR')[f].min()
            self.df[f'{f}_mean'] = buro.groupby('SK_ID_CURR')[f].mean()
            self.df[f'{f}_max'] = buro.groupby('SK_ID_CURR')[f].max()
            self.df[f'{f}_std'] = buro.groupby('SK_ID_CURR')[f].std()
            self.df[f'{f}_sum'] = buro.groupby('SK_ID_CURR')[f].sum()
        for f in cat_cols:
            self.df[f'{f}_nunique'] = buro.groupby('SK_ID_CURR')[f].nunique()
        self.df['count'] = buro.groupby('SK_ID_CURR').DAYS_CREDIT.count()


class BureauActive(SubfileFeature):
    def create_features(self):
        num_cols = [f for f in buro.columns if buro[f].dtype != 'object']
        cat_cols = [f for f in buro.columns if buro[f].dtype == 'object']
        self.df = pd.DataFrame()
        buro_active = buro.query('CREDIT_ACTIVE=="Active"').drop('CREDIT_ACTIVE', axis=1)
        for f in num_cols:
            self.df[f'{f}_min'] = buro_active.groupby('SK_ID_CURR')[f].min()
            self.df[f'{f}_mean'] = buro_active.groupby('SK_ID_CURR')[f].mean()
            self.df[f'{f}_max'] = buro_active.groupby('SK_ID_CURR')[f].max()
            self.df[f'{f}_std'] = buro_active.groupby('SK_ID_CURR')[f].std()
            self.df[f'{f}_sum'] = buro_active.groupby('SK_ID_CURR')[f].sum()
        for f in cat_cols:
            self.df[f'{f}_nunique'] = buro_active.groupby('SK_ID_CURR')[f].nunique()
        self.df['count'] = buro_active.groupby('SK_ID_CURR').DAYS_CREDIT.count()


class BureauClosed(SubfileFeature):
    def create_features(self):
        num_cols = [f for f in buro.columns if buro[f].dtype != 'object']
        cat_cols = [f for f in buro.columns if buro[f].dtype == 'object']
        self.df = pd.DataFrame()
        buro_closed = buro.query('CREDIT_ACTIVE=="Closed"').drop('CREDIT_ACTIVE', axis=1)
        for f in num_cols:
            self.df[f'{f}_min'] = buro_closed.groupby('SK_ID_CURR')[f].min()
            self.df[f'{f}_mean'] = buro_closed.groupby('SK_ID_CURR')[f].mean()
            self.df[f'{f}_max'] = buro_closed.groupby('SK_ID_CURR')[f].max()
            self.df[f'{f}_std'] = buro_closed.groupby('SK_ID_CURR')[f].std()
            self.df[f'{f}_sum'] = buro_closed.groupby('SK_ID_CURR')[f].sum()
        for f in cat_cols:
            self.df[f'{f}_nunique'] = buro_closed.groupby('SK_ID_CURR')[f].nunique()
        self.df['count'] = buro_closed.groupby('SK_ID_CURR').DAYS_CREDIT.count()


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
        # buro_bal = pd.read_feather(BURO_BAL)
    
    with timer('preprocessing'):
        buro.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
        buro = buro.sort_values(['SK_ID_CURR', 'DAYS_CREDIT']).reset_index(drop=True)
        buro.loc[:, buro.columns.str.startswith('AMT_')] = np.log1p(buro.filter(regex='^(AMT_)'))
        buro.loc[:, buro.columns.str.startswith('DAYS_')] = buro.filter(regex='^DAYS_').replace({365243: np.nan})
        buro.CREDIT_TYPE = buro.CREDIT_TYPE.str.replace(' ', '_')
    
    with timer('create dataset'):
        generate_features([
            BureauActiveCount(), BureauActiveAndTypeProduct(), BureauInterval(),
            BureauEnddate(), BureauAmountPairwise(), BureauProlonged(),
            BureauBasic('buro'), BureauActive('buro_active'), BureauClosed('buro_closed')
        ], args.force)
