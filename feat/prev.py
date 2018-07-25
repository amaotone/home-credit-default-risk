import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../../spica')
from spica.features.base import Feature, generate_features, get_arguments
from feat import SubfileFeature
from utils import timer
from config import *

Feature.dir = '../working'
Feature.prefix = 'prev'


def target_encoding(df, col, fill=None):
    res = pd.DataFrame(index=prev.index)
    for i, (trn_idx, val_idx) in tqdm(enumerate(cv.split(train.SK_ID_CURR)), total=cv.get_n_splits()):
        idx = train.SK_ID_CURR[trn_idx].values
        ref = df.query('SK_ID_CURR in @idx').groupby(col)['TARGET'].mean()
        res[i] = df.query('SK_ID_CURR not in @idx')[col].replace(ref)
        if fill is None:
            res[i] = res[i].replace(df[col].unique(), df.query('SK_ID_CURR in @idx')['TARGET'].mean()).astype(float)
        else:
            res[i] = res[i].replace(df[col].unique(), fill).astype(float)
    return res.mean(axis=1)


class PrevCount(SubfileFeature):
    def create_features(self):
        self.df['count'] = prev.groupby('SK_ID_CURR').size()


class PrevNullCount(SubfileFeature):
    def create_features(self):
        df = prev.copy()
        df['null_count'] = df.isnull().sum(axis=1)
        self.df['null_count'] = df.groupby('SK_ID_CURR').null_count.mean()


class PrevTarget(SubfileFeature):
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(prev, on='SK_ID_CURR', how='right')
        cols = [f for f in prev.columns if prev[f].dtype == 'object']
        df[cols] = df[cols].replace({'XNA': np.nan, 'XAP': np.nan})
        for f in cols:
            df[f'{f}_target'] = target_encoding(df, f)
        self.df = df.filter(regex='(SK_ID_CURR|_target$)').groupby('SK_ID_CURR').mean()


class PrevFirstTarget(SubfileFeature):
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(prev, on='SK_ID_CURR', how='right')
        df = df.groupby('SK_ID_CURR').head()
        cols = [f for f in prev.columns if prev[f].dtype == 'object']
        df[cols] = df[cols].replace({'XNA': np.nan, 'XAP': np.nan})
        for f in cols:
            df[f'{f}_first_target'] = target_encoding(df, f)
        self.df = df.filter(regex='(SK_ID_CURR|_target$)').groupby('SK_ID_CURR').mean()


class PrevLastTarget(SubfileFeature):
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(prev, on='SK_ID_CURR', how='right')
        df = df.groupby('SK_ID_CURR').tail()
        cols = [f for f in prev.columns if prev[f].dtype == 'object']
        df[cols] = df[cols].replace({'XNA': np.nan, 'XAP': np.nan})
        for f in cols:
            df[f'{f}_last_target'] = target_encoding(df, f)
        self.df = df.filter(regex='(SK_ID_CURR|_target$)').groupby('SK_ID_CURR').mean()


class PrevSellerplaceAreaTarget(SubfileFeature):
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(prev[['SK_ID_CURR', 'SELLERPLACE_AREA']], on='SK_ID_CURR',
                                                   how='right')
        df['bin'] = pd.qcut(df.SELLERPLACE_AREA.replace(-1, np.nan), q=10, labels=False, retbins=False)
        df['SELLERPLACE_AREA_target'] = target_encoding(df, 'bin')
        self.df = df.groupby('SK_ID_CURR').SELLERPLACE_AREA_target.mean().to_frame()


class PrevAmountChange(SubfileFeature):
    def create_features(self):
        df = prev.copy()
        df['credit_sub_app'] = df['AMT_CREDIT'] - df['AMT_APPLICATION']
        df['credit_div_app'] = df['AMT_CREDIT'] / df['AMT_APPLICATION']
        self.df = df.groupby('SK_ID_CURR')[['credit_sub_app', 'credit_div_app']].agg({'min', 'mean', 'max'})
        self.df.columns = [f'{f[0]}_{f[1]}' for f in self.df.columns]


class PrevAmount(SubfileFeature):
    def create_features(self):
        main_cols = ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE']
        prev_cols = ['SK_ID_CURR', 'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE']
        df = train[main_cols].merge(prev[prev_cols], on='SK_ID_CURR', suffixes=['_main', '_prev'], how='right')
        for f in ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE']:
            df[f + '_ratio'] = df[f + '_main'] / df[f + '_prev']
        df['installment_count_calc'] = df['AMT_CREDIT_prev'] / df['AMT_ANNUITY_prev'].replace(0, np.nan)
        df['installment_count_calc_ratio'] = \
            (df['AMT_CREDIT_main'] / df['AMT_ANNUITY_main'].replace(0, np.nan)) / df['installment_count_calc']
        df['credit_ratio'] = df['AMT_CREDIT_prev'] / df['AMT_GOODS_PRICE_prev'].replace(0, np.nan)
        self.df = df.groupby('SK_ID_CURR')[
            ['AMT_ANNUITY_ratio', 'AMT_CREDIT_ratio', 'AMT_GOODS_PRICE_ratio',
             'installment_count_calc', 'installment_count_calc_ratio',
             'credit_ratio']
        ].agg({'min', 'mean', 'max'})
        self.df.columns = ['_'.join(f) for f in self.df.columns]


class PrevDownpayment(SubfileFeature):
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(prev, on='SK_ID_CURR', how='right')
        self.df = df.groupby('SK_ID_CURR').RATE_DOWN_PAYMENT.agg({'mean', 'max'})
        self.df.columns = ['_'.join(f) for f in self.df.columns]


class PrevDayChange(SubfileFeature):
    def create_features(self):
        df = prev.copy()
        df['last_due_change'] = df['DAYS_LAST_DUE'] - df['DAYS_LAST_DUE_1ST_VERSION']
        df['prepayment'] = df['DAYS_TERMINATION'] - df['DAYS_LAST_DUE']
        self.df = df.groupby('SK_ID_CURR')[['last_due_change', 'prepayment']].agg({'min', 'mean', 'max'})
        self.df.columns = ['_'.join(f) for f in self.df.columns]


if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
        prev = pd.read_feather(PREV)
        cv_id = pd.read_feather(INPUT / 'cv_id.ftr')
        cv = PredefinedSplit(cv_id)
    
    # with timer('preprocessing'):
    
    with timer('create dataset'):
        generate_features(globals(), args.force)
