import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from feat import Feature, get_arguments, generate_features, SubfileFeature
from utils import timer
from config import *

CREDIT_CAT_COLS = ['NAME_CONTRACT_STATUS']


class InstLatest(Feature):
    def create_features(self):
        df = inst.drop('SK_ID_PREV', axis=1).groupby('SK_ID_CURR').last()
        df.columns = 'inst_' + df.columns + '_latest'
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class InstEwm(SubfileFeature):
    def create_features(self):
        cols = inst.drop(['SK_ID_CURR', 'SK_ID_PREV'], axis=1).columns
        self.df = pd.DataFrame()
        for f in tqdm(cols):
            self.df[f] = inst.groupby('SK_ID_CURR')[f].apply(lambda x: float(x.ewm(com=0.5).mean().tail(1).values))


class InstBasicDirect(SubfileFeature):
    def create_features(self):
        df = inst.drop('SK_ID_PREV', axis=1)
        df['DPD'] = np.maximum(df.DAYS_ENTRY_PAYMENT - df.DAYS_INSTALMENT, 0)
        df['DBD'] = np.maximum(df.DAYS_INSTALMENT - df.DAYS_ENTRY_PAYMENT, 0)
        df['AMT_INSTALMENT_sub_AMT_PAYMENT'] = df.AMT_INSTALMENT - df.AMT_PAYMENT
        df['AMT_INSTALMENT_div_AMT_PAYMENT'] = df.AMT_INSTALMENT / (df.AMT_PAYMENT + 0.1)
        aggs = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum'],
            'DBD': ['max', 'mean', 'sum'],
            'AMT_INSTALMENT_sub_AMT_PAYMENT': ['max', 'mean', 'sum', 'var'],
            'AMT_INSTALMENT_div_AMT_PAYMENT': ['max', 'mean', 'sum', 'var'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
        }
        self.df = df.groupby('SK_ID_CURR').agg(aggs)
        self.df.columns = [e[0] + "_" + e[1] for e in self.df.columns]
        self.df['count'] = df.groupby('SK_ID_CURR').size()


class InstBasicViaPrev(SubfileFeature):
    def create_features(self):
        df = inst.copy()
        df['DPD'] = np.maximum(df.DAYS_ENTRY_PAYMENT - df.DAYS_INSTALMENT, 0)
        df['DBD'] = np.maximum(df.DAYS_INSTALMENT - df.DAYS_ENTRY_PAYMENT, 0)
        df['AMT_INSTALMENT_sub_AMT_PAYMENT'] = df.AMT_INSTALMENT - df.AMT_PAYMENT
        df['AMT_INSTALMENT_div_AMT_PAYMENT'] = df.AMT_INSTALMENT / (df.AMT_PAYMENT + 0.1)
        aggs = {
            'SK_ID_CURR': ['first'],
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum'],
            'DBD': ['max', 'mean', 'sum'],
            'AMT_INSTALMENT_sub_AMT_PAYMENT': ['max', 'mean', 'sum', 'var'],
            'AMT_INSTALMENT_div_AMT_PAYMENT': ['max', 'mean', 'sum', 'var'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
        }
        via = df.groupby('SK_ID_PREV').agg(aggs).reset_index(drop=True)
        via.columns = ['_'.join(f) if f[0] != 'SK_ID_CURR' else f[0] for f in via.columns]
        self.df = via.groupby('SK_ID_CURR').mean()


# class InstBasic(SubfileFeature):
#     inst_ = inst.copy()
#

if __name__ == '__main__':
    args = get_arguments('POS CASH')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)[['SK_ID_CURR']]
        test = pd.read_feather(TEST)[['SK_ID_CURR']]
        inst = pd.read_feather(INST)
    
    with timer('preprocessing'):
        # inst.drop(['SK_ID_PREV'], axis=1, inplace=True)
        inst = inst.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT']).reset_index(drop=True)
        inst.loc[:, inst.columns.str.startswith('AMT_')] = np.log1p(inst.filter(regex='^(AMT_)'))
        inst.loc[:, inst.columns.str.startswith('DAYS_')] = inst.filter(regex='^DAYS_').replace({365243: np.nan})
    
    with timer('create dataset'):
        generate_features([
            InstBasicDirect('inst'),
            InstBasicViaPrev('inst', 'via_prev'),
            InstLatest(),
            InstEwm('inst', 'ewm')
        ], args.force)
