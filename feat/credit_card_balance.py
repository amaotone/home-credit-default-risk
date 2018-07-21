import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vega.streak import longest_streak
from feat import get_arguments, generate_features, SubfileFeature
from utils import timer
from config import *


def nonzero_ratio(x):
    return (x.notnull() & (x != 0)).sum() / len(x)


class CreditLatest(SubfileFeature):
    def create_features(self):
        self.df = card.drop('SK_ID_PREV', axis=1).groupby('SK_ID_CURR').last()


class CreditBasicDirect(SubfileFeature):
    def create_features(self):
        g = card.drop('SK_ID_PREV', axis=1).groupby('SK_ID_CURR')
        self.df = g.agg(['min', 'max', 'mean', 'sum', 'std'])
        self.df.columns = [f[0] + "_" + f[1] for f in self.df.columns]
        self.df['count'] = g.size()


class CreditBasicViaPrev(SubfileFeature):
    def create_features(self):
        g = card.groupby('SK_ID_PREV')
        self.df = g.agg(['min', 'max', 'mean', 'sum', 'std'])
        self.df.drop([('SK_ID_CURR', 'max'), ('SK_ID_CURR', 'mean'), ('SK_ID_CURR', 'sum'), ('SK_ID_CURR', 'std')],
                     axis=1, inplace=True)
        self.df.columns = [f[0] + '_' + f[1] if f[0] != 'SK_ID_CURR' else f[0] for f in self.df.columns]
        self.df['count'] = g.size()
        self.df = self.df.groupby('SK_ID_CURR').mean()


class CreditDrawing(SubfileFeature):
    def create_features(self):
        g = card.groupby('SK_ID_CURR')
        drawing_amount_cols = ['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
                               'AMT_DRAWINGS_POS_CURRENT']
        drawing_count_cols = ['CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT',
                              'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM']
        df = pd.DataFrame()
        # longest streak
        for f in tqdm(drawing_amount_cols):
            self.df[f + '_longest_streak'] = g[f].apply(longest_streak)
        self.df['longest_streak_min'] = self.df.filter(regex='_longest_streak$').min(axis=1)
        self.df['longest_streak_mean'] = self.df.filter(regex='_longest_streak$').mean(axis=1)
        self.df['longest_streak_max'] = self.df.filter(regex='_longest_streak$').max(axis=1)
        
        # nonzero ratio
        for f in tqdm(drawing_amount_cols):
            df[f + '_nonzero_ratio'] = g[f].apply(nonzero_ratio)
        self.df['nonzero_ratio_min'] = self.df.filter(regex='_nonzero_ratio$').min(axis=1)
        self.df['nonzero_ratio_mean'] = self.df.filter(regex='_nonzero_ratio$').mean(axis=1)
        self.df['nonzero_ratio_max'] = self.df.filter(regex='_nonzero_ratio$').max(axis=1)
        
        # increase count
        g = card.groupby('SK_ID_PREV')
        for f in tqdm(drawing_count_cols):
            name = f + '_increase_count'
            df[name] = g[f].apply(lambda x: (x > x.shift(1)).sum())


class CreditAmountNegativeCount(SubfileFeature):
    def create_features(self):
        cols = [f for f in card.columns if f.startswith('AMT_') and card[f].isnull().sum() > 0]
        for f in cols:
            self.df[f] = card.groupby('SK_ID_CURR')[f].apply(lambda x: x.isnull().sum())


class CreditNullCount(SubfileFeature):
    def create_features(self):
        df = card.copy()
        df['null_count'] = df.isnull().sum(axis=1)
        self.df['min'] = df.groupby('SK_ID_CURR').null_count.min()
        self.df['mean'] = df.groupby('SK_ID_CURR').null_count.mean()
        self.df['max'] = df.groupby('SK_ID_CURR').null_count.max()


class CreditFirstDelayIndex(SubfileFeature):
    def create_features(self):
        def first_nonzero(x):
            t = np.nonzero(x)[0]
            return 0 if len(t) == 0 else t[0] + 1
        
        g = card.groupby('SK_ID_PREV')
        df = pd.DataFrame()
        df['SK_ID_CURR'] = g.SK_ID_CURR.max()
        df['first_nonzero_SK_DPD'] = g.SK_DPD.apply(first_nonzero)
        df['first_nonzero_SK_DPD_DEF'] = g.SK_DPD_DEF.apply(first_nonzero)
        df['first_nonzero_diff'] = df.first_nonzero_SK_DPD_DEF - df.first_nonzero_SK_DPD
        g = df.reset_index(drop=True).groupby('SK_ID_CURR')
        self.df = pd.concat([
            g.mean().rename(columns=lambda x: x + '_mean'),
            g.max().rename(columns=lambda x: x + '_max'),
        ], axis=1)


if __name__ == '__main__':
    args = get_arguments('POS CASH')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)[['SK_ID_CURR']]
        test = pd.read_feather(TEST)[['SK_ID_CURR']]
        card = pd.read_feather(CREDIT)
    
    with timer('preprocessing'):
        # card.loc[:, card.columns.str.startswith('AMT_') | card.columns.str.startswith('SK_DPD_')] \
        #     = np.log1p(card.filter(regex='^(AMT_|SK_DPD_)'))
        card.replace({'XNA': np.nan, 'XAP': np.nan}, inplace=True)
        card.loc[:, card.columns.str.startswith('DAYS_')] = card.filter(regex='^DAYS_').replace({365243: np.nan})
    
    with timer('create dataset'):
        generate_features([
            CreditLatest('credit', 'latest'),
            CreditBasicDirect('credit'),
            CreditBasicViaPrev('credit', 'via_prev'),
            CreditDrawing('credit_drawing'),
            CreditAmountNegativeCount('credit', 'negative_count'),
            CreditNullCount('credit_null_count'),
            CreditFirstDelayIndex('credit')
        ], args.force)
