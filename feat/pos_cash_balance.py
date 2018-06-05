import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat import Feature, get_arguments, generate_features, SubfileFeature
from utils import timer
from config import *


class PosLatest(Feature):
    def create_features(self):
        df = pos.groupby('SK_ID_CURR').last().drop('SK_ID_PREV', axis=1)
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class PosCount(Feature):
    def create_features(self):
        df = pos.groupby('SK_ID_CURR').SK_ID_PREV.count().to_frame('pos_count')
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class PosDecay(SubfileFeature):
    def create_features(self):
        df = pos[['SK_ID_CURR']].copy()
        for i in [0.8, 0.9, 0.95, 0.99]:
            for f in ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE', 'SK_DPD', 'SK_DPD_DEF']:
                df[f'{f}_decay_{int(i*100)}'] = i ** (-pos.MONTHS_BALANCE) * pos[f]
        self.df = pd.concat([
            df.groupby('SK_ID_CURR').min().rename(columns=lambda x: x + '_min'),
            df.groupby('SK_ID_CURR').mean().rename(columns=lambda x: x + '_mean'),
            df.groupby('SK_ID_CURR').max().rename(columns=lambda x: x + '_max'),
            df.groupby('SK_ID_CURR').sum().rename(columns=lambda x: x + '_sum')
        ], axis=1)


if __name__ == '__main__':
    args = get_arguments('POS CASH')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)[['SK_ID_CURR']]
        test = pd.read_feather(TEST)[['SK_ID_CURR']]
        pos = pd.read_feather(POS)
    
    with timer('preprocessing'):
        pos = pos.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE']).reset_index(drop=True)
        pos.loc[:, pos.columns.str.startswith('SK_DPD')] = np.log1p(pos.filter(regex='^SK_DPD'))
    
    with timer('create dataset'):
        generate_features([
            PosLatest('pos', 'latest'),
            PosCount(),
            PosDecay('pos')
        ], args.force)
