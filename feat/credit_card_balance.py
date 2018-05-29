import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat import Feature, get_arguments, generate_features
from utils import timer
from config import *

CREDIT_CAT_COLS = ['NAME_CONTRACT_STATUS']


class CreditLatest(Feature):
    @property
    def categorical_features(self):
        return ['credit_NAME_CONTRACT_STATUS_latest']
    
    def create_features(self):
        df = card.groupby('SK_ID_CURR').last()
        df.columns = 'credit_' + df.columns + '_latest'
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


if __name__ == '__main__':
    args = get_arguments('POS CASH')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)[['SK_ID_CURR']]
        test = pd.read_feather(TEST)[['SK_ID_CURR']]
        card = pd.read_feather(CREDIT)
    
    with timer('preprocessing'):
        card.drop(['SK_ID_PREV'], axis=1, inplace=True)
        card = card.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE']).reset_index(drop=True)
        card.loc[:, card.columns.str.startswith('AMT_') | card.columns.str.startswith('SK_DPD_')] \
            = np.log1p(card.filter(regex='^(AMT_|SK_DPD_)'))
        card.replace({'XNA': np.nan, 'XAP': np.nan}, inplace=True)
        card.loc[:, card.columns.str.startswith('DAYS_')] = card.filter(regex='^DAYS_').replace({365243: np.nan})
    
    with timer('create dataset'):
        generate_features([CreditLatest()])
