import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat import Feature, get_arguments, generate_features
from utils import timer
from config import *

PREV_CAT_COLS = ['NAME_CONTRACT_STATUS', 'WEEKDAY_APPR_PROCESS_START',
                 # 'HOUR_APPR_PROCESS_START',
                 'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY', 'NAME_CASH_LOAN_PURPOSE',
                 'NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON',
                 'NAME_PAYMENT_TYPE', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO',
                 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'SELLERPLACE_AREA', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP',
                 'PRODUCT_COMBINATION', 'NFLAG_INSURED_ON_APPROVAL']


class PrevLatest(Feature):
    @property
    def categorical_features(self):
        return [f'prev_{f}_latest' for f in PREV_CAT_COLS]
    
    def create_features(self):
        df = prev.groupby('SK_ID_CURR').last()
        df.columns = 'prev_' + df.columns + '_latest'
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class PrevLastApproved(Feature):
    @property
    def categorical_features(self):
        return [f'prev_{f}_last_approved' for f in PREV_CAT_COLS \
                if f not in ('NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON')]
    
    def create_features(self):
        df = prev.query("NAME_CONTRACT_STATUS=='Approved'").groupby('SK_ID_CURR').last() \
            .drop(['NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON'], axis=1)
        df.columns = 'prev_' + df.columns + '_last_approved'
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class PrevCount(Feature):
    pass


if __name__ == '__main__':
    args = get_arguments('POS CASH')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)[['SK_ID_CURR']]
        test = pd.read_feather(TEST)[['SK_ID_CURR']]
        prev = pd.read_feather(PREV)
    
    with timer('preprocessing'):
        prev.drop(['SK_ID_PREV'], axis=1, inplace=True)
        prev = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION']).reset_index(drop=True)
        prev.loc[:, prev.columns.str.startswith('AMT_')] = np.log1p(prev.filter(regex='^AMT_'))
        prev.replace({'XNA': np.nan, 'XAP': np.nan}, inplace=True)
        prev.loc[:, prev.columns.str.startswith('DAYS_')] = prev.filter(regex='^DAYS_').replace({365243: np.nan})
    
    with timer('create dataset'):
        generate_features([PrevLatest(), PrevLastApproved()], args.force)
