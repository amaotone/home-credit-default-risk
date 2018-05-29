import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat import Feature, get_arguments, generate_features
from utils import timer
from config import *

CREDIT_CAT_COLS = ['NAME_CONTRACT_STATUS']


class InstLatest(Feature):
    def create_features(self):
        df = inst.groupby('SK_ID_CURR').last()
        df.columns = 'inst_' + df.columns + '_latest'
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


if __name__ == '__main__':
    args = get_arguments('POS CASH')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)[['SK_ID_CURR']]
        test = pd.read_feather(TEST)[['SK_ID_CURR']]
        inst = pd.read_feather(INST)
    
    with timer('preprocessing'):
        inst.drop(['SK_ID_PREV'], axis=1, inplace=True)
        inst = inst.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT']).reset_index(drop=True)
        inst.loc[:, inst.columns.str.startswith('AMT_')] = np.log1p(inst.filter(regex='^(AMT_)'))
        inst.loc[:, inst.columns.str.startswith('DAYS_')] = inst.filter(regex='^DAYS_').replace({365243: np.nan})
    
    with timer('create dataset'):
        generate_features([InstLatest()])
