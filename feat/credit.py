import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../../spica')
from spica.features.base import Feature, generate_features, get_arguments
from feat import SubfileFeature
from utils import timer
from config import *

Feature.dir = '../working'
Feature.prefix = 'credit'


class CreditAggByLimit(SubfileFeature):
    def create_features(self):
        g = card.query("AMT_BALANCE != 0").groupby(['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL'])
        df = g.agg({
            'SK_ID_CURR': ['first'],
            'AMT_BALANCE': ['max'],
            'AMT_DRAWINGS_CURRENT': ['max', 'mean'],
            'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean'],
            'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean'],
            'AMT_INST_MIN_REGULARITY': ['max'],
            'AMT_PAYMENT_CURRENT': ['max', 'mean'],
            'AMT_RECEIVABLE_PRINCIPAL': ['mean'],
            'AMT_RECIVABLE': ['mean'],
            'AMT_TOTAL_RECEIVABLE': ['mean'],
            'CNT_DRAWINGS_ATM_CURRENT': ['sum'],
            'CNT_DRAWINGS_CURRENT': ['sum'],
            'CNT_DRAWINGS_POS_CURRENT': ['sum'],
            'CNT_INSTALMENT_MATURE_CUM': ['sum'],
            'SK_DPD': ['mean', 'max', np.count_nonzero],
            'SK_DPD_DEF': ['mean', 'max', np.count_nonzero]
        }).reset_index()
        df.columns = [f[0] + '_' + f[1] if f[1] != '' else f[0] for f in df.columns]
        df.rename(columns={'SK_ID_CURR_first': 'SK_ID_CURR'}, inplace=True)
        amt_cols = df.filter(regex='^AMT_').columns.tolist()
        for i, j in itertools.combinations(amt_cols, 2):
            df[f'{i}_div_{j}'] = df[i] / df[j]
        df.replace(np.inf, np.nan, inplace=True)
        self.df = df.groupby('SK_ID_CURR').mean()


if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
        prev = pd.read_feather(PREV)
        card = pd.read_feather(CREDIT)
        cv_id = pd.read_feather(INPUT / 'cv_id.ftr')
        cv = PredefinedSplit(cv_id)
    
    # with timer('preprocessing'):
    
    with timer('create dataset'):
        generate_features(globals(), args.force)
