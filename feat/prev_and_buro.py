import os
import sys

import pandas as pd
from sklearn.model_selection import PredefinedSplit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../../spica')
from spica.features.base import Feature, generate_features, get_arguments
from feat import SubfileFeature
from utils import timer
from config import *

Feature.dir = '../working'
Feature.prefix = ''


class SubActiveDebt(SubfileFeature):
    def create_features(self):
        all_df = pd.concat([train.set_index('SK_ID_CURR'), test.set_index('SK_ID_CURR')])
        prev_df = prev.query("DAYS_TERMINATION > 0").groupby('SK_ID_CURR')
        buro_df = buro.query("CREDIT_ACTIVE == 'Active'").groupby('SK_ID_CURR')
        df = pd.DataFrame({
            'main': all_df.AMT_ANNUITY,
            'prev': prev_df.AMT_ANNUITY.sum(),
            'buro': buro_df.AMT_ANNUITY.sum(),
            'income': all_df.AMT_INCOME_TOTAL
        })
        df['sub_annuity'] = df['prev'].fillna(0) + df['buro'].fillna(0)
        df['all_annuity'] = df['sub_annuity'] + df['main'].fillna(0)
        df['sub_annuity_to_income'] = df['sub_annuity'] / df['income']
        df['all_annuity_to_income'] = df['all_annuity'] / df['income']
        self.df = df[['sub_annuity', 'all_annuity', 'sub_annuity_to_income', 'all_annuity_to_income']]


if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
        prev = pd.read_feather(PREV)
        buro = pd.read_feather(BURO)
        cv_id = pd.read_feather(INPUT / 'cv_id.ftr')
        cv = PredefinedSplit(cv_id)
    
    with timer('preprocessing'):
        prev = prev.query("NAME_CONTRACT_TYPE != 'XNA'")
    
    with timer('create dataset'):
        generate_features(globals(), args.force)
