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
Feature.prefix = 'sub_'


class SubActiveAnnuity(SubfileFeature):
    def create_features(self):
        all_df = pd.concat([train.set_index('SK_ID_CURR'), test.set_index('SK_ID_CURR')])
        prev_df = prev.query("DAYS_TERMINATION > 0").groupby('SK_ID_CURR')
        buro_df = buro.query("CREDIT_ACTIVE == 'Active'").groupby('SK_ID_CURR')
        df = pd.DataFrame({
            'main': all_df.AMT_ANNUITY,
            'prev': prev_df.AMT_ANNUITY.sum(),
            'buro': buro_df.AMT_ANNUITY.sum(),
            'income': all_df.AMT_INCOME_TOTAL
        }).fillna(0)
        df['sub_annuity'] = df['prev'] + df['buro']
        df['all_annuity'] = df['sub_annuity'] + df['main']
        df['sub_annuity_to_income'] = df['sub_annuity'] / df['income']
        df['all_annuity_to_income'] = df['all_annuity'] / df['income']
        self.df = df[['sub_annuity', 'all_annuity', 'sub_annuity_to_income', 'all_annuity_to_income']]


class SubActiveCredit(SubfileFeature):
    def create_features(self):
        all_df = pd.concat([train.set_index('SK_ID_CURR'), test.set_index('SK_ID_CURR')])
        prev_df = prev.query("DAYS_TERMINATION > 0").groupby('SK_ID_CURR')
        buro_df = buro.query("CREDIT_ACTIVE == 'Active'").groupby('SK_ID_CURR')
        df = pd.DataFrame({
            'main': all_df.AMT_CREDIT,
            'prev': prev_df.AMT_CREDIT.sum(),
            'buro1': buro_df.AMT_CREDIT_SUM.sum(),
            'buro2': buro_df.AMT_CREDIT_SUM_DEBT.sum(),
            'buro3': buro_df.AMT_CREDIT_SUM_LIMIT.sum(),
            'income': all_df.AMT_INCOME_TOTAL
        }).fillna(0)
        
        for i in range(1, 4):
            df[f'sub_credit{i}'] = df['prev'] + df[f'buro{i}']
            df[f'all_credit{i}'] = df[f'sub_credit{i}'] + df['main']
            df[f'sub_credit{i}_to_income'] = df[f'sub_credit{i}'] / df['income']
            df[f'all_credit{i}_to_income'] = df[f'all_credit{i}'] / df['income']
        
        self.df = df.filter(regex='(^sub_|^all_)')


if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
        prev = pd.read_feather(PREV)
        buro = pd.read_feather(BURO)
    
    with timer('preprocessing'):
        prev = prev.query("NAME_CONTRACT_TYPE != 'XNA'")
    
    with timer('create dataset'):
        generate_features(globals(), args.force)
