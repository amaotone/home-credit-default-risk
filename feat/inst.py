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
Feature.prefix = 'inst'


def target_encoding(df, col, fill=None):
    res = pd.DataFrame(index=df.index)
    for i, (trn_idx, val_idx) in tqdm(enumerate(cv.split(train.SK_ID_CURR)), total=cv.get_n_splits()):
        idx = train.SK_ID_CURR[trn_idx].values
        ref = df.query('SK_ID_CURR in @idx').groupby(col)['TARGET'].mean()
        res[i] = df.query('SK_ID_CURR not in @idx')[col].replace(ref)
        if fill is None:
            res[i] = res[i].replace(df[col].unique(), df.query('SK_ID_CURR in @idx')['TARGET'].mean()).astype(float)
        else:
            res[i] = res[i].replace(df[col].unique(), fill).astype(float)
    return res.mean(axis=1)


class InstBasic(SubfileFeature):
    def create_features(self):
        df = inst.copy()
        df['DPD'] = np.maximum(df.DAYS_ENTRY_PAYMENT - df.DAYS_INSTALMENT, 0)
        df['DBD'] = np.maximum(df.DAYS_INSTALMENT - df.DAYS_ENTRY_PAYMENT, 0)
        aggs = {
            'SK_ID_CURR': ['first'],
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'sum'],
            'DBD': ['max', 'sum'],
            'AMT_INSTALMENT': ['sum'],
            'AMT_PAYMENT': ['sum'],
        }
        via = df.groupby('SK_ID_PREV').agg(aggs).reset_index(drop=True)
        via.columns = ['_'.join(f) if f[0] != 'SK_ID_CURR' else f[0] for f in via.columns]
        via['DPD_nonzero'] = df.query("DPD > 0").groupby('SK_ID_PREV').size().fillna(0)
        via['DBD_nonzero'] = df.query("DBD > 0").groupby('SK_ID_PREV').size().fillna(0)
        self.df = via.groupby('SK_ID_CURR').agg({'mean', 'max'})
        self.df.columns = [f[0] + '_' + f[1] for f in self.df.columns]


class InstTarget(SubfileFeature):
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(inst, on='SK_ID_CURR', how='right')
        df['bin'] = pd.qcut(inst.NUM_INSTALMENT_VERSION, 500, duplicates='drop', labels=False, retbins=False)
        df['NUM_INSTALMENT_VERSION_target'] = target_encoding(df, 'bin')
        self.df['NUM_INSTALMENT_VERSION_target'] = df.groupby('SK_ID_CURR').NUM_INSTALMENT_VERSION_target.mean()


class InstAmount(SubfileFeature):
    def create_features(self):
        df = inst.copy()
        df['payment_to_schedule_ratio'] = np.log1p(df.AMT_PAYMENT) - np.log1p(df.AMT_INSTALMENT)
        df = df.groupby('SK_ID_PREV').mean()
        df['AMT_PAYMENT_variation_ratio'] = inst.groupby('SK_ID_PREV').AMT_PAYMENT.std() / inst.groupby(
            'SK_ID_PREV').AMT_PAYMENT.mean()
        df.reset_index(drop=True, inplace=True)
        self.df = df.groupby('SK_ID_CURR')[[
            'payment_to_schedule_ratio', 'AMT_PAYMENT_variation_ratio'
        ]].agg({'min', 'mean', 'max'})
        self.df.columns = [f[0] + '_' + f[1] for f in self.df.columns]


class InstDelayedAndPrepayed(SubfileFeature):
    def create_features(self):
        df = prev.copy().set_index('SK_ID_PREV')
        ins = inst.copy()
        ins['days_diff'] = ins.DAYS_INSTALMENT - ins.DAYS_ENTRY_PAYMENT
        g = ins.query('days_diff > 0').groupby('SK_ID_PREV')
        df['delayed_count'] = g.size()
        df['delayed_days_sum'] = g.days_diff.sum()
        df['delayed_days_mean'] = g.days_diff.mean()
        df['first_delayed_inst_num'] = g.NUM_INSTALMENT_NUMBER.first()
        df['delayed_AMT_ANNUITY'] = df.query("delayed_count > 0").AMT_ANNUITY
        df['delayed_AMT_CREDIT'] = df.query("delayed_count > 0").AMT_CREDIT
        
        g = ins.query('days_diff < 0').groupby('SK_ID_PREV')
        df['prepayed_count'] = g.size()
        df['prepayed_days_sum'] = -g.days_diff.sum()
        df['prepayed_days_mean'] = -g.days_diff.mean()
        df['first_prepayed_inst_num'] = g.NUM_INSTALMENT_NUMBER.first()
        df['prepayed_AMT_ANNUITY'] = df.query("prepayed_count > 0").AMT_ANNUITY
        df['prepayed_AMT_CREDIT'] = df.query("prepayed_count > 0").AMT_CREDIT
        
        aggs = {
            'delayed_count': ['mean', 'max', 'sum', 'count'],
            'delayed_days_sum': ['mean', 'max', 'sum'],
            'delayed_days_mean': ['mean', 'max', 'sum'],
            'first_delayed_inst_num': ['min', 'mean', 'max'],
            'delayed_AMT_ANNUITY': ['min', 'mean'],
            'delayed_AMT_CREDIT': ['min', 'mean'],
            'prepayed_count': ['mean', 'max', 'sum', 'count'],
            'prepayed_days_sum': ['mean', 'max', 'sum'],
            'prepayed_days_mean': ['mean', 'max', 'sum'],
            'first_prepayed_inst_num': ['min', 'mean', 'max'],
            'prepayed_AMT_ANNUITY': ['mean', 'max'],
            'prepayed_AMT_CREDIT': ['mean', 'max'],
        }
        all_df = df.groupby('SK_ID_CURR').agg(aggs)
        all_df.columns = ['all_' + f[0] + '_' + f[1] for f in all_df.columns]
        past_df = df.query("DAYS_TERMINATION < 0").groupby('SK_ID_CURR').agg(aggs)
        past_df.columns = ['past_' + f[0] + '_' + f[1] for f in past_df.columns]
        future_df = df.query("DAYS_TERMINATION > 0").groupby('SK_ID_CURR').agg(aggs)
        future_df.columns = ['future_' + f[0] + '_' + f[1] for f in future_df.columns]
        self.df = pd.concat([all_df, past_df, future_df], axis=1)


class InstAmountToApplication(Feature):
    prefix = ''
    
    def create_features(self):
        trn = pd.read_feather(WORKING / 'inst_delayed_and_prepayed_train.ftr')
        tst = pd.read_feather(WORKING / 'inst_delayed_and_prepayed_test.ftr')
        cols = trn.filter(regex='AMT_CREDIT').columns
        for f in cols:
            self.train[f'{f}_to_application'] = train['AMT_CREDIT'] / trn[f]
            self.test[f'{f}_to_application'] = test['AMT_CREDIT'] / tst[f]
        cols = trn.filter(regex='AMT_ANNUITY').columns
        for f in cols:
            self.train[f'{f}_to_application'] = train['AMT_ANNUITY'] / trn[f]
            self.test[f'{f}_to_application'] = test['AMT_ANNUITY'] / tst[f]


class InstPaidByPeriod(SubfileFeature):
    def create_features(self):
        inst_ = inst.copy()
        inst_['days_diff'] = inst_['DAYS_ENTRY_PAYMENT'] - inst_['DAYS_INSTALMENT']
        inst_['paid_late'] = np.maximum(inst_['days_diff'], 0)
        inst_['paid_early'] = np.minimum(inst_['days_diff'], 0).abs()
        df = inst_.groupby('SK_ID_PREV')[['days_diff', 'paid_late', 'paid_early']].mean()
        df.columns += '_mean'
        
        for period in tqdm([3, 5, 10]):
            df[f'paid_late_in_first_{period}_mean'] = \
                inst_.groupby('SK_ID_PREV').head(period).groupby('SK_ID_PREV').paid_late.mean()
            df[f'paid_late_in_last_{period}_mean'] = \
                inst_.groupby('SK_ID_PREV').tail(period).groupby('SK_ID_PREV').paid_late.mean()
            df[f'paid_early_in_first_{period}_mean'] = \
                inst_.groupby('SK_ID_PREV').head(period).groupby('SK_ID_PREV').paid_early.mean()
            df[f'paid_early_in_last_{period}_mean'] = \
                inst_.groupby('SK_ID_PREV').tail(period).groupby('SK_ID_PREV').paid_early.mean()
            df[f'paid_late_in_{period}_mean_diff'] = \
                df[f'paid_late_in_last_{period}_mean'] - df[f'paid_late_in_first_{period}_mean']
            df[f'paid_early_in_{period}_mean_diff'] = \
                df[f'paid_early_in_last_{period}_mean'] - df[f'paid_early_in_first_{period}_mean']

        df = df.merge(inst_[['SK_ID_PREV', 'SK_ID_CURR']].drop_duplicates(), left_index=True, right_on='SK_ID_PREV',
                      how='left')
        self.df = df.groupby('SK_ID_CURR').mean()

if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
        prev = pd.read_feather(PREV)
        inst = pd.read_feather(INST)
        cv_id = pd.read_feather(INPUT / 'cv_id.ftr')
        cv = PredefinedSplit(cv_id)
    
    # with timer('preprocessing'):
    
    with timer('create dataset'):
        generate_features(globals(), args.force)
