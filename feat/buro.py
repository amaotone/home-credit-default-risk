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
Feature.prefix = 'buro'


def target_encoding(df, col, fill=None):
    res = pd.DataFrame(index=buro.index)
    for i, (trn_idx, val_idx) in tqdm(enumerate(cv.split(train.SK_ID_CURR)), total=cv.get_n_splits()):
        idx = train.SK_ID_CURR[trn_idx].values
        ref = df.query('SK_ID_CURR in @idx').groupby(col)['TARGET'].mean()
        res[i] = df.query('SK_ID_CURR not in @idx')[col].replace(ref)
        if fill is None:
            res[i] = res[i].replace(buro[col].unique(), res[i].mean()).astype(float)
        else:
            res[i] = res[i].replace(buro[col].unique(), fill).astype(float)
    return res.mean(axis=1)


class BuroCount(SubfileFeature):
    def create_features(self):
        self.df = buro.groupby('SK_ID_CURR').size().to_frame('count')


class BuroNullCount(SubfileFeature):
    def create_features(self):
        df = buro.copy()
        df['null_count'] = buro.isnull().sum(axis=1)
        self.df = df.groupby('SK_ID_CURR').null_count.mean().to_frame()


class BuroBalanceCount(SubfileFeature):
    def create_features(self):
        t = buro[['SK_ID_BUREAU']].merge(bb, on='SK_ID_BUREAU', how='left')
        t = t.groupby('SK_ID_BUREAU').MONTHS_BALANCE.count().to_frame('balance_count')
        df = buro.merge(t, left_on='SK_ID_BUREAU', right_index=True, how='left')
        self.df = df.groupby('SK_ID_CURR').balance_count.mean().to_frame()


class BuroActive(SubfileFeature):
    """activeの個数"""
    
    def create_features(self):
        df = buro.query('CREDIT_ACTIVE == "Active"').copy()
        self.df['active_count'] = df.groupby('SK_ID_CURR').size()
        self.df['active_amount_total'] = df.groupby('SK_ID_CURR').AMT_CREDIT_SUM.sum()
        self.df['active_overdue_count'] = df.query('DAYS_CREDIT_ENDDATE < 0').groupby('SK_ID_CURR').size()


class BuroTarget(SubfileFeature):
    """targetを逆マージしてencoding"""
    
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(buro, on='SK_ID_CURR', how='right')
        df['credit_type_target'] = target_encoding(df, 'CREDIT_TYPE', 0)
        df['credit_active_target'] = target_encoding(df, 'CREDIT_ACTIVE')
        df['credit_currency_target'] = target_encoding(df, 'CREDIT_CURRENCY')
        self.df = df.filter(regex='(SK_ID_CURR|_target$)').groupby('SK_ID_CURR').mean()


class BuroOverdue(SubfileFeature):
    def create_features(self):
        df = buro.copy()
        df['overdue'] = buro.CREDIT_DAY_OVERDUE > 0
        self.df['overdue_day_count'] = df.groupby('SK_ID_CURR').overdue.sum()
        self.df['overdue_day_mean'] = df.query('CREDIT_ACTIVE == "Active"').groupby(
            'SK_ID_CURR').CREDIT_DAY_OVERDUE.mean()
        self.df['overdue_amount_mean'] = df.groupby('SK_ID_CURR').AMT_CREDIT_MAX_OVERDUE.fillna(0).mean()


class BuroPrepayment(SubfileFeature):
    def create_features(self):
        df = buro.copy()
        df['prepayment'] = buro.DAYS_ENDDATE_FACT - buro.DAYS_CREDIT_ENDDATE
        self.df = df.groupby('SK_ID_CURR').prepayment.agg({'min', 'mean', 'max', 'median'})
        self.df.columns = 'prepayment_' + self.df.columns


class BuroDaysRatio(Feature):
    def create_features(self):
        day_cols = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
        trn = train[['SK_ID_CURR'] + day_cols]
        tst = test[['SK_ID_CURR'] + day_cols]
        df = buro[['SK_ID_CURR']].copy()
        df['payment_terms'] = buro['DAYS_CREDIT_ENDDATE'] - buro['DAYS_CREDIT']
        df = df.groupby('SK_ID_CURR').payment_terms.sum().fillna(0).to_frame()
        trn = trn.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')
        tst = tst.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')
        for f in day_cols:
            name = f'payment_terms_div_{f}'
            self.train[name] = trn['payment_terms'] / trn[f]
            self.test[name] = tst['payment_terms'] / tst[f]


class BuroFutureExpires(SubfileFeature):
    prefix = 'buro_future_expire'
    
    def create_features(self):
        future = buro.query('DAYS_CREDIT_ENDDATE > 0').copy()
        g = future.groupby('SK_ID_CURR')
        self.df['days_min'] = g.DAYS_CREDIT_ENDDATE.min()
        self.df['amount_sum'] = g.AMT_CREDIT_SUM.sum()
        self.df['debt_sum'] = g.AMT_CREDIT_SUM_DEBT.sum()
        self.df['annuity_sum'] = g.AMT_ANNUITY.sum()
        future['rem_installments'] = future['AMT_CREDIT_SUM_DEBT'] / future['AMT_ANNUITY']
        future.loc[future['AMT_ANNUITY'] == 0, 'rem_installments'] = np.nan
        self.df['rem_installments'] = future.groupby('SK_ID_CURR').rem_installments.mean()


class BuroPeriodOverlap(SubfileFeature):
    """めっちゃ時間かかる"""
    prefix = 'buro_period_overlap'
    
    def create_features(self):
        def count(x):
            idx = {v: i for i, v in enumerate(np.unique(x[['DAYS_CREDIT', 'DAYS_ENDDATE_FACT']].values))}
            res = np.zeros(len(idx))
            for _, s in x.iterrows():
                res[idx[s.DAYS_CREDIT]] += 1
                res[idx[s.DAYS_ENDDATE_FACT]] -= 1
            return res.cumsum().max()
        
        def amount_sum(x):
            idx = {v: i for i, v in enumerate(np.unique(x[['DAYS_CREDIT', 'DAYS_ENDDATE_FACT']].values))}
            res = np.zeros(len(idx))
            for _, s in x.iterrows():
                res[idx[s.DAYS_CREDIT]] += s.AMT_CREDIT_SUM
                res[idx[s.DAYS_ENDDATE_FACT]] -= s.AMT_CREDIT_SUM
            return res.cumsum().max()
        
        df = buro[['SK_ID_CURR', 'DAYS_CREDIT', 'DAYS_ENDDATE_FACT', 'AMT_CREDIT_SUM']].fillna(0)
        print('count max')
        self.df['count_max'] = df.groupby('SK_ID_CURR').apply(count)
        print('amount sum max')
        self.df['amount_sum_max'] = df.groupby('SK_ID_CURR').apply(amount_sum)


class BuroProlong(SubfileFeature):
    def create_features(self):
        self.df['prolong_count'] = buro.groupby('SK_ID_CURR').CNT_CREDIT_PROLONG.sum()
        self.df['prolong_size'] = buro.query("CNT_CREDIT_PROLONG > 0").groupby('SK_ID_CURR').CNT_CREDIT_PROLONG.size()


if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
        buro = pd.read_feather(BURO)
        bb = pd.read_feather(BURO_BAL)
        cv_id = pd.read_feather(INPUT / 'cv_id.ftr')
        cv = PredefinedSplit(cv_id)
    
    # with timer('preprocessing'):
    
    with timer('create dataset'):
        generate_features(globals(), args.force)
