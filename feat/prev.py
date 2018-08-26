import os
import sys

import numpy as np
import pandas as pd
from scipy.special import erfinv
from scipy.stats import rankdata
from sklearn.model_selection import PredefinedSplit
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../../spica')
from spica.features.base import Feature, generate_features, get_arguments
from feat import SubfileFeature
from utils import timer
from config import *

Feature.dir = '../working'
Feature.prefix = 'prev'


def gauss_rank(x, fill=np.nan, eps=0.01):
    mask = x.isnull()
    x = rankdata(x) - 1
    x = erfinv(x / x.max() * 2 * (1 - eps) - (1 - eps))
    x[mask] = fill
    return x


def target_encoding(df, col, fill=None):
    res = pd.DataFrame(index=prev.index)
    for i, (trn_idx, val_idx) in tqdm(enumerate(cv.split(train.SK_ID_CURR)), total=cv.get_n_splits()):
        idx = train.SK_ID_CURR[trn_idx].values
        ref = df.query('SK_ID_CURR in @idx').groupby(col)['TARGET'].mean()
        res[i] = df.query('SK_ID_CURR not in @idx')[col].replace(ref)
        if fill is None:
            res[i] = res[i].replace(df[col].unique(), df.query('SK_ID_CURR in @idx')['TARGET'].mean()).astype(float)
        else:
            res[i] = res[i].replace(df[col].unique(), fill).astype(float)
    return res.mean(axis=1)


def time_weighted_average(df, data_cols, suffix='_time_weighted'):
    data_cols = data_cols if type(data_cols) in (pd.Index, list) else [data_cols]
    
    def wm(df, data_col, weight_col, by_col):
        df['_data_times_weight'] = df[data_col] * df[weight_col]
        df['_weight_where_notnull'] = df[weight_col] * pd.notnull(df[data_col])
        g = df.groupby(by_col)
        res = g['_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
        del df['_data_times_weight'], df['_weight_where_notnull']
        return res
    
    df_ = df.set_index('SK_ID_PREV')
    df_['_weight'] = 1 + np.maximum(prev.set_index('SK_ID_PREV').DAYS_DECISION, -1500) / 2000
    res = []
    for f in tqdm(data_cols):
        res.append(wm(df_, f, '_weight', 'SK_ID_CURR'))
    res = pd.concat(res, axis=1)
    res.columns = [f + suffix for f in data_cols]
    return res


class PrevTimeWeighted(SubfileFeature):
    def create_features(self):
        df = prev.copy()
        cols = df.select_dtypes(exclude=['object']).columns.tolist()
        cols = [f for f in cols if f not in ('SK_ID_CURR', 'SK_ID_PREV')]
        self.df = time_weighted_average(df, cols)


class PrevCount(SubfileFeature):
    def create_features(self):
        self.df['count'] = prev.groupby('SK_ID_CURR').size()


class PrevNullCount(SubfileFeature):
    def create_features(self):
        df = prev.copy()
        df['null_count'] = df.isnull().sum(axis=1)
        self.df['null_count'] = df.groupby('SK_ID_CURR').null_count.mean()
        self.df = pd.concat([self.df, time_weighted_average(df, 'null_count')], axis=1)


class PrevTarget(SubfileFeature):
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(prev, on='SK_ID_CURR', how='right')
        cols = [f for f in prev.columns if prev[f].dtype == 'object']
        df[cols] = df[cols].replace({'XNA': np.nan, 'XAP': np.nan})
        for f in cols:
            df[f'{f}_target'] = target_encoding(df, f)
        cols = df.filter(regex='_target$').columns.tolist()
        self.df = df.groupby('SK_ID_CURR')[cols].mean()
        self.df = pd.concat([self.df, time_weighted_average(df, cols)], axis=1)


class PrevFirstTarget(SubfileFeature):
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(prev, on='SK_ID_CURR', how='right')
        df = df.groupby('SK_ID_CURR').head()
        cols = [f for f in prev.columns if prev[f].dtype == 'object']
        df[cols] = df[cols].replace({'XNA': np.nan, 'XAP': np.nan})
        for f in cols:
            df[f'{f}_first_target'] = target_encoding(df, f)
        cols = df.filter(regex='_target$').columns.tolist()
        self.df = df.groupby('SK_ID_CURR')[cols].mean()
        self.df = pd.concat([self.df, time_weighted_average(df, cols)], axis=1)


class PrevLastTarget(SubfileFeature):
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(prev, on='SK_ID_CURR', how='right')
        df = df.groupby('SK_ID_CURR').tail()
        cols = [f for f in prev.columns if prev[f].dtype == 'object']
        df[cols] = df[cols].replace({'XNA': np.nan, 'XAP': np.nan})
        for f in cols:
            df[f'{f}_last_target'] = target_encoding(df, f)
        cols = df.filter(regex='_target$').columns.tolist()
        self.df = df.groupby('SK_ID_CURR')[cols].mean()
        self.df = pd.concat([self.df, time_weighted_average(df, cols)], axis=1)


class PrevSellerplaceArea(SubfileFeature):
    def create_features(self):
        self.df['SELLERPLACE_AREA_mean'] = prev.groupby('SK_ID_CURR').SELLERPLACE_AREA.mean()
        self.df = pd.concat([self.df, time_weighted_average(prev, 'SELLERPLACE_AREA')], axis=1)


class PrevSellerplaceAreaTarget(SubfileFeature):
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(prev, on='SK_ID_CURR', how='right')
        df['bin'] = pd.qcut(df.SELLERPLACE_AREA.replace(-1, np.nan), q=10, labels=False, retbins=False)
        df['SELLERPLACE_AREA_target'] = target_encoding(df, 'bin')
        self.df = df.groupby('SK_ID_CURR').SELLERPLACE_AREA_target.mean().to_frame()
        self.df = pd.concat([self.df, time_weighted_average(df, 'SELLERPLACE_AREA_target')], axis=1)


class PrevAmountChange(SubfileFeature):
    def create_features(self):
        df = prev.copy()
        df['credit_sub_app'] = df['AMT_CREDIT'] - df['AMT_APPLICATION']
        df['credit_div_app'] = df['AMT_CREDIT'] / df['AMT_APPLICATION']
        self.df = df.groupby('SK_ID_CURR')[['credit_sub_app', 'credit_div_app']].agg({'min', 'mean', 'max'})
        self.df.columns = [f'{f[0]}_{f[1]}' for f in self.df.columns]


class PrevAmount(SubfileFeature):
    def create_features(self):
        main_cols = ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE']
        prev_cols = ['SK_ID_CURR', 'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE']
        df = train[main_cols].merge(prev[prev_cols], on='SK_ID_CURR', suffixes=['_main', '_prev'], how='right')
        for f in ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE']:
            df[f + '_ratio'] = df[f + '_main'] / df[f + '_prev']
        df['installment_count_calc'] = df['AMT_CREDIT_prev'] / df['AMT_ANNUITY_prev'].replace(0, np.nan)
        df['installment_count_calc_ratio'] = \
            (df['AMT_CREDIT_main'] / df['AMT_ANNUITY_main'].replace(0, np.nan)) / df['installment_count_calc']
        df['credit_ratio'] = df['AMT_CREDIT_prev'] / df['AMT_GOODS_PRICE_prev'].replace(0, np.nan)
        self.df = df.groupby('SK_ID_CURR')[
            ['AMT_ANNUITY_ratio', 'AMT_CREDIT_ratio', 'AMT_GOODS_PRICE_ratio',
             'installment_count_calc', 'installment_count_calc_ratio',
             'credit_ratio']
        ].agg({'min', 'mean', 'max'})
        self.df.columns = ['_'.join(f) for f in self.df.columns]


class PrevDownpayment(SubfileFeature):
    def create_features(self):
        df = train[['SK_ID_CURR', 'TARGET']].merge(prev, on='SK_ID_CURR', how='right')
        self.df = df.groupby('SK_ID_CURR')[['RATE_DOWN_PAYMENT']].agg({'mean', 'max'})
        self.df.columns = ['_'.join(f) for f in self.df.columns]


class PrevDayChange(SubfileFeature):
    def create_features(self):
        df = prev.copy()
        df['last_due_change'] = df['DAYS_LAST_DUE'] - df['DAYS_LAST_DUE_1ST_VERSION']
        df['termination_to_last_due'] = df['DAYS_TERMINATION'] - df['DAYS_LAST_DUE']
        df['termination_to_last_due_1st'] = df['DAYS_TERMINATION'] - df['DAYS_LAST_DUE_1ST_VERSION']
        self.df = df.groupby('SK_ID_CURR')[
            ['last_due_change', 'termination_to_last_due', 'termination_to_last_due_1st']
        ].agg({'min', 'mean', 'max'})
        self.df.columns = ['_'.join(f) for f in self.df.columns]
        self.df = pd.concat([self.df, time_weighted_average(df, ['last_due_change', 'termination_to_last_due',
                                                                 'termination_to_last_due_1st'])], axis=1)


class PrevPayed(SubfileFeature):
    def create_features(self):
        df = prev.merge(inst.groupby('SK_ID_PREV').AMT_PAYMENT.sum().to_frame(), left_on='SK_ID_PREV',
                        right_index=True, how='left').query("DAYS_TERMINATION < 0")
        df['payed_ratio'] = df['AMT_PAYMENT'] / df['AMT_CREDIT']
        self.df['payed_amount_ratio_mean'] = df.groupby('SK_ID_CURR').payed_ratio.mean()


class PrevFutureExpires(SubfileFeature):
    def create_features(self):
        ins = inst.groupby('SK_ID_PREV').AMT_PAYMENT.agg(['sum', 'count'])
        ins.columns = ['AMT_PAYMENT', 'installment_count']
        df = prev.merge(ins, left_on='SK_ID_PREV', right_index=True, how='left').query("DAYS_TERMINATION > 0")
        df['payed_count_ratio'] = df['installment_count'] / df['CNT_PAYMENT'].replace(0, np.nan)
        df['remaining_debt'] = df['AMT_CREDIT'] * df['payed_count_ratio']
        g = df.groupby('SK_ID_CURR')
        self.df['remaining_annuity_sum'] = g.AMT_ANNUITY.sum()
        self.df['remaining_debt_sum'] = g.remaining_debt.sum()
        self.df['remaining_count_min'] = g.installment_count.min()
        self.df['remaining_count_mean'] = g.installment_count.mean()
        self.df['remaining_count_sum'] = g.installment_count.max()


class PrevAmountReduced(SubfileFeature):
    def create_features(self):
        df = prev.copy()
        df['amount_reduced'] = prev.AMT_APPLICATION - prev.AMT_CREDIT - prev.AMT_DOWN_PAYMENT
        self.df = df.groupby('SK_ID_CURR')[['amount_reduced']].agg(['min', 'mean', 'max'])
        self.df.columns = [f[0] + '_' + f[1] for f in self.df.columns]


class PrevApplicationHour(SubfileFeature):
    def create_features(self):
        df = prev.copy()
        t = prev.HOUR_APPR_PROCESS_START.value_counts()
        ref = t / t.sum()
        df['HOUR_APPR_PROCESS_START_freq'] = df['HOUR_APPR_PROCESS_START'].replace(ref)
        self.df = df.groupby('SK_ID_CURR')[['HOUR_APPR_PROCESS_START_freq']].agg(['min', 'mean', 'max'])
        self.df.columns = [f[0] + '_' + f[1] for f in self.df.columns]


class PrevInterest(SubfileFeature):
    def create_features(self):
        df = prev.copy()
        df.fillna(0, inplace=True)
        df['interest'] = df['RATE_INTEREST_PRIMARY'] * df['RATE_INTEREST_PRIVILEGED']
        self.df = df.groupby('SK_ID_CURR')[['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'interest']].agg([
            'min', 'mean', 'max'])
        self.df.columns = [f[0] + '_' + f[1] for f in self.df.columns]


class PrevNormalizedAmount(SubfileFeature):
    def create_features(self):
        df = prev.query("NAME_CONTRACT_STATUS == 'Approved' and NAME_CONTRACT_TYPE != 'XNA'")
        loan_types = ['Cash loans', 'Consumer loans', 'Revolving loans']
        amount_cols = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE']
        normed_cols = [f + '_norm' for f in amount_cols]
        for f in amount_cols:
            for tp in loan_types:
                idx = df.query("NAME_CONTRACT_TYPE == @tp").index
                df.loc[idx, f + '_norm'] = gauss_rank(df.loc[idx, f], fill=0)
        self.df = df.groupby('SK_ID_CURR')[normed_cols].agg(['min', 'mean', 'max'])
        self.df.columns = [f[0] + '_' + f[1] for f in self.df.columns]
        self.df = pd.concat([self.df, time_weighted_average(df, normed_cols)], axis=1)


class PrevInstallmentCount(SubfileFeature):
    def create_features(self):
        df = prev.copy()
        df['ideal_inst_cnt'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df = df.merge(inst.groupby('SK_ID_PREV').size().to_frame('act_inst_cnt'),
                      left_on='SK_ID_PREV', right_index=True, how='left')
        df['inst_cnt_ratio'] = df['act_inst_cnt'] / df['ideal_inst_cnt']
        self.df = df.groupby('SK_ID_CURR')[['ideal_inst_cnt', 'act_inst_cnt', 'inst_cnt_ratio']].agg(
            ['min', 'mean', 'max'])
        self.df.columns = [f[0] + '_' + f[1] for f in self.df.columns]
        self.df = pd.concat([self.df, time_weighted_average(df, ['ideal_inst_cnt', 'act_inst_cnt', 'inst_cnt_ratio'])],
                            axis=1)


if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
        prev = pd.read_feather(PREV)
        inst = pd.read_feather(INST)
        cv_id = pd.read_feather(INPUT / 'cv_id.ftr')
        cv = PredefinedSplit(cv_id)
    
    with timer('preprocessing'):
        prev = prev.query("NAME_CONTRACT_TYPE != 'XNA'")
    
    with timer('create dataset'):
        generate_features(globals(), args.force)
