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
Feature.prefix = 'pos'


class PosCount(SubfileFeature):
    def create_features(self):
        p = pos.copy()
        p = p.merge(p.groupby('SK_ID_PREV').size().to_frame('count'),
                    left_on='SK_ID_PREV', right_index=True, how='left')
        self.df = p.groupby('SK_ID_CURR').agg(['min', 'mean', 'max', 'sum'])
        self.df.columns = [f[0] + '_' + f[1] for f in self.df.columns]


class PosDpd(SubfileFeature):
    def create_features(self):
        p = pos.copy()
        p['is_delay'] = (p['SK_DPD'] > 0).astype(int)
        p['is_delay_w_tol'] = (p['SK_DPD_DEF'] > 0).astype(int)
        
        g = p.groupby('SK_ID_PREV').agg({
            'SK_ID_CURR': ['first'],
            'SK_DPD': ['sum', 'mean', 'max'],
            'SK_DPD_DEF': ['sum', 'mean', 'max'],
            'is_delay': ['sum', 'mean'],
            'is_delay_w_tol': ['sum', 'mean']
        })
        g.columns = [f[0] if f[0] == 'SK_ID_CURR' else f[0] + '_' + f[1] for f in g.columns]
        self.df = g.groupby('SK_ID_CURR').mean()


class PosDpdWeighted(SubfileFeature):
    def create_features(self):
        p = pos.copy()
        p['is_delay'] = (p['SK_DPD'] > 0).astype(int)
        p['is_delay_w_tol'] = (p['SK_DPD_DEF'] > 0).astype(int)
        p['CNT_INSTALMENT_PAST'] = p['CNT_INSTALMENT'] - p['CNT_INSTALMENT_FUTURE']
        p['first_weight'] = (0.8 ** p['CNT_INSTALMENT_PAST'])
        p['last_weight'] = (0.8 ** p['CNT_INSTALMENT_FUTURE'])
        
        cols = ['SK_DPD', 'SK_DPD_DEF', 'is_delay', 'is_delay_w_tol']
        for f in cols:
            p[f + '_first_weighted'] = p[f] * p['first_weight']
            p[f + '_last_weighted'] = p[f] * p['last_weight']
        
        cols = p.filter(regex='_weighted').columns
        g = pd.concat([
            p.groupby('SK_ID_PREV')[['SK_ID_CURR']].first(),
            p.groupby('SK_ID_PREV')[cols].sum().rename(columns=lambda x: x + '_sum')
        ], axis=1)
        
        self.df = g.groupby('SK_ID_CURR').agg(['mean', 'max', 'sum'])
        self.df.columns = [f[0] + '_' + f[1] for f in self.df.columns]
        
        # first_weight_sum = lambda x: np.sum(x * p.first_weight[x.index])
        # last_weight_sum = lambda x: np.sum(x * p.last_weight[x.index])
        # first_weight_mean = lambda x: np.average(x, weights=p.first_weight[x.index])
        # last_weight_mean = lambda x: np.average(x, weights=p.last_weight[x.index])
        # g = p.groupby('SK_ID_PREV').agg({
        #     'SK_ID_CURR': ['first'],
        #     'SK_DPD': {'first_weight_sum': first_weight_sum, 'last_weight_sum': last_weight_sum,
        #                'first_weight_mean': first_weight_mean, 'last_weight_mean': last_weight_mean},
        #     'SK_DPD_DEF': {'first_weight_sum': first_weight_sum, 'last_weight_sum': last_weight_sum,
        #                    'first_weight_mean': first_weight_mean, 'last_weight_mean': last_weight_mean},
        #     'is_delay': {'first_weight_sum': first_weight_sum, 'last_weight_sum': last_weight_sum,
        #                  'first_weight_mean': first_weight_mean, 'last_weight_mean': last_weight_mean},
        #     'is_delay_w_tol': {'first_weight_sum': first_weight_sum, 'last_weight_sum': last_weight_sum,
        #                        'first_weight_mean': first_weight_mean, 'last_weight_mean': last_weight_mean},
        # }).rename(lambda x: x[0] if x[0] == 'SK_ID_CURR' else x[0] + '_' + x[1])
        # self.df = g.groupby('SK_ID_CURR').agg(['mean', 'max', 'sum']).rename(columns=lambda x: x[0] + '_' + x[1])


class PosPaidByPeriod(SubfileFeature):
    def create_features(self):
        pos_ = pos.copy()
        
        dfs = []
        df = pos_.groupby('SK_ID_PREV')[['SK_DPD', 'SK_DPD_DEF']].agg(['mean', np.count_nonzero])
        df.columns = [f[0] + '_' + f[1] for f in df.columns]
        dfs.append(df)
        for period in tqdm([3, 5, 10]):
            df = pos_.groupby('SK_ID_PREV').head(period).groupby('SK_ID_PREV')[['SK_DPD', 'SK_DPD_DEF']].agg([
                'mean', np.count_nonzero])
            df.columns = df.columns = [f'first_{period}_{f[0]}_{f[1]}' for f in df.columns]
            dfs.append(df)
            
            df = pos_.groupby('SK_ID_PREV').tail(period).groupby('SK_ID_PREV')[['SK_DPD', 'SK_DPD_DEF']].agg([
                'mean', np.count_nonzero])
            df.columns = df.columns = [f'last_{period}_{f[0]}_{f[1]}' for f in df.columns]
            dfs.append(df)
        
        df = pd.concat(dfs, axis=1)  # type: pd.DataFrame
        df = df.merge(pos_[['SK_ID_PREV', 'SK_ID_CURR']].drop_duplicates(), left_index=True, right_on='SK_ID_PREV',
                      how='left')
        self.df = df.groupby('SK_ID_CURR').mean()


if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
        prev = pd.read_feather(PREV)
        pos = pd.read_feather(POS)
        cv_id = pd.read_feather(INPUT / 'cv_id.ftr')
        cv = PredefinedSplit(cv_id)
    
    # with timer('preprocessing'):
    
    with timer('create dataset'):
        generate_features(globals(), args.force)
