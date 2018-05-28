#TODO: めっちゃ再利用してる
import itertools

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

tqdm.pandas()

import sys
import numpy as np
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from utils import timer

PREFIX = 'pos_'


@timer('null count')
def null_count():
    new = pos.isnull().sum(axis=1).to_frame('null_cnt')
    new['SK_ID_CURR'] = pos.SK_ID_CURR
    return new.groupby('SK_ID_CURR').null_cnt.mean()


@timer('count')
def count():
    return pos.groupby('SK_ID_CURR').SK_ID_PREV.count()


@timer('amount_pairwise')
def amount_pairwise():
    new = pd.DataFrame()
    columns = pos_mean.filter('AMT_').columns.tolist()
    for i, j in tqdm(itertools.combinations(columns, 2)):
        name = f'{i.replace("_mean", "")}_minus_{j.replace("_mean", "")}'
        new[name] = pos[i] - pos[j]
    return new


@timer('days pairwise')
def days_pairwise():
    new = pd.DataFrame()
    columns = pos_mean.filter('DAYS_').columns.tolist()
    for i, j in tqdm(itertools.combinations(columns, 2)):
        name = f'{i.replace("_mean", "")}_minus_{j.replace("_mean", "")}'
        new[name] = pos[i] - pos[j]
    return new


@timer('category')
def category():
    new = pd.DataFrame()
    cat_cols = [f for f in pos.columns if pos[f].dtype == 'object']
    le = LabelEncoder()
    for f in tqdm(cat_cols):
        new[f + '_latest'] = le.fit_transform(pos.groupby('SK_ID_CURR')[f].tail(1).astype(str))
        new[f + '_nunique'] = pos.groupby('SK_ID_CURR')[f].nunique()
        new[f + '_count'] = pos.groupby('SK_ID_CURR')[f].count()


@timer('weekday to sin & cos')
def weekday_to_sin_cos():
    new = pd.DataFrame()
    new['appr_weekday_sin'] = np.sin(pos.WEEKDAY_APPR_PROCESS_START / 7)
    new['appr_weekday_cos'] = np.cos(pos.WEEKDAY_APPR_PROCESS_START / 7)
    return new


if __name__ == '__main__':
    print('load datasets')
    train = pd.read_feather(INPUT / 'application_train.ftr')
    test = pd.read_feather(INPUT / 'application_test.ftr')
    pos = pd.read_feather(INPUT / 'POS_CASH_balance.ftr')
    
    print('sort')
    pos = pos.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE']).reset_index(drop=True)
    
    print('np.log1p(SK_DPD_*)')
    pos.loc[:, pos.columns.str.startswith('SK_DPD')] = np.log1p(pos.filter(regex='^SK_DPD'))
    
    print('mean')
    pos_mean = pos.groupby('SK_ID_CURR').mean()
    # pos_mean.columns = pos_mean.columns + '_mean'
    #
    # print('max')
    # pos_max = pos.groupby('SK_ID_CURR').max()
    # pos_max.columns = pos_max.columns + '_max'
    #
    # print('min')
    # pos_min = pos.groupby('SK_ID_CURR').min()
    # pos_min.columns = pos_min.columns + '_min'
    
    # print('concat')
    # new = pd.concat([pos_min, pos_mean, pos_max], axis=1)
    new = pos_mean
    
    print('calc features')
    new['null_cnt'] = null_count()
    new['count'] = count()
    # new = pd.concat([new, amount_pairwise()])
    # new = pd.concat([new, days_pairwise()])
    # new = pd.concat([new, weekday_to_sin_cos()])
    new = pd.concat([new, category()], axis=1)
    
    new.drop(new.filter(regex='SK_ID_PREV').columns, axis=1, inplace=True)
    
    new.columns = PREFIX + new.columns
    train.merge(new, left_on='SK_ID_CURR', right_index=True, how='left') \
        .filter(regex=PREFIX).to_feather(WORKING / f'{PREFIX}train.ftr')
    test.merge(new, left_on='SK_ID_CURR', right_index=True, how='left') \
        .filter(regex=PREFIX).to_feather(WORKING / f'{PREFIX}test.ftr')