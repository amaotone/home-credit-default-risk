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

PREFIX = 'credit_'


@timer('null count')
def null_count():
    new = credit.isnull().sum(axis=1).to_frame('null_cnt')
    new['SK_ID_CURR'] = credit.SK_ID_CURR
    return new.groupby('SK_ID_CURR').null_cnt.mean()


@timer('count')
def count():
    return credit.groupby('SK_ID_CURR').SK_ID_PREV.count()


@timer('amount_pairwise')
def amount_pairwise():
    new = pd.DataFrame()
    columns = credit_mean.filter('AMT_').columns.tolist()
    for i, j in tqdm(itertools.combinations(columns, 2)):
        name = f'{i.replace("_mean", "")}_minus_{j.replace("_mean", "")}'
        new[name] = credit[i] - credit[j]
    return new


@timer('days pairwise')
def days_pairwise():
    new = pd.DataFrame()
    columns = credit_mean.filter('DAYS_').columns.tolist()
    for i, j in tqdm(itertools.combinations(columns, 2)):
        name = f'{i.replace("_mean", "")}_minus_{j.replace("_mean", "")}'
        new[name] = credit[i] - credit[j]
    return new


@timer('category')
def category():
    new = pd.DataFrame()
    cat_cols = [f for f in credit.columns if credit[f].dtype == 'object']
    le = LabelEncoder()
    for f in tqdm(cat_cols):
        new[f + '_latest'] = le.fit_transform(credit.groupby('SK_ID_CURR')[f].tail(1).astype(str))
        new[f + '_nunique'] = credit.groupby('SK_ID_CURR')[f].nunique()
        new[f + '_count'] = credit.groupby('SK_ID_CURR')[f].count()


@timer('weekday to sin & cos')
def weekday_to_sin_cos():
    new = pd.DataFrame()
    new['appr_weekday_sin'] = np.sin(credit.WEEKDAY_APPR_PROCESS_START / 7)
    new['appr_weekday_cos'] = np.cos(credit.WEEKDAY_APPR_PROCESS_START / 7)
    return new


if __name__ == '__main__':
    print('load datasets')
    train = pd.read_feather(INPUT / 'application_train.ftr')
    test = pd.read_feather(INPUT / 'application_test.ftr')
    credit = pd.read_feather(INPUT / 'credit_card_balance.ftr')
    
    print('sort')
    credit = credit.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE']).reset_index(drop=True)
    
    print('np.log1p(AMT_|SK_DPD_*)')
    credit.loc[:, credit.columns.str.startswith('AMT_')] = np.log1p(credit.filter(regex='^AMT_'))
    credit.loc[:, credit.columns.str.startswith('SK_DPD')] = np.log1p(credit.filter(regex='^SK_DPD'))
    
    print('mean')
    credit_mean = credit.groupby('SK_ID_CURR').mean()
    # credit_mean.columns = credit_mean.columns + '_mean'
    #
    # print('max')
    # credit_max = credit.groupby('SK_ID_CURR').max()
    # credit_max.columns = credit_max.columns + '_max'
    #
    # print('min')
    # credit_min = credit.groupby('SK_ID_CURR').min()
    # credit_min.columns = credit_min.columns + '_min'
    
    # print('concat')
    # new = pd.concat([credit_min, credit_mean, credit_max], axis=1)
    new = credit_mean
    
    print('calc features')
    new['null_cnt'] = null_count()
    new['count'] = count()
    new = pd.concat([new, amount_pairwise()], axis=1)
    # new = pd.concat([new, days_pairwise()])
    # new = pd.concat([new, weekday_to_sin_cos()])
    new = pd.concat([new, category()], axis=1)
    
    new.drop(new.filter(regex='SK_ID_PREV').columns, axis=1, inplace=True)
    
    new.columns = PREFIX + new.columns
    train.merge(new, left_on='SK_ID_CURR', right_index=True, how='left') \
        .filter(regex=PREFIX).to_feather(WORKING / f'{PREFIX}train.ftr')
    test.merge(new, left_on='SK_ID_CURR', right_index=True, how='left') \
        .filter(regex=PREFIX).to_feather(WORKING / f'{PREFIX}test.ftr')