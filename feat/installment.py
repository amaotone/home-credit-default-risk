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

PREFIX = 'inst_'


@timer('null count')
def null_count():
    new = inst.isnull().sum(axis=1).to_frame('null_cnt')
    new['SK_ID_CURR'] = inst.SK_ID_CURR
    return new.groupby('SK_ID_CURR').null_cnt.mean()


@timer('count')
def count():
    return inst.groupby('SK_ID_CURR').SK_ID_PREV.count()


@timer('amount_pairwise')
def amount_pairwise():
    new = pd.DataFrame()
    columns = inst_mean.filter('AMT_').columns.tolist()
    for i, j in tqdm(itertools.combinations(columns, 2)):
        name = f'{i.replace("_mean", "")}_minus_{j.replace("_mean", "")}'
        new[name] = inst[i] - inst[j]
    return new


@timer('days pairwise')
def days_pairwise():
    new = pd.DataFrame()
    columns = inst_mean.filter('DAYS_').columns.tolist()
    for i, j in tqdm(itertools.combinations(columns, 2)):
        name = f'{i.replace("_mean", "")}_minus_{j.replace("_mean", "")}'
        new[name] = inst[i] - inst[j]
    return new


@timer('category')
def category():
    new = pd.DataFrame()
    cat_cols = [f for f in inst.columns if inst[f].dtype == 'object']
    le = LabelEncoder()
    for f in tqdm(cat_cols):
        new[f + '_latest'] = le.fit_transform(inst.groupby('SK_ID_CURR')[f].tail(1).astype(str))
        new[f + '_nunique'] = inst.groupby('SK_ID_CURR')[f].nunique()
        new[f + '_count'] = inst.groupby('SK_ID_CURR')[f].count()


@timer('weekday to sin & cos')
def weekday_to_sin_cos():
    new = pd.DataFrame()
    new['appr_weekday_sin'] = np.sin(inst.WEEKDAY_APPR_PROCESS_START / 7)
    new['appr_weekday_cos'] = np.cos(inst.WEEKDAY_APPR_PROCESS_START / 7)
    return new


if __name__ == '__main__':
    print('load datasets')
    train = pd.read_feather(INPUT / 'application_train.ftr')
    test = pd.read_feather(INPUT / 'application_test.ftr')
    inst = pd.read_feather(INPUT / 'installments_payments.ftr')
    
    print('sort')
    inst = inst.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT']).reset_index(drop=True)
    
    print('handle missing and binary')
    inst.loc[:, inst.columns.str.startswith('DAYS_')] = inst.filter(regex='^DAYS_').replace({365243: np.nan})
    
    print('np.log1p(AMOUNT)')
    inst.loc[:, inst.columns.str.startswith('AMT_')] = np.log1p(inst.filter(regex='^AMT_'))
    
    print('mean')
    inst_mean = inst.groupby('SK_ID_CURR').mean()
    inst_mean.columns = inst_mean.columns + '_mean'
    
    print('max')
    inst_max = inst.groupby('SK_ID_CURR').max()
    inst_max.columns = inst_max.columns + '_max'

    print('min')
    inst_min = inst.groupby('SK_ID_CURR').min()
    inst_min.columns = inst_min.columns + '_min'
    
    print('concat')
    new = pd.concat([inst_min, inst_mean, inst_max], axis=1)
    # new = inst_mean
    
    print('calc features')
    new['null_cnt'] = null_count()
    new['count'] = count()
    new = pd.concat([new, amount_pairwise()], axis=1)
    new = pd.concat([new, days_pairwise()], axis=1)
    # new = pd.concat([new, weekday_to_sin_cos()])
    new = pd.concat([new, category()], axis=1)
    
    new.drop(new.filter(regex='SK_ID_PREV').columns, axis=1, inplace=True)
    
    new.columns = PREFIX + new.columns
    train.merge(new, left_on='SK_ID_CURR', right_index=True, how='left') \
        .filter(regex=PREFIX).to_feather(WORKING / f'{PREFIX}train.ftr')
    test.merge(new, left_on='SK_ID_CURR', right_index=True, how='left') \
        .filter(regex=PREFIX).to_feather(WORKING / f'{PREFIX}test.ftr')