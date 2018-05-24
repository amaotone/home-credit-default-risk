import itertools
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *


@timer('preprocessing')
def preprocessing():
    global train, test, income_cols
    
    print('handle binary + missing')
    # https://www.kaggle.com/c/home-credit-default-risk/discussion/57248
    train.iloc[:, 1:] = train.iloc[:, 1:].replace(
        {'Y': 1, 'N': 0, 'M': 0, 'F': 1, 'XNA': np.nan, 'XAP': np.nan, 365243: np.nan})
    test.iloc[:, 1:] = test.iloc[:, 1:].replace(
        {'Y': 1, 'N': 0, 'M': 0, 'F': 1, 'XNA': np.nan, 'XAP': np.nan, 365243: np.nan})
    
    print('np.log1p(INCOME)')
    train[list(income_cols.values())] = np.log1p(train[list(income_cols.values())])
    test[list(income_cols.values())] = np.log1p(test[list(income_cols.values())])


@timer('days_to_years')
def days_to_years():
    new_train, new_test = pd.DataFrame(), pd.DataFrame()
    for k, v in tqdm(days_cols.items()):
        name = f'years_{k}'
        new_train[name] = train[v] // 365.25
        new_test[name] = test[v] // 365.25
    
    filename = 'main_days_to_years'
    new_train.to_feather(WORKING / f'{filename}_train.ftr')
    new_test.to_feather(WORKING / f'{filename}_test.ftr')
    return new_train, new_test


@timer('days_pairwise')
def days_pairwise():
    new_train, new_test = pd.DataFrame(), pd.DataFrame()
    for (i, j) in tqdm(itertools.combinations(days_cols.keys(), 2)):
        name = f'days_{i}_minus_{j}'
        new_train[name] = train[days_cols[i]] - train[days_cols[j]]
        new_test[name] = test[days_cols[i]] - test[days_cols[j]]
    
    filename = 'main_days_pairwise'
    new_train.to_feather(WORKING / f'{filename}_train.ftr')
    new_test.to_feather(WORKING / f'{filename}_test.ftr')
    return new_train, new_test


@timer('money_pairwise')
def money_pairwise():
    new_train, new_test = pd.DataFrame(), pd.DataFrame()
    for (i, j) in tqdm(itertools.combinations(income_cols.keys(), 2)):
        name = f'income_{i}_minus_{j}'
        new_train[name] = train[income_cols[i]] - train[income_cols[j]]
        new_test[name] = test[income_cols[i]] - test[income_cols[j]]
    
    filename = 'main_money_pairwise'
    new_train.to_feather(WORKING / f'{filename}_train.ftr')
    new_test.to_feather(WORKING / f'{filename}_test.ftr')
    return new_train, new_test


if __name__ == '__main__':
    print('load dataset')
    train = pd.read_feather(INPUT / 'application_train.ftr')
    test = pd.read_feather(INPUT / 'application_test.ftr')
    days_cols = {'birth': 'DAYS_BIRTH', 'employ': 'DAYS_EMPLOYED', 'register': 'DAYS_REGISTRATION',
                 'publish': 'DAYS_ID_PUBLISH', 'phone': 'DAYS_LAST_PHONE_CHANGE'}
    income_cols = {'total': 'AMT_INCOME_TOTAL', 'credit': 'AMT_CREDIT', 'annuity': 'AMT_ANNUITY',
                   'goods': 'AMT_GOODS_PRICE'}

    preprocessing()
    days_to_years()
    days_pairwise()
    money_pairwise()
