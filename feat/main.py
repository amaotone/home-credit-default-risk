import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import *

print('load dataset')
train = pd.read_feather(INPUT + 'application_train.ftr')
test = pd.read_feather(INPUT + 'application_test.ftr')
new_feats = []

# ==================================================
# binary + missing
# https://www.kaggle.com/c/home-credit-default-risk/discussion/57248
# ==================================================
print('replace binary and nan')
train.iloc[:, 1:] = train.iloc[:, 1:].replace(
    {'Y': 1, 'N': 0, 'M': 0, 'F': 1, 'XNA': np.nan, 'XAP': np.nan, 365243: np.nan})
test.iloc[:, 1:] = test.iloc[:, 1:].replace(
    {'Y': 1, 'N': 0, 'M': 0, 'F': 1, 'XNA': np.nan, 'XAP': np.nan, 365243: np.nan})

# ==================================================
# DAYS
# ==================================================
days_cols = {'birth': 'DAYS_BIRTH', 'employed': 'DAYS_EMPLOYED', 'registered': 'DAYS_REGISTERED',
             'publish': 'DAYS_ID_PUBLISH', 'phone': 'DAYS_LAST_PHONE_CHANGE'}

print('DAYS => year')
for k, v in tqdm(days_cols.items()):
    name = f'years_{k}'
    train[name] = train[v] // 365.25
    test[name] = test[v] // 365.25
    new_feats.append(name)

print('DAYS pairwise')
for (i, j) in tqdm(itertools.combinations(days_cols.keys(), 2)):
    name = f'days_{i}_minus_{j}'
    train[name] = train[days_cols[i]] - train[days_cols[j]]
    test[name] = train[days_cols[i]] - train[days_cols[j]]
    new_feats.append(name)

# ==================================================
# INCOME
# ==================================================
print('income features')
for df in [train, test]:
    df['credit_income_ratio'] = df.AMT_CREDIT / df.AMT_INCOME_TOTAL
    df['goods_income_ratio'] = df.AMT_CREDIT / df.AMT_INCOME_TOTAL
    df['annuity_income_ratio'] = df.AMT_ANNUITY / df.AMT_INCOME_TOTAL
    df['goods_credit_ratio'] = df.AMT_GOODS_PRICE / df.AMT_CREDIT
    df['n_division'] = df.AMT_CREDIT / df.AMT_ANNUITY
new_feats += ['credit_income_ratio', 'goods_income_ratio', 'annuity_income_ratio', 'goods_credit_ratio', 'n_division']
