import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

tqdm.pandas()

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

print('load datasets')
train = pd.read_feather(INPUT / 'application_train.ftr')
test = pd.read_feather(INPUT / 'application_test.ftr')
bureau = pd.read_feather(INPUT / 'bureau.ftr')
bureau_balance = pd.read_feather(INPUT / 'bureau_balance.ftr')

print('preprocessing bureau balance')
buro_bal = bureau_balance.groupby('SK_ID_BUREAU') \
    .STATUS.value_counts().unstack('STATUS').fillna(0).astype(int)

buro_bal.columns = 'status_' + buro_bal.columns + '_cnt'

buro_bal['months_cnt'] = bureau_balance.groupby('SK_ID_BUREAU').MONTHS_BALANCE.size()
buro_bal['months_max'] = bureau_balance.groupby('SK_ID_BUREAU').MONTHS_BALANCE.max()
buro_bal['months_min'] = bureau_balance.groupby('SK_ID_BUREAU').MONTHS_BALANCE.min()

for f in tqdm(buro_bal.filter(regex='status_').columns):
    buro_bal[f.replace('_cnt', '_ratio')] = buro_bal[f] / buro_bal.months_cnt

buro = bureau.merge(buro_bal.reset_index(), how='left', on='SK_ID_BUREAU')

print('encode categorical features')
le = LabelEncoder()
buro_cat = [f for f in buro.columns if buro[f].dtype == 'object']
for f in tqdm(buro_cat):
    buro[f] = le.fit_transform(buro[f].astype(str))
    nunique = buro[['SK_ID_CURR', f]].groupby('SK_ID_CURR').nunique()
    nunique = nunique[f].reset_index().rename(columns={f: 'nunique_' + f})
    buro = buro.merge(nunique, on='SK_ID_CURR', how='left')
    buro.drop(f, axis=1, inplace=True)

avg_buro = buro.groupby('SK_ID_CURR').mean().drop('SK_ID_BUREAU', axis=1)
avg_buro.columns = 'bureau_' + avg_buro.columns

print('add count and null feature')
avg_buro['bureau_cnt'] = buro.groupby('SK_ID_CURR').SK_ID_BUREAU.count()
avg_buro['bureau_null_cnt'] = avg_buro.isnull().sum(axis=1)

train.merge(avg_buro, left_on='SK_ID_CURR', right_index=True, how='left') \
    .filter(regex='bureau_').to_feather(WORKING / 'bureau_train.ftr')
test.merge(avg_buro, left_on='SK_ID_CURR', right_index=True, how='left') \
    .filter(regex='bureau_').to_feather(WORKING / 'bureau_test.ftr')
