import itertools

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

tqdm.pandas()

import sys
import numpy as np
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from utils import timer

PREFIX = 'prev_'


@timer('null count')
def null_count():
    new = prev.isnull().sum(axis=1).to_frame('null_cnt')
    new['SK_ID_CURR'] = prev.SK_ID_CURR
    return new.groupby('SK_ID_CURR').null_cnt.mean()


@timer('count')
def count():
    return prev.groupby('SK_ID_CURR').SK_ID_PREV.count()


@timer('amount_pairwise')
def amount_pairwise():
    new = pd.DataFrame()
    columns = prev_mean.filter('AMT_').columns.tolist()
    for i, j in tqdm(itertools.combinations(columns, 2)):
        name = f'{i.replace("_mean", "")}_minus_{j.replace("_mean", "")}'
        new[name] = prev[i] - prev[j]
    return new


@timer('days pairwise')
def days_pairwise():
    new = pd.DataFrame()
    columns = prev_mean.filter('DAYS_').columns.tolist()
    for i, j in tqdm(itertools.combinations(columns, 2)):
        name = f'{i.replace("_mean", "")}_minus_{j.replace("_mean", "")}'
        new[name] = prev[i] - prev[j]
    return new


@timer('category')
def category():
    new = pd.DataFrame()
    cat_cols = [f for f in prev.columns if prev[f].dtype == 'object']
    le = LabelEncoder()
    for f in tqdm(cat_cols):
        new[f + '_latest'] = le.fit_transform(prev.groupby('SK_ID_CURR')[f].tail(1).astype(str))
        new[f + '_nunique'] = prev.groupby('SK_ID_CURR')[f].nunique()
        new[f + '_count'] = prev.groupby('SK_ID_CURR')[f].count()


@timer('weekday to sin & cos')
def weekday_to_sin_cos():
    new = pd.DataFrame()
    new['appr_weekday_sin'] = np.sin(prev.WEEKDAY_APPR_PROCESS_START / 7)
    new['appr_weekday_cos'] = np.cos(prev.WEEKDAY_APPR_PROCESS_START / 7)
    return new


@timer('sellerplace LDA')
def sellerplace_lda(n_components=5):
    new = pd.DataFrame()
    cat_cols = [f for f in prev.columns if prev[f].dtype == 'object']
    lda = LatentDirichletAllocation(n_components=n_components, learning_method='batch')
    for f in tqdm(cat_cols):
        piv = pd.pivot_table(prev, index='SELLERPLACE_AREA', values='SK_ID_PREV', columns=f, aggfunc='count',
                             fill_value=0)
        col_names = [f'lda_{f}_{i}' for i in range(n_components)]
        df = pd.DataFrame(lda.fit_transform(piv), index=piv.index, columns=col_names)
        new = pd.concat([new, df], axis=1)
    return new


@timer('mode of sellerplace')
def sellerplace_mode():
    return prev.groupby('SK_ID_CURR').SELLERPLACE_AREA.apply(lambda x: x.value_counts().index[0])


if __name__ == '__main__':
    print('load datasets')
    train = pd.read_feather(INPUT / 'application_train.ftr')
    test = pd.read_feather(INPUT / 'application_test.ftr')
    prev = pd.read_feather(INPUT / 'previous_application.ftr')
    
    print('{XNA, XAP, 365243} => NaN')
    prev.replace({'XNA': np.nan, 'XAP': np.nan}, inplace=True)
    prev.loc[:, prev.columns.str.startswith('DAYS_')] = prev.filter(regex='^DAYS_').replace({365243: np.nan})
    
    print('sort')
    prev = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION']).reset_index(drop=True)
    
    print('np.log1p(AMOUNT)')
    prev.loc[:, prev.columns.str.startswith('AMT_')] = np.log1p(prev.filter(regex='^AMT_'))
    
    print('is_approved')
    prev['is_approved'] = (prev.NAME_CONTRACT_TYPE == 'Approved').astype(int)
    
    print('mean')
    prev_mean = prev.groupby('SK_ID_CURR').mean()
    # prev_mean.columns = prev_mean.columns + '_mean'
    #
    # print('max')
    # prev_max = prev.groupby('SK_ID_CURR').max()
    # prev_max.columns = prev_max.columns + '_max'
    #
    # print('min')
    # prev_min = prev.groupby('SK_ID_CURR').min()
    # prev_min.columns = prev_min.columns + '_min'
    
    # print('concat')
    # new = pd.concat([prev_min, prev_mean, prev_max], axis=1)
    #
    new = prev_mean
    
    print('calc features')
    new['null_cnt'] = null_count()
    new['count'] = count()
    new = pd.concat([new, amount_pairwise()], axis=1)
    new = pd.concat([new, days_pairwise()], axis=1)
    # new = pd.concat([new, weekday_to_sin_cos()])
    new = pd.concat([new, category()], axis=1)
    new['sellerplace_mode'] = sellerplace_mode()
    
    # tmp = sellerplace_lda()
    # lda_mean = prev.merge(tmp, left_on='SELLERPLACE_AREA', right_index=True, how='left') \
    #     .groupby('SK_ID_CURR')[tmp.columns].mean()
    # lda_mean.columns = tmp.columns + '_mean'
    # new = pd.concat([new, lda_mean], axis=1)
    # tmp.columns = tmp.columns + '_mode'
    # new = new.merge(tmp, left_on='sellerplace_mode', right_index=True, how='left')
    
    new.drop(new.filter(regex='SK_ID_PREV').columns, axis=1, inplace=True)
    new.columns = PREFIX + new.columns
    train.merge(new, left_on='SK_ID_CURR', right_index=True, how='left') \
        .filter(regex=PREFIX).to_feather(WORKING / f'{PREFIX}train.ftr')
    test.merge(new, left_on='SK_ID_CURR', right_index=True, how='left') \
        .filter(regex=PREFIX).to_feather(WORKING / f'{PREFIX}test.ftr')
