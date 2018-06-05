import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat import SubfileFeature, get_arguments, generate_features
from utils import timer
from config import *

PREV_CAT_COLS = ['NAME_CONTRACT_STATUS', 'WEEKDAY_APPR_PROCESS_START',
                 # 'HOUR_APPR_PROCESS_START',
                 'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY', 'NAME_CASH_LOAN_PURPOSE',
                 'NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON',
                 'NAME_PAYMENT_TYPE', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO',
                 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'SELLERPLACE_AREA', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP',
                 'PRODUCT_COMBINATION', 'NFLAG_INSURED_ON_APPROVAL']


class PrevLatest(SubfileFeature):
    def create_features(self):
        self.df = prev.groupby('SK_ID_CURR').last()


class PrevLastApproved(SubfileFeature):
    def create_features(self):
        self.df = prev.query("NAME_CONTRACT_STATUS=='Approved'").groupby('SK_ID_CURR').last() \
            .drop(['NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON'], axis=1)


class PrevCategoryCount(SubfileFeature):
    def create_features(self):
        dfs = []
        for f in tqdm(prev.select_dtypes(['object']).columns):
            count = prev.groupby('SK_ID_CURR')[f].value_counts().unstack().fillna(0).astype(int)
            ratio = count.apply(lambda x: x / count.sum(axis=1))
            count.columns = f + '_' + count.columns + '_count'
            ratio.columns = f + '_' + ratio.columns + '_ratio'
            df = pd.concat([count, ratio], axis=1)
            df.columns = f + '_' + df.columns
            df[f + '_nunique'] = prev.groupby('SK_ID_CURR')[f].nunique()
            # df[f + '_latest'] = prev.groupby('SK_ID_CURR')[f].last()
            dfs.append(df)
        self.df = pd.concat(dfs, axis=1)


class PrevCategoryTfidf(SubfileFeature):
    def create_features(self):
        dfs = []
        tfidf_transformer = TfidfTransformer()
        for f in tqdm(prev.select_dtypes(['object']).columns):
            count = prev.groupby('SK_ID_CURR')[f].value_counts().unstack().fillna(0).astype(int)
            df = pd.DataFrame(
                tfidf_transformer.fit_transform(count).toarray(),
                index=count.index, columns=[f'{f}_tfidf_{i}' for i in range(count.shape[1])])
            dfs.append(df)
        self.df = pd.concat(dfs, axis=1)


class PrevCategoryLda(SubfileFeature):
    def create_features(self):
        dfs = []
        n_components = 2
        lda = LatentDirichletAllocation(
            n_components=n_components, learning_method='online', n_jobs=-1, random_state=71,
            batch_size=256, max_iter=5
        )
        for f in tqdm(prev.select_dtypes(['object']).columns):
            count = prev.groupby('SK_ID_CURR')[f].value_counts().unstack().fillna(0).astype(int)
            df = pd.DataFrame(lda.fit_transform(count), index=count.index,
                              columns=[f'{f}_lda_{i}' for i in range(n_components)])
            dfs.append(df)
        self.df = pd.concat(dfs, axis=1)


class PrevBasic(SubfileFeature):
    def create_features(self):
        df = prev.query('NAME_CONTRACT_STATUS == "Approved"').copy()
        
        amt_cols = prev.filter(regex='AMT_').columns
        for i, j in tqdm(itertools.combinations(amt_cols, 2)):
            df[f'{i}_minus_{j}'] = df[i] - df[j]
        
        day_cols = prev.filter(regex='DAY_').columns
        for i, j in tqdm(itertools.combinations(day_cols, 2)):
            df[f'{i}_minus_{j}'] = df[i] - df[j]
        
        df.loc[:, df.columns.str.startswith('DAYS_')] \
            = df.filter(regex='DAYS_').replace({365243: np.nan})
        
        print('min')
        min_df = df.select_dtypes(exclude=['object']).groupby('SK_ID_CURR').min()
        min_df.columns += '_min'
        print('mean')
        mean_df = df.select_dtypes(exclude=['object']).groupby('SK_ID_CURR').mean()
        mean_df.columns += '_mean'
        print('max')
        max_df = df.select_dtypes(exclude=['object']).groupby('SK_ID_CURR').max()
        max_df.columns += '_max'
        
        self.df = pd.concat([min_df, mean_df, max_df], axis=1)


if __name__ == '__main__':
    args = get_arguments(Path(__file__).stem)
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)[['SK_ID_CURR']]
        test = pd.read_feather(TEST)[['SK_ID_CURR']]
        prev = pd.read_feather(PREV)
    
    with timer('preprocessing'):
        prev.drop(['SK_ID_PREV', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED'], axis=1, inplace=True)
        prev = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION']).reset_index(drop=True)
        prev.loc[:, prev.columns.str.startswith('AMT_')] = np.log1p(prev.filter(regex='^AMT_'))
        # prev.replace({'XNA': np.nan, 'XAP': np.nan}, inplace=True)
        # prev.loc[:, prev.columns.str.startswith('DAYS_')] = prev.filter(regex='^DAYS_').replace({365243: np.nan})
        prev.AMT_DOWN_PAYMENT.fillna(0)
        prev.RATE_DOWN_PAYMENT.fillna(0)
        cat_cols = prev.select_dtypes(['object']).columns
        prev[cat_cols] = prev[cat_cols].fillna('NaN')
    
    with timer('create dataset'):
        generate_features([
            PrevLatest('prev', 'latest'),
            PrevLastApproved('prev', 'last_approved'),
            PrevCategoryCount('prev', ''),
            PrevCategoryTfidf('prev'),
            PrevCategoryLda('prev'),
            PrevBasic('prev', '')
        ], args.force)
