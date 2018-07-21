import itertools
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../../spica')
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

from feat import SubfileFeature
from utils import timer
from config import *
from spica.features.base import Feature, get_arguments, generate_features

Feature.dir = '../working'
SubfileFeature.dir = '../working'


class PrevFeature(SubfileFeature):
    prefix = 'prev'


# class PrevLatest(PrevFeature):
#     suffix = 'latest'
#
#     def create_features(self):
#         self.df = prev.groupby('SK_ID_CURR').last()

#
# class PrevLastApproved(PrevFeature):
#     suffix = 'last_approved'
#
#     def create_features(self):
#         self.df = prev.query("NAME_CONTRACT_STATUS=='Approved'").groupby('SK_ID_CURR').last() \
#             .drop(['NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON'], axis=1)


# class PrevBasicAll(SubfileFeature):
#     def create_features(self):
#         prev_ = prev.copy()
#         prev_['APP_PER_CREDIT'] = prev_['AMT_APPLICATION'] / prev_['AMT_CREDIT']
#         num_cols = [f for f in prev_.columns if prev_[f].dtype != 'object']
#         cat_cols = [f for f in prev_.columns if prev_[f].dtype == 'object']
#         self.df = pd.DataFrame()
#         for f in num_cols:
#             self.df[f'{f}_min'] = prev_.groupby('SK_ID_CURR')[f].min()
#             self.df[f'{f}_mean'] = prev_.groupby('SK_ID_CURR')[f].mean()
#             self.df[f'{f}_max'] = prev_.groupby('SK_ID_CURR')[f].max()
#             self.df[f'{f}_std'] = prev_.groupby('SK_ID_CURR')[f].std()
#             self.df[f'{f}_sum'] = prev_.groupby('SK_ID_CURR')[f].sum()
#         for f in cat_cols:
#             self.df[f'{f}_nunique'] = prev_.groupby('SK_ID_CURR')[f].nunique()
#         self.df['count'] = prev_.groupby('SK_ID_CURR').DAYS_DECISION.count()
#
#         approved = prev_.query('NAME_CONTRACT_STATUS == "Approved"').drop('NAME_CONTRACT_STATUS', axis=1)
#         for f in num_cols:
#             self.df[f'approved_{f}_min'] = approved.groupby('SK_ID_CURR')[f].min()
#             self.df[f'approved_{f}_mean'] = approved.groupby('SK_ID_CURR')[f].mean()
#             self.df[f'approved_{f}_max'] = approved.groupby('SK_ID_CURR')[f].max()
#             self.df[f'approved_{f}_std'] = approved.groupby('SK_ID_CURR')[f].std()
#             self.df[f'approved_{f}_sum'] = approved.groupby('SK_ID_CURR')[f].sum()
#         for f in cat_cols:
#             self.df[f'approved_{f}_nunique'] = approved.groupby('SK_ID_CURR')[f].nunique()
#         self.df['approved_count'] = approved.groupby('SK_ID_CURR').DAYS_DECISION.count()
#
#         refused = prev_.query('NAME_CONTRACT_STATUS == "Refused"').drop('NAME_CONTRACT_STATUS', axis=1)
#         for f in num_cols:
#             self.df[f'refused_{f}_min'] = refused.groupby('SK_ID_CURR')[f].min()
#             self.df[f'refused_{f}_mean'] = refused.groupby('SK_ID_CURR')[f].mean()
#             self.df[f'refused_{f}_max'] = refused.groupby('SK_ID_CURR')[f].max()
#             self.df[f'refused_{f}_std'] = refused.groupby('SK_ID_CURR')[f].std()
#             self.df[f'refused_{f}_sum'] = refused.groupby('SK_ID_CURR')[f].sum()
#         for f in cat_cols:
#             self.df[f'refused_{f}_nunique'] = refused.groupby('SK_ID_CURR')[f].nunique()
#         self.df['refused_count'] = refused.groupby('SK_ID_CURR').DAYS_DECISION.count()
#
#
# class PrevAmountPairwise(SubfileFeature):
#     def create_features(self):
#         amt_cols = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE']
#         prev_ = prev.copy()
#         for funcname, func in {'log': np.log1p, 'sqrt': np.sqrt}.items():
#             for f in amt_cols:
#                 name = f'{f}_{funcname}'
#                 prev_[name] = func(prev[f])
#                 prev_[name].fillna(prev_[name].mean(), inplace=True)
#
#             cols = [f'{f}_{funcname}' for f in amt_cols]
#             for i, j in itertools.combinations(cols, 2):
#                 prev_[f'{i}_sub_{j}'] = prev_[i] - prev_[j]
#                 prev_[f'{i}_div_{j}'] = prev_[i] / (prev_[j] + 0.1)
#
#             prev_[f'AMT_{funcname}_mean'] = prev_[cols].mean(axis=1)
#         self.df = pd.concat([
#             prev_.groupby('SK_ID_CURR').min().rename(columns=lambda x: x + '_min'),
#             prev_.groupby('SK_ID_CURR').mean().rename(columns=lambda x: x + '_mean'),
#             prev_.groupby('SK_ID_CURR').max().rename(columns=lambda x: x + '_max'),
#             prev_.groupby('SK_ID_CURR').sum().rename(columns=lambda x: x + '_sum'),
#             prev_.groupby('SK_ID_CURR').std().rename(columns=lambda x: x + '_std'),
#         ], axis=1)


class PrevCategoryCount(PrevFeature):
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


class PrevCategoryTfidf(PrevFeature):
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


# class PrevCategoryLda(PrevFeature):
#     def create_features(self):
#         dfs = []
#         n_components = 2
#         lda = LatentDirichletAllocation(
#             n_components=n_components, learning_method='online', n_jobs=-1, random_state=71,
#             batch_size=256, max_iter=5
#         )
#         for f in tqdm(prev.select_dtypes(['object']).columns):
#             count = prev.groupby('SK_ID_CURR')[f].value_counts().unstack().fillna(0).astype(int)
#             df = pd.DataFrame(lda.fit_transform(count), index=count.index,
#                               columns=[f'{f}_lda_{i}' for i in range(n_components)])
#             dfs.append(df)
#         self.df = pd.concat(dfs, axis=1)


class PrevBasic(PrevFeature):
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


# class PrevProductCombination(PrevFeature):
#     def create_features(self):
#         df = prev[['SK_ID_CURR']].copy()
#         mapping = {
#             'interest': 'with interest',
#             'pos': 'POS',
#             'cash': 'Cash',
#             'card': 'Card',
#             'mobile': 'mobile',
#             'household': 'household',
#             'industry': 'industry',
#             'x_sell': 'X-Sell',
#             'street': 'Street',
#             'low': 'low',
#             'middle': 'middle',
#             'high': 'high'
#         }
#         for k, v in mapping.items():
#             df[k] = prev.PRODUCT_COMBINATION.str.contains(v).fillna(False).astype(int)
#         self.df = pd.concat([
#             df.groupby('SK_ID_CURR').min().rename(columns=lambda x: x + '_min'),
#             df.groupby('SK_ID_CURR').mean().rename(columns=lambda x: x + '_mean'),
#             df.groupby('SK_ID_CURR').max().rename(columns=lambda x: x + '_max')
#         ], axis=1)


class PrevNullCount(PrevFeature):
    prefix = 'prev_null_count'
    
    def create_features(self):
        df = prev.copy()
        df['null_count'] = df.isnull().sum(axis=1)
        self.df['min'] = df.groupby('SK_ID_CURR').null_count.min()
        self.df['mean'] = df.groupby('SK_ID_CURR').null_count.mean()
        self.df['max'] = df.groupby('SK_ID_CURR').null_count.max()


class PrevAmountToMain(Feature):
    prefix = 'prev'
    
    def create_features(self):
        main_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
        prev_cols = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE']
        g = prev[['SK_ID_CURR'] + prev_cols].groupby('SK_ID_CURR')
        prev_df = pd.concat([
            g.max().rename(columns=lambda x: x + '_max'),
            g.mean().rename(columns=lambda x: x + '_mean')
        ], axis=1)
        trn = train.merge(prev_df, left_on='SK_ID_CURR', right_index=True, how='left')
        tst = test.merge(prev_df, left_on='SK_ID_CURR', right_index=True, how='left')
        for m, p in itertools.product(main_cols, prev_cols):
            self.train[f'{m}_sub_{p}_max'] = trn[m] - trn[p + '_max']
            self.train[f'{m}_sub_{p}_mean'] = trn[m] - trn[p + '_mean']
            self.test[f'{m}_sub_{p}_max'] = tst[m] - tst[p + '_max']
            self.test[f'{m}_sub_{p}_mean'] = tst[m] - tst[p + '_mean']
            
            self.train[f'{m}_div_{p}_max'] = trn[m] / trn[p + '_max']
            self.train[f'{m}_div_{p}_mean'] = trn[m] / trn[p + '_mean']
            self.test[f'{m}_div_{p}_max'] = tst[m] / tst[p + '_max']
            self.test[f'{m}_div_{p}_mean'] = tst[m] / tst[p + '_mean']


if __name__ == '__main__':
    args = get_arguments(Path(__file__).stem)
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
        prev = pd.read_feather(PREV)
    
    with timer('preprocessing'):
        prev.drop(['SK_ID_PREV', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED'], axis=1, inplace=True)
        prev = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION']).reset_index(drop=True)
        prev.loc[:, prev.columns.str.startswith('DAYS_')] = prev.filter(regex='^DAYS_').replace({365243: np.nan})
        prev.AMT_DOWN_PAYMENT.fillna(0)
        prev.RATE_DOWN_PAYMENT.fillna(0)
        cat_cols = prev.select_dtypes(['object']).columns
        prev[cat_cols] = prev[cat_cols].fillna('NaN')
    
    with timer('create dataset'):
        generate_features(globals(), args.force)
