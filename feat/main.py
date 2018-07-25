import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../../spica')
from spica.features.base import Feature, generate_features, get_arguments

from utils import timer
from config import *

ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
person_cols = ['CNT_FAM_MEMBERS', 'CNT_CHILDREN']
amt_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

Feature.dir = '../working'
Feature.prefix = 'main'


class MainAmount(Feature):
    def create_features(self):
        cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']
        self.train = train[cols]
        self.test = test[cols]


class MainAmountReq(Feature):
    def create_features(self):
        self.train = train.filter(regex='AMT_REQ_')
        self.test = test.filter(regex='AMT_REQ_')


class MainHouse(Feature):
    def create_features(self):
        self.train = train.filter(regex='(_AVG|_MEDI|_MODE)$')
        self.test = test.filter(regex='(_AVG|_MEDI|_MODE)$')


class MainGender(Feature):
    def create_features(self):
        self.train = train[['CODE_GENDER']].copy()
        self.test = test[['CODE_GENDER']].copy()
        print(self.train.dtypes)


class MainDays(Feature):
    def create_features(self):
        self.train = train.filter(regex='DAYS_')
        self.test = test.filter(regex='DAYS_')


class MainSocial(Feature):
    def create_features(self):
        self.train = train.filter(regex='SOCIAL')
        self.test = test.filter(regex='SOCIAL')


class MainExt(Feature):
    def create_features(self):
        self.train = train.filter(regex='EXT_SOURCE')
        self.test = test.filter(regex='EXT_SOURCE')


class MainDocument(Feature):
    def create_features(self):
        self.train = train.filter(regex='DOCUMENT')
        self.test = test.filter(regex='DOCUMENT')


class MainFlag(Feature):
    def create_features(self):
        cols = list(train.filter(regex='FLAG_(?!DOCUMENT)').columns) + \
               ['LIVE_CITY_NOT_WORK_CITY', 'LIVE_REGION_NOT_WORK_REGION',
                'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
                'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION']
        self.train = train[cols]
        self.test = test[cols]


class MainRegion(Feature):
    def create_features(self):
        cols = train.filter(regex='^REGION_').columns
        self.train = train[cols]
        self.test = test[cols]


class MainCar(Feature):
    def create_features(self):
        self.train = train[['OWN_CAR_AGE']]
        self.test = test[['OWN_CAR_AGE']]


class MainApplication(Feature):
    def create_features(self):
        self.train = train.filter(regex='APPR')
        self.test = test.filter(regex='APPR')


class MainNullCount(Feature):
    def create_features(self):
        for new, df in zip([self.train, self.test], [train, test]):
            new['null_count'] = df.isnull().sum(axis=1)


class MainCategory(Feature):
    def create_features(self):
        self.train = train.filter(regex='(NAME_|_TYPE)')
        self.train.columns = self.train.columns
        self.test = test.filter(regex='(NAME_|_TYPE)')
        self.test.columns = self.test.columns


class MainDayPairwise(Feature):
    def create_features(self):
        day_cols = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE']
        # 欠損処理してない
        for i, j in itertools.combinations(day_cols, 2):
            for df, new in zip([train.copy(), test.copy()], [self.train, self.test]):
                df['OWN_CAR_AGE'] *= -365.25
                df.loc[df['FLAG_OWN_CAR'] == 0, 'OWN_CAR_AGE'] = 0
                # new[f'{i}_sub_{j}'] = df[i] - df[j]
                new[f'{i}_div_{j}'] = df[i] / (df[j] + 0.1)


class MainAmountPerPerson(Feature):
    def create_features(self):
        # 欠損処理してない
        for person, amt in itertools.product(person_cols, amt_cols):
            self.train[f'{amt}_per_{person}'] = train[amt] / (train[person] + 0.1)
            self.test[f'{amt}_per_{person}'] = test[amt] / (test[person] + 0.1)


class MainAmountPairwise(Feature):
    def create_features(self):
        for i, j in itertools.combinations(amt_cols, 2):
            # sub = f'{i}_sub_{j}'
            # self.train[sub] = train[i] - train[j]
            # self.test[sub] = test[i] - test[j]
            div = f'{i}_div_{j}'
            self.train[div] = train[i] / (train[j] + 0.1)
            self.test[div] = test[i] / (test[j] + 0.1)
            
            for d in ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                      'DAYS_LAST_PHONE_CHANGE']:
                self.train[div + '_div_' + d] = self.train[div] / (abs(train[d]) + 0.1)
                self.test[div + '_div_' + d] = self.test[div] / (abs(test[d]) + 0.1)


class MainFlagCount(Feature):
    def create_features(self):
        self.train['zero_count'] = (train.filter(regex='(FLAG_|REG_|LIVE_|_AVG|_MODE|_MEDI)') == 0).sum(axis=1)
        self.test['zero_count'] = (test.filter(regex='(FLAG_|REG_|LIVE_|_AVG|_MODE|_MEDI)') == 0).sum(axis=1)
        self.train['one_count'] = (train.filter(regex='(FLAG_|REG_|LIVE_|_AVG|_MODE|_MEDI)') == 1).sum(axis=1)
        self.test['one_count'] = (test.filter(regex='(FLAG_|REG_|LIVE_|_AVG|_MODE|_MEDI)') == 1).sum(axis=1)


class MainExtPairwise(Feature):
    def create_features(self):
        self.train[ext_cols] = train[ext_cols].fillna(train[ext_cols].mean())
        self.test[ext_cols] = test[ext_cols].fillna(train[ext_cols].mean())
        for i, j in itertools.combinations(ext_cols, 2):
            self.train[f'{i}_add_{j}'] = self.train[i] + self.train[j]
            self.train[f'{i}_sub_{j}'] = self.train[i] - self.train[j]
            self.train[f'{i}_mul_{j}'] = self.train[i] * self.train[j]
            self.train[f'{i}_div_{j}'] = self.train[i] / (self.train[j] + 0.1)
            self.test[f'{i}_add_{j}'] = self.test[i] + self.test[j]
            self.test[f'{i}_sub_{j}'] = self.test[i] - self.test[j]
            self.test[f'{i}_mul_{j}'] = self.test[i] * self.test[j]
            self.test[f'{i}_div_{j}'] = self.test[i] / (self.test[j] + 0.1)
        assert self.train.isnull().sum().sum() + self.test.isnull().sum().sum() == 0


class MainExtMean(Feature):
    def create_features(self):
        trn = train[ext_cols].fillna(train[ext_cols].mean())
        tst = test[ext_cols].fillna(train[ext_cols].mean())
        self.train['EXT_SOURCE_mean'] = trn[ext_cols].mean(axis=1)
        self.test['EXT_SOURCE_mean'] = tst[ext_cols].mean(axis=1)
        assert self.train.isnull().sum().sum() + self.test.isnull().sum().sum() == 0


# class MainExtNull(Feature):
#     def create_features(self):
#         self.train = pd.DataFrame()
#         self.test = pd.DataFrame()
#         for i in range(1, 4):
#             self.train[f'EXT_SOURCE_{i}_isnull'] = train[f'EXT_SOURCE_{i}'].isnull()
#             self.test[f'EXT_SOURCE_{i}_isnull'] = test[f'EXT_SOURCE_{i}'].isnull()
#         for i, j in itertools.combinations_with_replacement(range(1, 4), 2):
#             if i == j:
#                 name = f'EXT_SOURCE_{i}_isnull'
#                 self.train[name] = train[f'EXT_SOURCE_{i}'].isnull().astype(int)
#                 self.test[name] = test[f'EXT_SOURCE_{i}'].isnull().astype(int)
#             else:
#                 name = f'EXT_SOURCE_isnull_{i}_or_{j}'
#                 self.train[name] = (train[f'EXT_SOURCE_{i}'].isnull() | train[f'EXT_SOURCE_{i}'].isnull()).astype(int)
#                 self.test[name] = (test[f'EXT_SOURCE_{i}'].isnull() | test[f'EXT_SOURCE_{i}'].isnull()).astype(int)
#                 name = f'EXT_SOURCE_isnull_{i}_and_{j}'
#                 self.train[name] = (train[f'EXT_SOURCE_{i}'].isnull() & train[f'EXT_SOURCE_{i}'].isnull()).astype(int)
#                 self.test[name] = (test[f'EXT_SOURCE_{i}'].isnull() & test[f'EXT_SOURCE_{i}'].isnull()).astype(int)
#         self.train['EXT_SOURCE_null_cnt'] = train.filter(regex='EXT_').isnull().sum(axis=1)
#         self.test['EXT_SOURCE_null_cnt'] = test.filter(regex='EXT_').isnull().sum(axis=1)


class MainDocumentCount(Feature):
    prefix = 'main'
    
    def create_features(self):
        self.train = train.filter(regex='FLAG_DOCUMENT').sum(axis=1).to_frame('document_count')
        self.test = test.filter(regex='FLAG_DOCUMENT').sum(axis=1).to_frame('document_count')


class MainEnquiry(Feature):
    prefix = 'main'
    suffix = 'cumsum'
    
    def create_features(self):
        self.train = train.filter(regex='AMT_REQ_').cumsum(axis=1).drop('AMT_REQ_CREDIT_BUREAU_HOUR', axis=1)
        self.test = test.filter(regex='AMT_REQ_').cumsum(axis=1).drop('AMT_REQ_CREDIT_BUREAU_HOUR', axis=1)


class MainFamily(Feature):
    def create_features(self):
        df = pd.DataFrame()
        cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
        for i in cols:
            df[f'{i}_per_family'] = X[i] / X.CNT_FAM_MEMBERS
            df[f'{i}_per_children'] = X[i] / (X.CNT_CHILDREN + 1)
        df['children_ratio'] = X.CNT_CHILDREN / X.CNT_FAM_MEMBERS
        df['cnt_parent'] = X.CNT_FAM_MEMBERS - X.CNT_CHILDREN
        self.train, self.test = df[:len(train)].copy(), df[len(train):].copy()
        self.train['CNT_FAM_MEMBERS'] = train['CNT_FAM_MEMBERS']
        self.train['CNT_CHILDREN'] = train['CNT_CHILDREN']
        self.test['CNT_FAM_MEMBERS'] = test['CNT_FAM_MEMBERS']
        self.test['CNT_CHILDREN'] = test['CNT_CHILDREN']


class MainAreaIncome(Feature):
    def create_features(self):
        area = 'REGION_POPULATION_RELATIVE'
        cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
        mean = X.groupby(area)[cols].mean().rename(columns=lambda x: 'area_' + x + '_mean')
        medi = X.groupby(area)[cols].median().rename(columns=lambda x: 'area_' + x + '_median')
        self.train = train.merge(mean, left_on=area, right_index=True, how='left') \
                         .merge(medi, left_on=area, right_index=True, how='left').iloc[:, -len(cols) * 2:]
        self.test = test.merge(mean, left_on=area, right_index=True, how='left') \
                        .merge(medi, left_on=area, right_index=True, how='left').iloc[:, -len(cols) * 2:]
        for f in cols:
            self.train[f'area_{f}_mean_diff'] = train[f] - self.train[f'area_{f}_mean']
            self.train[f'area_{f}_median_diff'] = train[f] - self.train[f'area_{f}_median']
            self.train[f'area_{f}_mean_ratio'] = train[f] / self.train[f'area_{f}_mean']
            self.train[f'area_{f}_median_ratio'] = train[f] / self.train[f'area_{f}_median']
            self.test[f'area_{f}_mean_diff'] = test[f] - self.test[f'area_{f}_mean']
            self.test[f'area_{f}_median_diff'] = test[f] - self.test[f'area_{f}_median']
            self.test[f'area_{f}_mean_ratio'] = test[f] / self.test[f'area_{f}_mean']
            self.test[f'area_{f}_median_ratio'] = test[f] / self.test[f'area_{f}_median']


class MainRegionAsCategory(Feature):
    def create_features(self):
        self.train = train['REGION_POPULATION_RELATIVE'].astype(str).to_frame('region_as_category')
        self.test = test['REGION_POPULATION_RELATIVE'].astype(str).to_frame('region_as_category')


if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
        X = pd.concat([
            train.drop('TARGET', axis=1),
            test
        ])  # type: pd.DataFrame
    
    with timer('preprocessing'):
        train.AMT_INCOME_TOTAL.replace(117000000.0, 1170000, inplace=True)
        train.replace({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0, 'XAP': np.nan, 'XAN': np.nan}, inplace=True)
        test.replace({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0, 'XAP': np.nan, 'XAN': np.nan}, inplace=True)
        day_cols = train.filter(regex='DAYS_').columns
        train[day_cols] = train[day_cols].replace(365243, np.nan)
        test[day_cols] = test[day_cols].replace(365243, np.nan)
    
    with timer('create dataset'):
        generate_features(globals(), args.force)
