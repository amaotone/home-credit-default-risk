import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat import Feature, get_arguments, generate_features
from utils import timer
from config import *

ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
person_cols = ['CNT_FAM_MEMBERS', 'CNT_CHILDREN']
amt_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']


class MainCategory(Feature):
    def create_features(self):
        self.train = train.filter(regex='(NAME_|_TYPE)')
        self.train.columns = 'main_' + self.train.columns
        self.test = test.filter(regex='(NAME_|_TYPE)')
        self.test.columns = 'main_' + self.test.columns


class MainDayPairwise(Feature):
    def create_features(self):
        day_cols = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']
        # 欠損処理してない
        for i, j in itertools.combinations(day_cols, 2):
            self.train[f'{i}_sub_{j}'] = train[i] - train[j]
            self.train[f'{i}_div_{j}'] = train[i] / (train[j] + 0.1)
            self.test[f'{i}_sub_{j}'] = test[i] - test[j]
            self.test[f'{i}_div_{j}'] = test[i] / (test[j] + 0.1)


class MainAmountPerPerson(Feature):
    def create_features(self):
        # 欠損処理してない
        for person, amt in itertools.product(person_cols, amt_cols):
            self.train[f'{amt}_per_{person}'] = train[amt] / (train[person] + 0.1)
            self.test[f'{amt}_per_{person}'] = test[amt] / (test[person] + 0.1)


class MainAmountPairwise(Feature):
    def create_features(self):
        for funcname, func in {'log': np.log1p, 'sqrt': np.sqrt}.items():
            for f in amt_cols:
                name = f'{f}_{funcname}'
                self.train[name] = func(train[f])
                self.test[name] = func(test[f])
            
            cols = self.train.filter(regex=f'^AMT_(.*)_{funcname}$').columns.tolist()
            for i, j in itertools.combinations(cols, 2):
                sub = f'{i}_sub_{j}'
                div = f'{i}_div_{j}'
                self.train[sub] = self.train[i] - self.train[j]
                self.train[div] = self.train[i] / (self.train[j] + 0.1)
                
                self.test[sub] = self.test[i] - self.test[j]
                self.test[div] = self.test[i] / (self.test[j] + 0.1)
                
                for d in ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                          'DAYS_LAST_PHONE_CHANGE']:
                    self.train[div + '_div_' + d] = self.train[div] / (abs(train[d]) + 0.1)
                    self.test[div + '_div_' + d] = self.test[div] / (abs(test[d]) + 0.1)
            
            self.train[f'AMT_{funcname}_mean'] = self.train[cols].mean(axis=1)
            self.test[f'AMT_{funcname}_mean'] = self.test[cols].mean(axis=1)


class MainExtRound(Feature):
    def create_features(self):
        self.train = pd.concat([np.round(train[ext_cols], 1).rename(columns=lambda x: x + '_r1'),
                                np.round(train[ext_cols], 2).rename(columns=lambda x: x + '_r2')], axis=1)
        self.test = pd.concat([np.round(test[ext_cols], 1).rename(columns=lambda x: x + '_r1'),
                               np.round(test[ext_cols], 2).rename(columns=lambda x: x + '_r2')], axis=1)
        # r1_cols = self.train.filter(regex='_r1$').columns
        # r2_cols = self.train.filter(regex='_r2$').columns
        # for i, j in list(itertools.combinations(r1_cols, 2)) + list(itertools.combinations(r2_cols, 2)):
        #     self.train[f'{i}_add_{j}'] = self.train[i] + self.train[j]
        #     self.train[f'{i}_sub_{j}'] = self.train[i] - self.train[j]
        #     self.train[f'{i}_mul_{j}'] = self.train[i] * self.train[j]
        #     self.train[f'{i}_div_{j}'] = self.train[i] / (self.train[j] + 0.1)
        #
        #     self.test[f'{i}_add_{j}'] = self.test[i] + self.test[j]
        #     self.test[f'{i}_sub_{j}'] = self.test[i] - self.test[j]
        #     self.test[f'{i}_mul_{j}'] = self.test[i] * self.test[j]
        #     self.test[f'{i}_div_{j}'] = self.test[i] / (self.test[j] + 0.1)


class MainFlagCount(Feature):
    def create_features(self):
        self.train['zero_count'] = (train.filter(regex='(FLAG_|REG_|LIVE_|_AVG|_MODE|_MEDI)') == 0).sum(axis=1)
        self.test['zero_count'] = (test.filter(regex='(FLAG_|REG_|LIVE_|_AVG|_MODE|_MEDI)') == 0).sum(axis=1)
        self.train['one_count'] = (train.filter(regex='(FLAG_|REG_|LIVE_|_AVG|_MODE|_MEDI)') == 1).sum(axis=1)
        self.test['one_count'] = (test.filter(regex='(FLAG_|REG_|LIVE_|_AVG|_MODE|_MEDI)') == 1).sum(axis=1)


class MainExtPairwise(Feature):
    def create_features(self):
        self.train[ext_cols] = train[ext_cols]
        self.test[ext_cols] = test[ext_cols]
        for i, j in itertools.combinations(ext_cols, 2):
            self.train[f'{i}_add_{j}'] = self.train[i] + self.train[j]
            self.train[f'{i}_sub_{j}'] = self.train[i] - self.train[j]
            self.train[f'{i}_mul_{j}'] = self.train[i] * self.train[j]
            self.train[f'{i}_div_{j}'] = self.train[i] / (self.train[j] + 0.1)
            
            self.test[f'{i}_add_{j}'] = self.test[i] + self.test[j]
            self.test[f'{i}_sub_{j}'] = self.test[i] - self.test[j]
            self.test[f'{i}_mul_{j}'] = self.test[i] * self.test[j]
            self.test[f'{i}_div_{j}'] = self.test[i] / (self.test[j] + 0.1)
        
        self.train['EXT_SOURCE_mean'] = self.train[ext_cols].mean()
        self.test['EXT_SOURCE_mean'] = self.test[ext_cols].mean()


class MainExtNull(Feature):
    def create_features(self):
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        for i in range(1, 4):
            self.train[f'EXT_SOURCE_{i}_isnull'] = train[f'EXT_SOURCE_{i}'].isnull()
            self.test[f'EXT_SOURCE_{i}_isnull'] = test[f'EXT_SOURCE_{i}'].isnull()
        for i, j in itertools.combinations_with_replacement(range(1, 4), 2):
            if i == j:
                name = f'EXT_SOURCE_{i}_isnull'
                self.train[name] = train[f'EXT_SOURCE_{i}'].isnull().astype(int)
                self.test[name] = test[f'EXT_SOURCE_{i}'].isnull().astype(int)
            else:
                name = f'EXT_SOURCE_isnull_{i}_or_{j}'
                self.train[name] = (train[f'EXT_SOURCE_{i}'].isnull() | train[f'EXT_SOURCE_{i}'].isnull()).astype(int)
                self.test[name] = (test[f'EXT_SOURCE_{i}'].isnull() | test[f'EXT_SOURCE_{i}'].isnull()).astype(int)
                name = f'EXT_SOURCE_isnull_{i}_and_{j}'
                self.train[name] = (train[f'EXT_SOURCE_{i}'].isnull() & train[f'EXT_SOURCE_{i}'].isnull()).astype(int)
                self.test[name] = (test[f'EXT_SOURCE_{i}'].isnull() & test[f'EXT_SOURCE_{i}'].isnull()).astype(int)
        self.train['EXT_SOURCE_null_cnt'] = train.filter(regex='EXT_').isnull().sum(axis=1)
        self.test['EXT_SOURCE_null_cnt'] = test.filter(regex='EXT_').isnull().sum(axis=1)


class MainDocument(Feature):
    def create_features(self):
        self.train = train.filter(regex='FLAG_DOCUMENT').sum(axis=1).to_frame('document_count')
        self.test = test.filter(regex='FLAG_DOCUMENT').sum(axis=1).to_frame('document_count')


class MainEnquiry(Feature):
    def create_features(self):
        self.train = train.filter(regex='AMT_REQ_').cumsum(axis=1).drop('AMT_REQ_CREDIT_BUREAU_HOUR', axis=1)
        self.test = test.filter(regex='AMT_REQ_').cumsum(axis=1).drop('AMT_REQ_CREDIT_BUREAU_HOUR', axis=1)


if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
    
    with timer('preprocessing'):
        train.replace({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0, 'M': 1, 'F': 0, 'XAP': np.nan, 'XAN': np.nan}, inplace=True)
        test.replace({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0, 'M': 1, 'F': 0, 'XAP': np.nan, 'XAN': np.nan}, inplace=True)
        train.filter(regex='DAYS_').replace(365243, np.nan, inplace=True)
        test.filter(regex='DAYS_').replace(365243, np.nan, inplace=True)
    
    with timer('create dataset'):
        generate_features([
            MainCategory(),
            MainExtRound('main'),
            MainExtNull(),
            MainFlagCount('main'),
            MainDocument('main'),
            MainEnquiry('main', 'cumsum'),
            MainAmountPairwise('main'),
            MainExtPairwise('main'),
            MainDayPairwise('main'),
            MainAmountPerPerson('main')
        ], args.force)
