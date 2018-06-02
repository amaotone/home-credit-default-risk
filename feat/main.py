import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat import Feature, get_arguments, generate_features
from utils import timer
from config import *


class MainCategory(Feature):
    def create_features(self):
        self.train = train.filter(regex='(NAME_|_TYPE)')
        self.train.columns = 'main_' + self.train.columns
        self.test = test.filter(regex='(NAME_|_TYPE)')
        self.test.columns = 'main_' + self.test.columns


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


if __name__ == '__main__':
    args = get_arguments('main')
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)
        test = pd.read_feather(TEST)
    
    with timer('preprocessing'):
        train.replace({'XAP': np.nan, 'XAN': np.nan}, inplace=True)
        test.replace({'XAP': np.nan, 'XAN': np.nan}, inplace=True)
    
    with timer('create dataset'):
        generate_features([
            MainCategory(), MainExtNull()
        ], args.force)
