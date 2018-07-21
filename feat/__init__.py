import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../../spica')
from utils import timer
from config import *
from spica.features.base import Feature


def get_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def generate_features(features, overwrite):
    for f in features:
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()


# class Feature(object):
#     def __init__(self, prefix=None, suffix=None):
#         self.prefix = prefix + '_' if prefix else ''
#         self.suffix = '_' + suffix if suffix else ''
#         self.name = re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__).lstrip('_')
#         self.train = pd.DataFrame()
#         self.test = pd.DataFrame()
#         self.train_path = WORKING / f'{self.name}_train.ftr'
#         self.test_path = WORKING / f'{self.name}_test.ftr'
#
#     @property
#     def categorical_features(self):
#         return []
#
#     def run(self):
#         with timer(self.name):
#             self.create_features()
#             # if self.categorical_features:
#             #     self.train.loc[:, self.categorical_features] = self.train.loc[:, self.categorical_features].astype(
#             # str)
#             #     self.test.loc[:, self.categorical_features] = self.test.loc[:, self.categorical_features].astype(
# str)
#             self.train.columns = self.prefix + self.train.columns.str.replace('\s+', '_') + self.suffix
#             self.test.columns = self.prefix + self.test.columns.str.replace('\s+', '_') + self.suffix
#         return self
#
#     def create_features(self):
#         raise NotImplementedError
#
#     def save(self):
#         self.train.to_feather(self.train_path)
#         self.test.to_feather(self.test_path)
#
#     def load(self):
#         self.train = pd.read_feather(self.train_path)
#         self.test = pd.read_feather(self.test_path)


class SubfileFeature(Feature):
    def __init__(self):
        self.df = pd.DataFrame()
        super().__init__()
    
    def run(self):
        with timer(self.name):
            train_idx = pd.read_feather(INPUT / 'train_idx.ftr')
            test_idx = pd.read_feather(INPUT / 'test_idx.ftr')
            self.create_features()
            self.train = train_idx.merge(self.df, left_on='SK_ID_CURR', right_index=True, how='left')[self.df.columns]
            self.test = test_idx.merge(self.df, left_on='SK_ID_CURR', right_index=True, how='left')[self.df.columns]
            self.train.columns = self.prefix + self.train.columns.str.replace('\s+', '_') + self.suffix
            self.test.columns = self.prefix + self.test.columns.str.replace('\s+', '_') + self.suffix
        return self

# class MainFileFeature(Feature, metaclass=ABCMeta):
#     prefix = 'main'
#
#     def create_features(self):
#         self.train = self.feature_impl(train)
#         self.test = self.feature_impl(test)
#
#     def feature_impl(self, df):
#         raise NotImplementedError
