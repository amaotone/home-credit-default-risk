import os
import re
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import timer
from config import *


class Feature(object):
    def __init__(self):
        self.name = re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__).lstrip('_')
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = WORKING / f'{self.name}_train.ftr'
        self.test_path = WORKING / f'{self.name}_test.ftr'
    
    @property
    def categorical_features(self):
        return []
    
    def run(self):
        with timer(self.name):
            self.create_features()
            if self.categorical_features:
                self.train[self.categorical_features] = self.train[self.categorical_features].astype(str)
                self.test[self.categorical_features] = self.test[self.categorical_features].astype(str)
        return self
    
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.train.to_feather(self.train_path)
        self.test.to_feather(self.test_path)

    def load(self):
        self.train = pd.read_feather(self.train_path)
        self.test = pd.read_feather(self.test_path)