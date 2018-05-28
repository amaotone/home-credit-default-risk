import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat import Feature
from utils import timer
from config import *


class PosLatest(Feature):
    @property
    def categorical_features(self):
        return ['NAME_CONTRACT_STATUS']
    
    def create_features(self):
        df = pos.groupby('SK_ID_CURR').last().drop('SK_ID_PREV', axis=1)
        df.columns = 'pos_' + df.columns + '_latest'
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


class PosCount(Feature):
    def create_features(self):
        df = pos.groupby('SK_ID_CURR').SK_ID_PREV.count().to_frame('pos_count')
        self.train = train.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]
        self.test = test.merge(df, left_on='SK_ID_CURR', right_index=True, how='left')[df.columns]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='POS CASH features')
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()
    
    with timer('load dataset'):
        train = pd.read_feather(TRAIN)[['SK_ID_CURR']]
        test = pd.read_feather(TEST)[['SK_ID_CURR']]
        pos = pd.read_feather(POS)
    
    with timer('preprocessing'):
        pos = pos.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE']).reset_index(drop=True)
        pos.loc[:, pos.columns.str.startswith('SK_DPD')] = np.log1p(pos.filter(regex='^SK_DPD'))
    
    with timer('create dataset'):
        features = [PosLatest()]
        for f in features:
            if args.force or not Path(f.train_path).exists() or not Path(f.test_path).exists():
                f.run().save()
            else:
                print(f.name, 'skipped')
