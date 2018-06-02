import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import target_encoding, timer
from config import *

@timer('target_enc')
def target_enc():
    cat_cols = list(train.filter(regex='(NAME_|_TYPE)').columns)
    train_new, test_new = target_encoding(train[cat_cols], train.TARGET, test[cat_cols], cat_cols, cv_id)
    train_new.to_feather(WORKING / 'main_target_enc_train.ftr')
    test_new.to_feather(WORKING / 'main_target_enc_test.ftr')



if __name__ == '__main__':
    print('load dataset')
    train = pd.read_feather(INPUT / 'application_train.ftr')
    test = pd.read_feather(INPUT / 'application_test.ftr')
    cv_id = pd.read_feather(INPUT / 'cv_id.ftr').cv_id
    target_enc()
