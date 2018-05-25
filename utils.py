import time
from contextlib import contextmanager

import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import PredefinedSplit
from tqdm import tqdm

from config import *


def target_encoding(X_train, y_train, X_test, cols, cv_id):
    cols = list(cols)
    train_new = X_train.copy()
    test_new = X_test.copy()
    test_new[:] = 0
    cv = PredefinedSplit(cv_id)
    X_train.index = X_train.index.astype(int)
    for trn_idx, val_idx in tqdm(cv.split(X_train), total=cv.get_n_splits()):
        enc = TargetEncoder(cols=cols)
        enc.fit(X_train.iloc[trn_idx], y_train[trn_idx])
        train_new.iloc[val_idx] = enc.transform(X_train.iloc[val_idx])
        test_new += enc.transform(X_test)
    test_new /= cv.get_n_splits()
    train_new = train_new[cols]
    test_new = test_new[cols]
    train_new.columns = train_new.columns + '_target'
    test_new.columns = test_new.columns + '_target'
    print(list(train_new.columns))
    return train_new, test_new


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def generate_submit(p, name):
    print('load sample submit')
    sub = pd.read_feather(INPUT / 'sample_submission.ftr')
    sub.TARGET = p
    filename = OUTPUT / f"{time.strftime('%y%m%d_%H%M%S')}_{name}.csv.gz"
    print(f'output {filename}')
    sub.to_csv(filename, index=None, compression='gzip')
