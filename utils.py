import time
from contextlib import contextmanager

import pandas as pd
import requests
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


def load_dataset(names):
    dfs = [pd.read_feather(WORKING / f'{f}_train.ftr') for f in names]
    X_train = pd.concat(dfs, axis=1)
    y_train = pd.read_feather(WORKING / 'y_train.ftr').TARGET
    dfs = [pd.read_feather(WORKING / f'{f}_test.ftr') for f in names]
    X_test = pd.concat(dfs, axis=1)
    cv_id = pd.read_feather(INPUT / 'cv_id.ftr').cv_id
    cv = PredefinedSplit(cv_id)
    return X_train, y_train, X_test, cv


def send_line_notification(message):
    line_token = 'BAgnMPDq3jOv2OnMabEQmMlz3qI4glluFuXBaTSU37W'  # 終わったら無効化する
    line_notify_api = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}  # 発行したトークン
    requests.post(line_notify_api, data=payload, headers=headers)


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
