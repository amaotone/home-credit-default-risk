import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.random.seed(71)
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Input, concatenate, Dense, Embedding, Flatten, BatchNormalization, Dropout
from keras.models import Model
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from utils import generate_submit, load_dataset, send_line_notification, timer
from nn.gaussrank import gauss_rank
from config import *

from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping

sns.set_style('darkgrid')

feats = ['main_numeric', 'main_days_to_years', 'main_days_pairwise', 'main_money_pairwise', 'main_category',
         'main_ext_source_pairwise', 'bureau', 'prev', 'pos', 'credit', 'inst',
         'pos_latest', 'credit_latest', 'inst_latest',
         'bureau_active_count', 'bureau_enddate', 'bureau_amount_pairwise', 'bureau_prolonged',
         'main_ext_null',
         'prev_basic', 'prev_category_count', 'prev_category_tfidf',
         'main_document', 'main_enquiry']
rank_average = False
use_cache = True

NAME = Path(__file__).stem
print(NAME)

with timer('load datasets'):
    X_train, y_train, X_test, _ = load_dataset(feats)
    cv = StratifiedKFold(5, shuffle=True, random_state=71)
    print('train:', X_train.shape)
    print('test :', X_test.shape)
    # print('feats: ', X_train.columns.tolist())


# lgb_params = {
#     'n_estimators': 4000,
#     'learning_rate': 0.05,
#     'num_leaves': 34,
#     'colsample_bytree': 0.95,
#     'subsample': 0.85,
#     'max_depth': 8,
#     'reg_alpha': 0.05,
#     'reg_lambda': 0.075,
#     'min_split_gain': 0.02,
#     'min_child_weight': 40,
#     'random_state': 71,
#     'silent': -1,
#     'verbose': -1,
#     'n_jobs': -1,
# }
# fit_params = {
#     'eval_metric': 'auc',
#     'early_stopping_rounds': 150,
#     'verbose': 100
# }

def get_keras_model(X_train, cat_cols):
    inputs = []
    xs = []
    for f in X_train.columns:
        x_in = Input(shape=(1,), name=f)
        inputs.append(x_in)
        if f in cat_cols:
            x = Embedding(X_train[f].max() + 1, 2, input_length=1, name=f + '_emb')(x_in)
            x = Flatten()(x)
        else:
            x = Dense(2)(x_in)
        xs.append(x)
    h = concatenate(xs)
    h = BatchNormalization()(h)
    h = Dropout(.2)(h)
    h = Dense(500, activation='relu')(h)
    h = BatchNormalization()(h)
    h = Dropout(.2)(h)
    out = Dense(1, activation='sigmoid', name='out')(h)
    model = Model(inputs, [out])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


if use_cache:
    cat_cols = X_train.select_dtypes(['object']).columns
    X_train = pd.read_feather(WORKING / 'X_train_tmp.ftr')
    X_test = pd.read_feather(WORKING / 'X_test_tmp.ftr')
else:
    with timer('preprocessing'):
        num_cols = X_train.select_dtypes(exclude=['object']).columns
        mean = X_train.loc[:, num_cols].mean()
        X_train.loc[:, num_cols], X_test.loc[:, num_cols] = gauss_rank([X_train, X_test], num_cols, 0, 0.001)
        
        cat_cols = X_train.select_dtypes(['object']).columns
        lbl = LabelEncoder()
        for f in cat_cols:
            X_train[f] = lbl.fit_transform(X_train[f].fillna('NaN'))
            X_test[f] = lbl.transform(X_test[f].fillna('NaN'))

X_train.columns = X_train.columns.str.replace('\s+', '_')
X_test.columns = X_test.columns.str.replace('\s+', '_')
X_train.columns = X_train.columns.str.replace('(,|\+|\(|:|\))', '')
X_test.columns = X_test.columns.str.replace('(,|\+|\(|:|\))', '')

with timer('training'):
    cv_results = []
    val_series = y_train.copy()
    test_df = pd.DataFrame()
    feat_df = None
    for i, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_trn = X_train.iloc[trn_idx].copy()
        X_trn = [X_trn[f].values.astype(np.float32) for f in X_trn.columns]
        y_trn = y_train[trn_idx].values.reshape(-1, 1).astype(np.float32)
        X_val = X_train.iloc[val_idx].copy()
        X_val = [X_val[f].values.astype(np.float32) for f in X_val.columns]
        y_val = y_train[val_idx].values.reshape(-1, 1).astype(np.float32)
        X_tst = X_test.copy()
        X_tst = [X_tst[f].values.astype(np.float32) for f in X_tst.columns]
        print('=' * 30, f'FOLD {i+1}/{cv.get_n_splits()}', '=' * 30)
        
        with timer('fit'):
            model = get_keras_model(X_train, cat_cols)
            # early_stopping = EarlyStopping()
            # roc = ROC((X_trn, y_trn), (X_val, y_val))
            earlystopper = EarlyStopping(patience=0)
            model.fit(X_trn, y_trn, batch_size=512, epochs=100, validation_data=(X_val, y_val),
                      callbacks=[earlystopper])
        
        p = model.predict(X_val).ravel()
        val_series.iloc[val_idx]
        score = roc_auc_score(y_val, p)
        print('cv:', score)
        cv_results.append(score)
        test_df[i] = model.predict(X_tst).ravel()

val_df = pd.DataFrame({'TARGET': y_train, 'p': val_series}).to_csv(OUTPUT / f'{NAME}_cv_pred.csv', index=False)
valid_score = np.mean(cv_results)

message = f"""cv: {valid_score: .5f}
feats: {feats}"""
send_line_notification(message)
print('=' * 60)
print(message)
print('=' * 60)

if rank_average:
    pred = test_df.apply(lambda x: rankdata(x) / len(x)).mean(axis=1).ravel()
else:
    pred = test_df.mean(axis=1).ravel()
generate_submit(pred, f'{NAME}_{valid_score:.5f}')
