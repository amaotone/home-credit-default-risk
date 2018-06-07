import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.random.seed(71)
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

from utils import load_dataset, timer
from nn.gaussrank import gauss_rank
from config import *

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

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


def get_denoising_autoencoders(X_train, hiddens=None, drop_ratio=.15):
    hiddens = hiddens if hiddens else [500]
    x_in = Input((X_train.shape[1],), name='input')
    h = Dropout(drop_ratio)(x_in)
    for i, dim in enumerate(hiddens):
        h = Dense(dim, activation='relu', name=f'hidden_{i}')(h)
    x_out = Dense(X_train.shape[1], activation='linear', name='out')(h)
    model = Model(x_in, x_out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    get_emb = Model(inputs=model.input, outputs=model.get_layer(f'hidden_{len(hiddens)-1}').output)
    return model, get_emb


if use_cache:
    cat_cols = X_train.select_dtypes(['object']).columns
    X_train = pd.read_feather(WORKING / 'X_train_tmp.ftr')
    X_test = pd.read_feather(WORKING / 'X_test_tmp.ftr')
else:
    with timer('preprocessing'):
        num_cols = X_train.select_dtypes(exclude=['object']).columns
        mean = X_train.loc[:, num_cols].mean()
        X_train.loc[:, num_cols], X_test.loc[:, num_cols] = gauss_rank([X_train, X_test], num_cols, 0, 0.001)

X_train.columns = X_train.columns.str.replace('\s+', '_')
X_test.columns = X_test.columns.str.replace('\s+', '_')
X_train.columns = X_train.columns.str.replace('(,|\+|\(|:|\))', '')
X_test.columns = X_test.columns.str.replace('(,|\+|\(|:|\))', '')

# use only continuous columns
X_train = X_train.select_dtypes(exclude=['object']).fillna(0)
X_test = X_test.select_dtypes(exclude=['object']).fillna(0)

with timer('training'):
    model, get_emb = get_denoising_autoencoders(X_train, [1500, 1500, 1500], 0.2)
    model.summary()
    earlystopper = EarlyStopping(patience=20)
    csv = CSVLogger(str(OUTPUT/'dae_log.csv'))
    checkpointer = ModelCheckpoint(str(OUTPUT / 'dae_02_1500_1500_1500.h5'), verbose=1, save_best_only=True)
    model.fit(X_train.values, X_train.values, batch_size=256, epochs=1000, validation_split=.2,
              callbacks=[earlystopper, checkpointer, csv])

with timer('save embedding'):
    emb_train = get_emb.predict(X_train.values)
    emb_test = get_emb.predict(X_test.values)
    X_train_df = pd.DataFrame(emb_train).rename(columns=lambda x: f'dae_emb_{x}')
    X_train_df = X_train_df.loc[:, X_train_df.std() != 0]
    X_test_df = pd.DataFrame(emb_test).rename(columns=lambda x: f'dae_emb_{x}')
    X_test_df = X_test_df.loc[:, X_train_df.columns]
    X_train_df.to_feather(WORKING / 'dae_emb_train.ftr')
    X_test_df.to_feather(WORKING / 'dae_emb_test.ftr')
    
print(mean_squared_error(X_test.values.ravel(), model.predict(X_test.values).ravel()))
