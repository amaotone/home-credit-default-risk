import os
import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate_submit, load_dataset
from config import *

sns.set_style('darkgrid')

NAME = 'v9_sellerplace'

print('load datasets')
feats = ['main_numeric', 'main_days_to_years', 'main_days_pairwise', 'main_money_pairwise', 'main_target_enc',
         'main_ext_source_pairwise', 'bureau', 'prev', 'pos', 'credit', 'inst']
X_train, y_train, X_test, cv = load_dataset(feats)

print('train:', X_train.shape)
print('test :', X_test.shape)

lgb_params = {
    'n_estimators': 4000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_split_gain': 0.01,
    'min_child_weight': 2,
    'random_state': 77,
    'silent': True,
    'n_jobs': -1,
}
fit_params = {
    'eval_metric': 'auc',
    'early_stopping_rounds': 100,
    'verbose': 100
}

cv_results = []
pred_df = pd.DataFrame()
feat_df = pd.DataFrame(index=X_train.columns)
for i, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_trn = X_train.iloc[trn_idx]
    y_trn = y_train[trn_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_train[val_idx]
    
    print('=' * 30, f'FOLD {i+1}/{cv.get_n_splits()}', '=' * 30)
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_trn, y_trn, eval_set=[(X_trn, y_trn), (X_val, y_val)], **fit_params)
    
    cv_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    cv_results.append(cv_score)
    pred_df[i] = model.predict_proba(X_test)[:, 1]
    feat_df[i] = model.feature_importances_

valid_score = np.mean(cv_results)

print('=' * 60)
print(f'cv = {valid_score}')
print(f'feats = {feats}')
print(f'lgb_params = {lgb_params}')
print(f'fit_params = {fit_params}')
print('=' * 60)

pred = pred_df.mean(axis=1).values.ravel()
generate_submit(pred, f'{NAME}_{valid_score:.4f}')

print('output feature importances')
imp = feat_df.mean(axis=1).sort_values(ascending=False)[:50]
imp[::-1].plot.barh(figsize=(20, 15))
plt.savefig(str(DROPBOX / f'{NAME}_feature_importances.pdf'))
