import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import PredefinedSplit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

sns.set_style('darkgrid')

NAME = 'v5'

print('load datasets')
feats = ['basic', 'main_days_to_years', 'main_days_pairwise', 'main_money_pairwise', 'main_target_enc']
dfs = [pd.read_feather(WORKING / f'{f}_train.ftr') for f in feats]
X_train = pd.concat(dfs, axis=1)
dfs = [pd.read_feather(WORKING / f'{f}_test.ftr') for f in feats]
X_test = pd.concat(dfs, axis=1)

y_train = pd.read_feather(WORKING / 'y_train.ftr').TARGET

print(X_train.shape)
print(X_test.shape)

cv_id = pd.read_feather(INPUT / 'cv_id.ftr').cv_id
cv = PredefinedSplit(cv_id)

lgb_params = {
    'n_estimators': 4000,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'colsample_bytree': 0.8,
    'subsample': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_split_gain': 0.01,
    'min_child_weight': 2,
    'random_state': 77,
    'silent': True
}

print('5-fold CV')
cv_results = []
pred_df = pd.DataFrame()
feat_df = pd.DataFrame(index=X_train.columns)
for i, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_trn = X_train.iloc[trn_idx]
    y_trn = y_train[trn_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_train[val_idx]
    
    print('='*30, f'FOLD {i+1}', '='*30)
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_trn, y_trn, eval_set=[(X_trn, y_trn), (X_val, y_val)], eval_metric='auc',
              early_stopping_rounds=100, verbose=100)
    
    cv_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    cv_results.append(cv_score)
    pred_df[i] = model.predict_proba(X_test)[:, 1]
    feat_df[i] = model.feature_importances_

valid_score = np.mean(cv_results)

print('=' * 60)
print(f'val = {valid_score};\nfeats = {feats};\nlgb_params = {lgb_params}')
print('=' * 60)

pred = pred_df.mean(axis=1).values.ravel()
generate_submit(pred, f'{NAME}_{valid_score:.4f}')

print('output feature importances')
feat_df.to_csv(f'{NAME}_feat.csv')
# feat_df = pd.DataFrame({'importance': model.feature_importances_}, index=X_train.columns).sort_values('importance')
# feat_df[-50:].plot.barh(figsize=(20, 15))
# plt.savefig(str(Path().home() / f'Dropbox/kaggle/{NAME}_feats.pdf'))
