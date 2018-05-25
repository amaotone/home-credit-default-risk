import os
import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_validate, PredefinedSplit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

sns.set_style('darkgrid')

NAME = 'v4'

print('load datasets')
feats = ['basic', 'main_days_to_years', 'main_days_pairwise', 'main_money_pairwise']
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
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'colsample_bytree': 0.8,
    'subsample': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_split_gain': 0.01,
    'min_child_weight': 2,
    'random_state': 77
}

print('5-fold CV')
score = cross_validate(lgb.LGBMClassifier(**lgb_params), X_train, y_train, cv=cv.split(X_train, y_train),
                       scoring='roc_auc', n_jobs=4, verbose=4)

valid_score = score['test_score'].mean()
print('val:', valid_score)

print('train')
model = lgb.LGBMClassifier(**lgb_params)
model.fit(X_train, y_train)

print(f'val = {valid_score};\nfeats = {feats};\nlgb_params = {lgb_params}')
generate_submit(model.predict_proba(X_test)[:, 1], f'{NAME}_{valid_score:.4f}')

print('output feature importances')
feat_df = pd.DataFrame({'importance': model.feature_importances_}, index=X_train.columns).sort_values('importance')
feat_df[-50:].plot.barh(figsize=(20, 15))
plt.savefig(str(Path().home() / f'Dropbox/kaggle/{NAME}_feats.pdf'))
