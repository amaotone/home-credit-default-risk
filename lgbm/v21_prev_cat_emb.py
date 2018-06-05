import os
import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate_submit, load_dataset, send_line_notification
from category_encoders import TargetEncoder
from config import *
from utils import timer

sns.set_style('darkgrid')

feats = ['main_numeric', 'main_days_to_years', 'main_days_pairwise', 'main_money_pairwise', 'main_category',
         'main_ext_source_pairwise', 'bureau', 'prev', 'pos', 'credit', 'inst',
         'pos_latest', 'credit_latest', 'inst_latest',
         'bureau_active_and_type_product', 'bureau_active_count',
         'bureau_enddate', 'bureau_amount_pairwise', 'bureau_prolonged', 'main_ext_null',
         'prev_basic', 'prev_category_count', 'prev_category_tfidf', 'prev_category_lda']
rank_average = False


NAME = Path(__file__).stem
print(NAME)

with timer('load datasets'):
    X_train, y_train, X_test, _ = load_dataset(feats)
    cv = StratifiedKFold(5, shuffle=True, random_state=71)
    print('train:', X_train.shape)
    print('test :', X_test.shape)
    # print('feats: ', X_train.columns.tolist())

lgb_params = {
    'n_estimators': 4000,
    'learning_rate': 0.05,
    'num_leaves': 34,
    'colsample_bytree': 0.95,
    'subsample': 0.85,
    'max_depth': 8,
    'reg_alpha': 0.05,
    'reg_lambda': 0.075,
    'min_split_gain': 0.02,
    'min_child_weight': 40,
    'random_state': 71,
    'silent': -1,
    'verbose': -1,
    'n_jobs': -1,
}
fit_params = {
    'eval_metric': 'auc',
    'early_stopping_rounds': 150,
    'verbose': 100
}

X_train.fillna(-9999, inplace=True)
X_test.fillna(-9999, inplace=True)

with timer('training'):
    cv_results = []
    val_series = y_train.copy()
    test_df = pd.DataFrame()
    feat_df = pd.DataFrame(index=X_train.columns)
    for i, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_trn = X_train.iloc[trn_idx].copy()
        y_trn = y_train[trn_idx]
        X_val = X_train.iloc[val_idx].copy()
        y_val = y_train[val_idx]
        X_test_ = X_test.copy()
        print('=' * 30, f'FOLD {i+1}/{cv.get_n_splits()}', '=' * 30)
        with timer('target encoding'):
            cat_cols = X_train.select_dtypes(['object']).columns.tolist()
            te = TargetEncoder(cols=cat_cols)
            X_trn.loc[:, cat_cols] = te.fit_transform(X_trn.loc[:, cat_cols], y_trn)
            X_val.loc[:, cat_cols] = te.transform(X_val.loc[:, cat_cols])
            X_test_.loc[:, cat_cols] = te.transform(X_test.loc[:, cat_cols])
        
        with timer('fit'):
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(X_trn, y_trn, eval_set=[(X_trn, y_trn), (X_val, y_val)], **fit_params)
        
        p = model.predict_proba(X_val)[:, 1]
        val_series.iloc[val_idx] = p
        cv_results.append(roc_auc_score(y_val, p))
        test_df[i] = model.predict_proba(X_test_)[:, 1]
        feat_df[i] = model.feature_importances_

val_df = pd.DataFrame({'TARGET': y_train, 'p': val_series}).to_csv(OUTPUT / f'{NAME}_cv_pred.csv', index=False)
valid_score = np.mean(cv_results)

message = f"""cv: {valid_score: .5f}
feats: {feats}
model_params: {lgb_params}
fit_params: {fit_params}"""
send_line_notification(message)
print('=' * 60)
print(message)
print('=' * 60)

if rank_average:
    pred = test_df.apply(lambda x: rankdata(x)/len(x)).mean(axis=1).ravel()
else:
    pred = test_df.mean(axis=1).ravel()
generate_submit(pred, f'{NAME}_{valid_score:.5f}')

print('output feature importances')
feat_df.mean(axis=1).sort_values(ascending=False).to_csv(OUTPUT / f'{NAME}_feat.csv')
imp = feat_df.mean(axis=1).sort_values(ascending=False)[:50]
imp[::-1].plot.barh(figsize=(20, 15))
plt.savefig(str(DROPBOX / f'{NAME}_feature_importances.pdf'), bbox_inches='tight')
