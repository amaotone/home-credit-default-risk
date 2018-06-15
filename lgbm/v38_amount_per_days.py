import os
import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate_submit, load_dataset, send_line_notification
from category_encoders import TargetEncoder
from config import *
from utils import timer, timestamp

sns.set_style('darkgrid')

NAME = Path(__file__).stem
print(NAME)

feats = [
    'main_numeric', 'main_amount_pairwise', 'main_category', 'main_ext_null', 'main_ext_pairwise', 'main_flag_count',
    'bureau', 'pos', 'pos_latest', 'credit_latest',
    'bureau_active_count', 'bureau_enddate', 'bureau_amount_pairwise', 'bureau_prolonged',
    'prev_basic', 'prev_category_count', 'prev_category_tfidf', 'prev_product_combination',
    'main_document', 'main_enquiry', 'main_day_pairwise', 'main_amount_per_person', 'main_ext_round',
    'inst_basic_direct', 'inst_basic_via_prev', 'inst_latest', 'inst_ewm', 'inst_basic_direct', 'inst_basic_via_prev',
    'credit_basic_direct', 'credit_basic_via_prev', 'credit_drawing', 'credit_amount_negative_count',
    'credit_null_count', 'pos_null_count', 'bureau_null_count', 'inst_null_count', 'prev_null_count',
]

lgb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.1,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 34,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'reg_alpha': 0.05,
    'reg_lambda': 0.075,
    'min_split_gain': 0.02,
    'min_child_weight': 40,
    'random_state': 71,
    # 'boosting_type': 'dart',
    'silent': -1,
    'verbose': -1,
    'n_jobs': -1,
}


def learning_rates(ratio=0.995, lr_min=0.02):
    tmp = lgb_params['learning_rate']
    res = []
    for _ in range(lgb_params['n_estimators']):
        res.append(tmp)
        tmp *= ratio
        tmp = max(tmp, lr_min)
    return res


fit_params = {
    'eval_metric': 'auc',
    'early_stopping_rounds': 150,
    'verbose': 100
}

with timer('load datasets'):
    X_train, y_train, X_test, _ = load_dataset(feats)
    cv = StratifiedKFold(5, shuffle=True, random_state=71)
    print('train:', X_train.shape)
    print('test :', X_test.shape)

with timer('drop low importance feats'):
    ref = pd.read_csv('/home/ubuntu/kaggle-home-credit/output/180615_014745_v32_credit_drawing/feats.csv', index_col=0,
                      header=None)
    drop_cols = ref[1][ref[1] < 1].index
    drop_cols = drop_cols[drop_cols.isin(X_train.columns)]
    X_train.drop(drop_cols, axis=1, inplace=True)
    X_test.drop(drop_cols, axis=1, inplace=True)
    print('train:', X_train.shape)
    print('test :', X_test.shape)

with timer('add PCA feats'):
    pca = PCA(n_components=10, random_state=71)
    X_pca_train = pd.DataFrame(pca.fit_transform(X_train.select_dtypes(exclude=['object']).fillna(0)),
                               index=X_train.index, columns=['pca_' + str(i) for i in range(10)])
    X_train = pd.concat([X_train, X_pca_train], axis=1)
    X_pca_test = pd.DataFrame(pca.transform(X_test.select_dtypes(exclude=['object']).fillna(0)),
                              index=X_test.index, columns=['pca_' + str(i) for i in range(10)])
    X_test = pd.concat([X_test, X_pca_test], axis=1)

with timer('impute missing'):
    X_train.fillna(-9999, inplace=True)
    X_test.fillna(-9999, inplace=True)
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0

with timer('training'):
    cv_results = []
    val_series = y_train.copy()
    test_df = pd.DataFrame()
    eval_df = pd.DataFrame(np.zeros((lgb_params['n_estimators'], cv.get_n_splits())), columns=range(cv.get_n_splits()))
    feat_df = None
    for i, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_trn = X_train.iloc[trn_idx].copy()
        y_trn = y_train[trn_idx]
        X_val = X_train.iloc[val_idx].copy()
        y_val = y_train[val_idx]
        X_tst = X_test.copy()
        print('=' * 30, f'FOLD {i+1}/{cv.get_n_splits()}', '=' * 30)
        
        with timer('target encoding'):
            cat_cols = X_trn.select_dtypes(['object']).columns.tolist()
            te = TargetEncoder(cols=cat_cols)
            X_trn.loc[:, cat_cols] = te.fit_transform(X_trn.loc[:, cat_cols], y_trn)
            X_val.loc[:, cat_cols] = te.transform(X_val.loc[:, cat_cols])
            X_tst.loc[:, cat_cols] = te.transform(X_test.loc[:, cat_cols])
        
        with timer('fit'):
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(X_trn, y_trn,
                      callbacks=[lgb.reset_parameter(learning_rate=learning_rates())],
                      eval_set=[(X_val, y_val)], **fit_params)
        
        p = model.predict_proba(X_val)[:, 1]
        val_series.iloc[val_idx] = p
        cv_results.append(roc_auc_score(y_val, p))
        test_df[i] = model.predict_proba(X_tst)[:, 1]
        if feat_df is None:
            feat_df = pd.DataFrame(index=X_trn.columns)
        feat_df[i] = model.feature_importances_
        eval_df[i][:len(model.evals_result_['valid_0']['auc'])] = model.evals_result_['valid_0']['auc']
    
    # print('=' * 30, 'REFIT', '=' * 30)
    # X_trn = X_train
    # y_trn = y_train
    # X_tst = X_test
    # with timer('target encoding'):
    #     cat_cols = X_trn.select_dtypes(['object']).columns.tolist()
    #     te = TargetEncoder(cols=cat_cols)
    #     X_trn.loc[:, cat_cols] = te.fit_transform(X_trn.loc[:, cat_cols], y_trn)
    #     X_tst.loc[:, cat_cols] = te.transform(X_test.loc[:, cat_cols])
    #
    # with timer('fit'):
    #     best_iter = round(eval_df.mean(axis=1).idxmax() * 1.1)
    #     print('best iter:', best_iter)
    #     lgb_params['n_estimators'] = best_iter
    #     model = lgb.LGBMClassifier(**lgb_params)
    #     model.fit(X_trn, y_trn)
    #
    # test_df[cv.get_n_splits()] = model.predict_proba(X_tst)[:, 1]

valid_score = np.mean(cv_results)
message = f"""cv: {valid_score:.5f}
feats: {feats}
model_params: {lgb_params}
fit_params: {fit_params}"""
send_line_notification(message)
print('=' * 60)
print(message)
print('=' * 60)

with timer('output results'):
    RESULT_DIR = OUTPUT / (timestamp() + '_' + NAME)
    RESULT_DIR.mkdir()
    
    val_df = pd.DataFrame({'TARGET': y_train, 'p': val_series}).to_csv(RESULT_DIR / f'{NAME}_cv_pred.csv', index=False)
    
    pred = test_df.mean(axis=1).ravel()
    generate_submit(pred, f'{NAME}_{valid_score:.5f}', RESULT_DIR)
    
    print('output feature importances')
    feat_df.mean(axis=1).sort_values(ascending=False).to_csv(RESULT_DIR / 'feats.csv')
    imp = feat_df.mean(axis=1).sort_values(ascending=False)[:50]
    imp[::-1].plot.barh(figsize=(20, 15))
    plt.savefig(str(RESULT_DIR / 'feature_importances.pdf'), bbox_inches='tight')
