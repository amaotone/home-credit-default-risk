import os
import sys
from logging import getLogger, StreamHandler, DEBUG

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate_submit, load_dataset, send_line_notification
from category_encoders import TargetEncoder
from config import *
from utils import timer, timestamp

sns.set_style('darkgrid')


def run(name, feats, params, fit_params, fill=-9999):
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    
    ch = StreamHandler()
    ch.setLevel(DEBUG)
    
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    
    train = pd.read_feather(str(TRAIN))
    
    with timer('load datasets'):
        X_train, y_train, X_test, cv = load_dataset(feats)
        print('train:', X_train.shape)
        print('test :', X_test.shape)
    
    with timer('impute missing'):
        if fill == 'mean':
            assert X_train.mean().isnull().sum() == 0
            print('fill nan with mean')
            X_train.fillna(X_train.mean(), inplace=True)
            X_test.fillna(X_train.mean(), inplace=True)
        else:
            print(f'fill nan with {fill}')
            X_train.fillna(fill, inplace=True)
            X_test.fillna(fill, inplace=True)
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0
    
    if 'colsample_bytree' in params and params['colsample_bytree'] == 'auto':
        n_samples = X_train.shape[1]
        params['colsample_bytree'] = np.sqrt(n_samples) / n_samples
        print(f'set colsample_bytree = {params["colsample_bytree"]}')
    
    with timer('training'):
        cv_results = []
        cv_df = pd.DataFrame(index=range(len(y_train)), columns=range(cv.get_n_splits()))
        test_df = pd.DataFrame()
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
                model = lgb.LGBMClassifier(**params)
                model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], **fit_params)
            
            p = model.predict_proba(X_val)[:, 1]
            cv_df.loc[val_idx, i] = p
            cv_results.append(roc_auc_score(y_val, p))
            test_df[i] = model.predict_proba(X_tst)[:, 1]
            if feat_df is None:
                feat_df = pd.DataFrame(index=X_trn.columns)
            feat_df[i] = model.feature_importances_
    
    valid_score = np.mean(cv_results)
    message = f"""cv: {valid_score:.5f}
scores: {[round(c, 4) for c in cv_results]}
feats: {feats}
model_params: {params}
fit_params: {fit_params}"""
    
    send_line_notification(message)
    
    with timer('output results'):
        RESULT_DIR = OUTPUT / (timestamp() + '_' + name)
        RESULT_DIR.mkdir()
        
        # output cv prediction
        tmp = pd.DataFrame({'SK_ID_CURR': train['SK_ID_CURR'], 'TARGET': cv_df.mean(axis=1)})
        tmp.to_csv(RESULT_DIR / f'{name}_cv.csv', index=None)
        
        # output test prediction
        pred = test_df.mean(axis=1).ravel()
        generate_submit(pred, f'{name}_{valid_score:.5f}', RESULT_DIR, compression=False)
        
        # output feature importances
        feat_df = (feat_df / feat_df.mean(axis=0)) * 100
        feat_df.mean(axis=1).sort_values(ascending=False).to_csv(RESULT_DIR / 'feats.csv')
        imp = feat_df.mean(axis=1).sort_values(ascending=False)[:50]
        imp[::-1].plot.barh(figsize=(20, 15))
        plt.savefig(str(RESULT_DIR / 'feature_importances.pdf'), bbox_inches='tight')
        
        print('=' * 60)
        print(message)
        print('=' * 60)
