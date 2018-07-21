import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from lgbm import run

NAME = Path(__file__).stem
print(NAME)

feats = [
    'main_numeric', 'main_amount_pairwise', 'main_category', 'main_ext_null', 'main_ext_pairwise', 'main_flag_count',
    'prev_basic', 'prev_category_count',
    'prev_category_tfidf',
    'prev_product_combination',
    'main_document', 'main_enquiry', 'main_day_pairwise', 'main_amount_per_person',
    'main_ext_round',
    'prev_null_count',
    'prev_amount_to_main'
]

lgb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'colsample_bytree': 0.95,
    'subsample': 0.85,
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

fit_params = {
    'eval_metric': 'auc',
    'early_stopping_rounds': 100,
    'verbose': 50
}

if __name__ == '__main__':
    run(NAME, feats, lgb_params, fit_params)
