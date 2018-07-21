import os
import sys

import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from lgbm import run

sns.set_style('darkgrid')

NAME = Path(__file__).stem
print(NAME)

feats = [
    'main_numeric', 'main_amount_pairwise', 'main_category', 'main_ext_null', 'main_ext_pairwise', 'main_flag_count',
    'bureau', 'pos',
    # 'pos_latest', 'credit_latest',
    'bureau_active_count', 'bureau_enddate', 'bureau_amount_pairwise', 'bureau_prolonged',
    'prev_basic',
    'prev_category_count',
    'prev_category_tfidf',
    'prev_product_combination',
    'main_document', 'main_enquiry', 'main_day_pairwise', 'main_amount_per_person',
    # 'main_ext_round',
    'inst_basic_direct', 'inst_basic_via_prev',
    'inst_latest',
    # 'inst_ewm',
    'inst_basic_direct', 'inst_basic_via_prev',
    'credit_basic_direct', 'credit_basic_via_prev', 'credit_drawing', 'credit_amount_negative_count',
    'credit_null_count', 'pos_null_count', 'bureau_null_count', 'inst_null_count', 'prev_null_count',
    'pos_count', 'pos_decay', 'pos_month_duplicate', 'pos_complete_decay',
    'pos_first_delay_index', 'credit_first_delay_index',
    'prev_amount_to_main'
]

lgb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.05,
    'num_leaves': 35,
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
    'early_stopping_rounds': 200,
    'verbose': 50
}

if __name__ == '__main__':
    run(NAME, feats, lgb_params, fit_params)
