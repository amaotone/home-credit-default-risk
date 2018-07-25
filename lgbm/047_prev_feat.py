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
    # 'main_null_count',
    # 'main_numeric',
    'main_amount',
    'main_amount_req',
    # 'main_house'
    'main_gender',
    'main_days',
    'main_social',
    'main_ext',
    # 'main_document',
    # 'main_flag',
    'main_region',
    'main_car',
    'main_application',
    'main_amount_pairwise',
    'main_category',
    # 'main_ext_pairwise',
    'main_document_count',
    # 'main_enquiry',
    'main_day_pairwise',
    'main_family',
    'main_area_income',
    'main_ext_mean',
    'main_region_as_category',
    'buro_count',
    'buro_balance_count',
    'buro_target',
    'buro_active',
    'buro_overdue',
    'buro_prepayment',
    'buro_days_ratio',
    'buro_future_expires',
    'buro_null_count',
    'buro_period_overlap',
    'buro_prolong',
    'prev_count',
    'prev_null_count',
    'prev_target',
    'prev_sellerplace_area_target',
    # 'bureau', 'pos',
    # 'pos_latest', 'credit_latest',
    # 'bureau_active_count',
    # 'bureau_enddate',
    # 'bureau_amount_pairwise',
    # 'bureau_prolonged',
    # 'prev_basic',
    # 'prev_category_count',
    # 'prev_category_tfidf',
    # 'prev_product_combination',
    # 'main_ext_round',
    # 'inst_basic_direct',
    # 'inst_basic_via_prev',
    # 'inst_latest',
    # 'inst_ewm',
    # 'credit_basic_direct',
    # 'credit_basic_via_prev',
    # 'credit_drawing',
    # 'credit_amount_negative_count',
    # 'credit_null_count', 'pos_null_count', 'bureau_null_count', 'inst_null_count',
    # 'pos_count', 'pos_decay', 'pos_month_duplicate', 'pos_complete_decay',
    # 'pos_first_delay_index', 'credit_first_delay_index',
    # 'prev_null_count', 'prev_amount_to_main'
]

lgb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.03,
    'num_leaves': 20,
    'max_depth': 8,
    'colsample_bytree': 0.3,
    'subsample': 0.8,
    'reg_alpha': 0.04,
    'reg_lambda': 0.075,
    'min_split_gain': 0.02,
    'min_child_weight': 10,
    'random_state': 71,
    # 'boosting_type': 'dart',
    'silent': -1,
    'verbose': -1,
    'n_jobs': -1,
    'metric': 'auc',
}

fit_params = {
    'eval_metric': 'auc',
    'early_stopping_rounds': 300,
    'verbose': 100,
}

if __name__ == '__main__':
    run(NAME, feats, lgb_params, fit_params)
