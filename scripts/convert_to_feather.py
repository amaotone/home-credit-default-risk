import pandas as pd

print('train / test')
pd.read_csv('../input/application_train.csv.zip').to_feather('../input/application_train.ftr')
pd.read_csv('../input/application_test.csv.zip').to_feather('../input/application_test.ftr')

print('previous_application')
pd.read_csv('../input/previous_application.csv.zip') \
    .sort_values(['SK_ID_CURR', 'DAYS_DECISION']).reset_index(drop=True) \
    .to_feather('../input/previous_application.ftr')

print('bureau')
pd.read_csv('../input/bureau.csv.zip') \
    .sort_values(['SK_ID_CURR', 'DAYS_CREDIT']).reset_index(drop=True) \
    .to_feather('../input/bureau.ftr')

print('bureau_balance')
pd.read_csv('../input/bureau_balance.csv.zip') \
    .sort_values(['SK_ID_BUREAU', 'MONTHS_BALANCE']).reset_index(drop=True) \
    .to_feather('../input/bureau_balance.ftr')

print('credit_card_balance')
pd.read_csv('../input/credit_card_balance.csv.zip') \
    .sort_values(['SK_ID_CURR', 'MONTHS_BALANCE']).reset_index(drop=True) \
    .to_feather('../input/credit_card_balance.ftr')

print('installments_payments')
pd.read_csv('../input/installments_payments.csv.zip') \
    .sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT']).reset_index(drop=True) \
    .to_feather('../input/installments_payments.ftr')

print('pos_cash')
pd.read_csv('../input/POS_CASH_balance.csv.zip') \
    .sort_values(['SK_ID_CURR', 'MONTHS_BALANCE']).reset_index(drop=True) \
    .to_feather('../input/POS_CASH_balance.ftr')

print('indices')
pd.read_feather('../input/application_train.ftr')[['SK_ID_CURR']].to_feather('../input/train_idx.ftr')
pd.read_feather('../input/application_test.ftr')[['SK_ID_CURR']].to_feather('../input/test_idx.ftr')
