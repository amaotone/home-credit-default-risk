from pathlib import Path

# project dirs
PROJECT_PATH = Path(__file__).parent
INPUT = PROJECT_PATH / 'input'
OUTPUT = PROJECT_PATH / 'output'
WORKING = PROJECT_PATH / 'working'
CONFIG = PROJECT_PATH / 'configs'
RESULT = PROJECT_PATH / 'results'
DROPBOX = Path().home() / 'Dropbox/kaggle'

# datasets
TRAIN = INPUT / 'application_train.ftr'
TEST = INPUT / 'application_test.ftr'
BURO = INPUT / 'bureau.ftr'
BURO_BAL = INPUT / 'bureau_balance.ftr'
INST = INPUT / 'installments_payments.ftr'
POS = INPUT / 'POS_CASH_balance.ftr'
PREV = INPUT / 'previous_application.ftr'
CREDIT = INPUT / 'credit_card_balance.ftr'
