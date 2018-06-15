import numpy as np
import pandas as pd
from tqdm import tqdm

files = ['application_train', 'bureau', 'credit_card_balance', 'previous_application', 'POS_CASH_balance',
         'installments_payments']
for f in tqdm(files):
    df = pd.read_feather('../input/' + f + '.ftr')
    idx = np.random.choice(df.SK_ID_CURR.unique(), 100)
    df.query('SK_ID_CURR in @idx').to_csv('../sampled/' + f + '_sampled.csv')
