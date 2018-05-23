import time
from datetime import datetime

import pandas as pd
def generate_submit(p, name):
    print('load sample submit')
    sub = pd.read_feather('../data/input/sample_submission.ftr')
    sub.TARGET = p
    filename = f"../data/output/{time.strftime('%y%m%d_%H%M%S')}_{name}.csv.gz"
    print(f'output {filename}')
    sub.to_csv(filename, index=None, compression='gzip')