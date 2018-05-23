import time
from pathlib import Path

PROJECT_PATH = Path(__file__).parent
INPUT = PROJECT_PATH / 'input'
OUTPUT = PROJECT_PATH / 'output'
WORKING = PROJECT_PATH / 'working'


def generate_submit(p, name):
    import pandas as pd
    print('load sample submit')
    sub = pd.read_feather('../data/input/sample_submission.ftr')
    sub.TARGET = p
    filename = f"../data/output/{time.strftime('%y%m%d_%H%M%S')}_{name}.csv.gz"
    print(f'output {filename}')
    sub.to_csv(filename, index=None, compression='gzip')
