import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_PATH = Path(__file__).parent
INPUT = PROJECT_PATH / 'input'
OUTPUT = PROJECT_PATH / 'output'
WORKING = PROJECT_PATH / 'working'


def stacking(X, cols, cv_id):
    from sklearn.model_selection import PredefinedSplit
    cv = PredefinedSplit(cv_id)


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def generate_submit(p, name):
    import pandas as pd
    print('load sample submit')
    sub = pd.read_feather(INPUT / 'sample_submission.ftr')
    sub.TARGET = p
    filename = OUTPUT / f"{time.strftime('%y%m%d_%H%M%S')}_{name}.csv.gz"
    print(f'output {filename}')
    sub.to_csv(filename, index=None, compression='gzip')
