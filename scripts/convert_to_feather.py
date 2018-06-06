import pandas as pd

from pathlib import Path

for f in Path('../input').glob('*.csv.gz'):
    out = f.parent / f.name.replace('.csv.gz', '.ftr')
    pd.read_csv(f, index_col=0).to_feather(out)

pd.read_feather('../input/application_train.ftr')[['SK_ID_CURR']].to_feather('../input/train_idx.ftr')
pd.read_feather('../input/application_test.ftr')[['SK_ID_CURR']].to_feather('../input/test_idx.ftr')
