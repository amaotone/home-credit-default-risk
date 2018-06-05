import pandas as pd

pd.read_feather('../input/application_train.ftr')[['SK_ID_CURR']].to_feather('../input/train_idx.ftr')
pd.read_feather('../input/application_test.ftr')[['SK_ID_CURR']].to_feather('../input/test_idx.ftr')
