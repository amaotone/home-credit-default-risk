import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from config import *

train = pd.read_feather(INPUT / 'application_train.ftr')
cv = KFold(5, shuffle=True, random_state=71)
cv_id = np.zeros(len(train))
for i, (_, val) in enumerate(cv.split(train, train.TARGET)):
    cv_id[val] = i

print(f'save {INPUT/"cv_id.ftr"}')
pd.DataFrame(cv_id, columns=['cv_id']).to_feather(INPUT / 'cv_id.ftr')
