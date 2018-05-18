import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold

SEED = 77

files = Path('.').glob('../data/input/application_*.ftr')
for f in files:
    print(f)
    globals()[f.stem] = pd.read_feather(f)

train = application_train
test = application_test

categorical_columns = [
    'NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START',
    'ORGANIZATION_TYPE',
    'FONDKAPREMONT_MODE',
    'HOUSETYPE_MODE',
    'WALLSMATERIAL_MODE',
    'EMERGENCYSTATE_MODE'
]
enc = OrdinalEncoder(cols=categorical_columns, verbose=1)
train[categorical_columns] = enc.fit_transform(train[categorical_columns])
test[categorical_columns] = enc.transform(test[categorical_columns])

X_train = train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y_train = train.TARGET.values
X_test = test.drop(['SK_ID_CURR'], axis=1)

params = {
    'metric': ['auc'],
    'learning_rate': [0.1],
    'num_leaves': [i * 10 for i in range(2, 6)],
    'min_data_in_leaf': [5, 10, 15, 20],
    'random_state': [SEED],
    'verbose': [1]
}

cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
model = GridSearchCV(lgb.LGBMClassifier(), params, scoring='roc_auc', n_jobs=3, cv=cv.split(X_train, y_train),
                     verbose=3)
model.fit(X_train, y_train)

pred = model.predict_proba(X_test)[:, 1]
plt.hist(pred, bins=50)
plt.show()

feat_df = pd.DataFrame({'importance': model.best_estimator_.feature_importances_}, index=X_train.columns).sort_values(
    'importance', ascending=False)
feat_df[:30].plot.bar(figsize=(20, 5))

sample_submission.TARGET = pred
sample_submission.to_csv(f"{time.strftime('%y%m%d_%H%M%S')}_submission.csv.gz", index=None, compression='gzip')
