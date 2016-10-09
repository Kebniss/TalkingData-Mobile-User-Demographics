import os
import numpy as np
import pandas as pd
from os import path
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from dotenv import load_dotenv, find_dotenv
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

%matplotlib inline

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")
VISUALIZATION_DIR = os.environ.get("VISUALIZATION_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")

train_data = pd.read_csv(path.join(FEATURES_DATA_DIR, 'train_dataset.csv'))
labels = train_data['group']
data = train_data.drop('group', 1)
data['phone_brand'], map_brand = pd.factorize(data['phone_brand'])

# ------ DATA EXPLORATION ------
# ignore id and string columns because their values don't come from a distribution
string_cols = [col for col in list(data.columns)
               if any([s in col for s in ['_id', '_most_used', '_app_dly']])]
string_cols.extend(['device_model', 'phone_brand'])
process_cols = [col for col in data.columns if col not in string_cols]
data[process_cols].describe()

corr = data.corr()
# mask = np.zeros_like(corr)
# mask[np.triu_indices_from(mask)] = True
# with sns.axes_style('white'):
#     plt.figure()
#     ax = sns.heatmap(corr, mask=mask, square=True, annot=True, cmap='RdBu')
#     plt.xticks(rotation=45, ha='center')
#     plt.yticks(rotation=0, ha='right')

def class_frequency(labels):
    f = {}
    for l in labels:
        if l in f:
            f[l] += 1
        else:
            f[l] = 1
    return f
f = class_frequency(labels)

X, X_dev, y, y_dev = train_test_split(data,
                                      labels,
                                      test_size=0.30,
                                      random_state=0)
X_1, X_2, y_1, y_2 = train_test_split(X_dev,
                                      y_dev,
                                      test_size=0.50,
                                      random_state=0)

gbm = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, nthread=4)
parameters = {'max_depth': (3, 5, 6, 7, 8, 9, 11)}
f1_scorer = make_scorer(f1_score, greater_is_better=True, average='weighted')
clf = RandomizedSearchCV(gbm,
                         parameters,
                         n_jobs=4,
                         n_iter=20,
                         random_state=42,
                         scoring=f1_scorer)
gbm.fit(X.as_matrix(), y.as_matrix())

y_pred = gbm.predict(X_1.as_matrix())

sig_clf = CalibratedClassifierCV(gbm, method='sigmoid', cv='prefit' )
sig_clf.fit(X_1, y_1)
sig_cdlf_probs = sig_clf.predict_proba(X_2)
sig_score = log_loss(y_2, sig_cdlf_probs)

import pickle
with open(path.join(MODELS_DIR, 'gbm_12c_3d_300est.csv'), 'w') as f:
    pickle.dump(gbm, f)
with open(path.join(MODELS_DIR, 'gbm_12c_3d_300est_clb.csv'), 'w') as f:
    pickle.dump(sig_clf, f)

importances = gbm.feature_importances_
features = data.keys()
features_importance = pd.DataFrame(importances,
                                   index=features,
                                   columns=['Importance']).sort(columns='Importance',
                                                                ascending=False)
