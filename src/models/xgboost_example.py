import os
import numpy as np
import pandas as pd
from os import path
import xgboost as xgb
import seaborn as sns
from time import time
from scipy import sparse, io
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

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")

data = io.mmread(path.join(FEATURES_DATA_DIR, 'train_set_dd')).tocsr()
gatrain = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_train.csv'),
                      index_col='device_id')
labels = gatrain['group']
targetencoder = LabelEncoder().fit(labels)
y = targetencoder.transform(labels)
nclasses = len(targetencoder.classes_)

def score(clf, X, y, nclasses, random_state=None):
    kf=StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
    pred = np.zeros((y.shape[0], nclasses))
    for itrain, itest in kf:
        Xtr, Xte = X[itrain, :], X[itest, :]
        ytr, yte = y[itrain], y[itest]
        clf.fit(Xtr, ytr)
        pred[itest, :] = clf.predict_proba(Xte)
        return log_loss(yte, pred[itest, :])
    return log_loss(y, pred)

X, X_dev, y, y_dev = train_test_split(data,
                                      labels,
                                      test_size=0.20,
                                      random_state=0)

parameters = {'max_depth': (3, 5, 6, 7, 8, 9, 11)}
f1_scorer = make_scorer(f1_score, greater_is_better=True, average='weighted')

gbm = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, nthread=4)
clf = RandomizedSearchCV(gbm,
                         parameters,
                         n_jobs=4,
                         n_iter=20,
                         random_state=42,
                         scoring=f1_scorer)
t0 = time()
gbm.fit(X, y)
t1 = time()
(t1-t0)/60

#y_pred = gbm.predict(X.as_matrix())

sig_clf = CalibratedClassifierCV(gbm, method='sigmoid', cv='prefit' )
sig_clf.fit(X_dev, y_dev)
#sig_cdlf_probs = sig_clf.predict_proba(X_2)
#sig_score = log_loss(y_2, sig_cdlf_probs)

import pickle
with open(path.join(MODELS_DIR, 'gbm_12c_3d_300est.pkl'), 'wb') as f:
    pickle.dump(gbm, f)
with open(path.join(MODELS_DIR, 'gbm_12c_3d_300est_clb.pkl'), 'wb') as f:
    pickle.dump(sig_clf, f)

importances = gbm.feature_importances_
features = data.keys()
features_importance = pd.DataFrame(importances,
                                   index=features,
                                   columns=['Importance']).sort(columns='Importance',
                                                                ascending=False)
