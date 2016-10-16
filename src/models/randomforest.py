''' THIS MODEL WITH DD FEATURES OBTAINS A SCORE OF 2.33 ON KAGGLE'''
''' THIS MODEL WITH MY FEATURES OBTAINS A SCORE OF 2.34 ON KAGGLE'''

import os
import numpy as np
import pandas as pd
from os import path
import seaborn as sns
from time import time
from scipy import sparse, io
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from dotenv import load_dotenv, find_dotenv
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
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

data = io.mmread(path.join(FEATURES_DATA_DIR, 'train_set_dd_enh')).tocsr()
gatrain = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_train.csv'),
                      index_col='device_id')
labels = gatrain['group']
targetencoder = LabelEncoder().fit(labels)
y = targetencoder.transform(labels)
nclasses = len(targetencoder.classes_)

def score(clf, X, y, nclasses, random_state=None):
    kf=StratifiedKFold(y, n_folds=1, shuffle=True, random_state=random_state)
    pred = np.zeros((y.shape[0], nclasses))
    for itrain, itest in kf:
        Xtr, Xte = X[itrain, :], X[itest, :]
        ytr, yte = y[itrain], y[itest]
        clf.fit(Xtr, ytr)
        pred[itest, :] = clf.predict_proba(Xte)
        return log_loss(yte, pred[itest])
    return log_loss(y, pred)

X, X_dev, y, y_dev = train_test_split(data,
                                      labels,
                                      test_size=0.20,
                                      random_state=0)
# X_1, X_2, y_1, y_2 = train_test_split(X_dev,
#                                       y_dev,
#                                       test_size=0.30,
#                                       random_state=0)
parameters = {'max_depth': (3, 5, 6, 7, 8, 9, 11),
              'min_samples_split': (50, 100, 500, 1000)}
f1_scorer = make_scorer(f1_score, greater_is_better=True, average='weighted')
rfc = RandomForestClassifier(n_estimators=200,  n_jobs=4)
clf = RandomizedSearchCV(rfc,
                         parameters,
                         n_jobs=4,
                         n_iter=20,
                         random_state=42,
                         scoring=f1_scorer)
t0 = time()
clf.fit(X, y)
t1 = time()
(t1-t0)/60
#y_pred = gbm.predict(X.as_matrix())

sig_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit' )
sig_clf.fit(X_dev, y_dev)

import pickle
with open(path.join(MODELS_DIR, 'rfc_200e_calib_enh_feat.pkl'), 'wb') as f:
    pickle.dump(sig_clf, f)
