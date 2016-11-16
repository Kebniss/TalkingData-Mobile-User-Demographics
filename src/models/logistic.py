''' THIS MODEL SCORES 2.36923 ON KAGGLE'''

import os
import pickle
import numpy as np
import pandas as pd
from os import path
import seaborn as sns
from scipy import sparse, io
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix, hstack
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
%matplotlib inline

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")

data = io.mmread(path.join(FEATURES_DATA_DIR, 'sparse_train_p_al_d')).tocsr()
gatrain = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_train.csv'),
                      index_col='device_id')
labels = gatrain['group']
targetencoder = LabelEncoder().fit(labels)
y = targetencoder.transform(labels)
nclasses = len(targetencoder.classes_)

with open(path.join(FEATURES_DATA_DIR, 'targetencoder_logistic.pkl'), 'wb') as f:
    pickle.dump(targetencoder, f)

def score(clf, X, y, nclasses, random_state=None):
    kf=StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
    pred = np.zeros((y.shape[0], nclasses))
    for itrain, itest in kf:
        Xtr, Xte = X[itrain, :], X[itest, :]
        ytr, yte = y[itrain], y[itest]
        clf.fit(Xtr, ytr)
        pred[itest, :] = clf.predict_proba(Xte)
        return log_loss(yte, pred[itest])
    return log_loss(y, pred)

Cs = np.logspace(-5,0,7)
res1 = []
res2 = []
for C in Cs:
    res1.append(score(LogisticRegression(C = C), data, y, nclasses))
    res2.append(score(LogisticRegression(C = C, multi_class='multinomial',solver='lbfgs')
                      , data, y, nclasses))
plt.semilogx(Cs, res1,'-o')
plt.semilogx(Cs, res2,'-o')

clf = LogisticRegression(C=0.03)
clf.fit(data, y)

with open(path.join(MODELS_DIR, 'logistic_003c_specs_feat.pkl'), 'wb') as f:
    pickle.dump(clf, f)
