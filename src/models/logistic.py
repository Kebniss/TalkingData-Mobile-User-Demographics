''' BASIC MODEL SCORES 2.36979 ON KAGGLE'''
''' NEWTON MODEL SCORES 2.36923 ON KAGGLE'''

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
res3 = []
res4 = []
res5 = []
res6 = []
for C in Cs:
    res1.append(score(LogisticRegression(C = C, n_jobs=4), data, y, nclasses))
    res2.append(score(LogisticRegression(C = C, multi_class='multinomial',solver='lbfgs', n_jobs=4)
                      , data, y, nclasses))
    res3.append(score(LogisticRegression(C = C, class_weight='balanced', n_jobs=4), data, y, nclasses))
    res4.append(score(LogisticRegression(C = C, multi_class='multinomial',solver='lbfgs', class_weight='balanced', n_jobs=4)
                      , data, y, nclasses))
    res5.append(score(LogisticRegression(C = C, multi_class='multinomial', solver='newton-cg', n_jobs=4)
                      , data, y, nclasses))
    res6.append(score(LogisticRegression(C = C, multi_class='multinomial', solver='newton-cg', class_weight='balanced', n_jobs=4)
                          , data, y, nclasses))
plt.figure(figsize=(12,6))
plt.semilogx(Cs, res1,'-o', label='basic')
plt.semilogx(Cs, res2,'-o', label='multinomial lbfgs')
plt.semilogx(Cs, res3,'-o', label='balanced')
plt.semilogx(Cs, res4,'-o', label='multinomial  + balanced')
plt.semilogx(Cs, res5,'-o', label='multinomial newton-cg')
plt.semilogx(Cs, res6,'-o', label='multinomial newton-cg + balanced')
plt.legend(loc=2)
plt.savefig('log-loss errors', format='png')
clf = LogisticRegression(C=0.03, multi_class='multinomial', solver='newton-cg', n_jobs=4)
clf.fit(data, y)

with open(path.join(MODELS_DIR, 'logistic_003c_newton_specs_feat.pkl'), 'wb') as f:
    pickle.dump(clf, f)
