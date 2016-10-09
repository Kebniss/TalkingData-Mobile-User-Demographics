import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from dotenv import load_dotenv, find_dotenv
from os import path
%matplotlib inline

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")
VISUALIZATION_DIR = os.environ.get("VISUALIZATION_DIR")

train_data = pd.read_csv(path.join(FEATURES_DATA_DIR, 'train_dataset.csv'))
labels = train_data['group']
data = train_data.drop('group', 1)
data['phone_brand'], map_brand = pd.factorize(data['phone_brand'])
Xtrain = data

targetencoder = LabelEncoder().fit(labels)
y = targetencoder.transform(labels)
nclasses = len(targetencoder.classes_)

def score(clf, random_state=0):
    kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
    pred = np.zeros((y.shape[0], nclasses))
    for itrain, itest in kf:
        Xtr, Xte = Xtrain.iloc[itrain, :], Xtrain.iloc[itest, :]
        ytr, yte = y[itrain], y[itest]
        clf.fit(Xtr, ytr)
        pred[itest, :] = clf.predict_proba(Xte)
        return log_loss(yte, pred[itest, :])
    print "{:.5f}".format(log_loss(yte, pred[itest, :]))
    print('')
    return log_loss(y, pred)

Cs = np.logspace(-3, 0, 4)

res = []
for C in Cs:
    res.append(score(LogisticRegression(C=C)))
plt.semilogx(Cs, res, '-o')

score(LogisticRegression(C=0.01))

score(LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs'))
