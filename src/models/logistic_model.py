'''This model scores 2.439 on Kaggle'''

import pandas as pd
import numpy as np
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import make_scorer, log_loss
from sklearn.cross_validation import train_test_split
from os import path
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")
VISUALIZATION_DIR = os.environ.get("VISUALIZATION_DIR")
MODELS_DIR = os.environ.get("MODELS_DIR")

data = pd.read_csv(os.path.join(FEATURES_DATA_DIR,'train_dataset.csv'))
labels = data['group']
data = data.drop(['group', 'device_model', 'device_id'], 1)

targetencoder = LabelEncoder().fit(labels)
labels = targetencoder.transform(labels)
nclasses = len(targetencoder.classes_)

X, X_dev, y, y_dev = train_test_split(data, labels, test_size=0.2, random_state=0)

classfier = LogisticRegression(C=0.01)
#Cs = np.logspace(-3,0,4)
#parameters = {'C': Cs}
#logl_scorer = make_scorer(log_loss, greater_is_better=False)
# clf = GridSearchCV(clf,
#                    param_grid=parameters,
#                    scoring=logl_scorer)
clf = classfier
clf.fit(X,y)
pred = clf.predict_proba(X_dev)

res = log_loss(y_dev, pred)
import pickle
with open(path.join(MODELS_DIR, 'mine_feat_log_reg_001c_encoded_label.pkl'), 'wb') as f:
    pickle.dump(clf, f)

data = pd.read_csv(path.join(FEATURES_DATA_DIR, 'test_dataset.csv'))
data.shape
dev_id = data['device_id']
data = data.drop(['device_model', 'device_id'], 1)
pred = clf.predict_proba(data)

pred = pd.DataFrame(pred)
pred.columns = targetencoder.inverse_transform(pred.columns)
pred['device_id'] = dev_id
pred.to_csv(path.join(FEATURES_DATA_DIR, 'submission_mine_features.csv'), index=False)
