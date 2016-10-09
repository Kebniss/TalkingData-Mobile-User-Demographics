import os
import numpy as np
import pandas as pd
from time import time
from os import path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from dotenv import load_dotenv, find_dotenv
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")


%matplotlib inline

data = pd.read_csv(path.join(FEATURES_DATA_DIR, 'train_dataset.csv'))
labels = data['group']
data = data.drop('group', 1)

# ANALYZE CLASSES DISTRIBUTION
# ignore id and string columns because their values don't come from a distribution
string_cols = [col for col in list(data.columns)
               if any([s in col for s in ['_id', '_most_used', '_app_dly']])]
string_cols.extend(['device_model', 'phone_brand'])
process_cols = [col for col in data.columns if col not in string_cols]
#data[process_cols].describe()

def class_frequency(labels):
    f = {}
    for l in labels:
        if l in f:
            f[l] += 1
        else:
            f[l] = 1
    return f
f = class_frequency(labels)

bin_labels = []
males = 0
females = 0
for l in labels:
    if 'M' in l:
        bin_labels.extend([0])
        males +=1
    else:
        bin_labels.extend([1])
        females +=1

float(males)/len(labels)
float(females)/len(labels)

data['phone_brand'], map_id = pd.factorize(data['phone_brand'])
data = data.drop('device_id',1)
data = data.drop('phone_brand',1)
X, X_dev, y, y_dev = train_test_split(data,
                                      labels,
                                      test_size=0.30,
                                      random_state=0)
X_1, X_2, y_1, y_2 = train_test_split(X_dev,
                                      y_dev,
                                      test_size=0.50,
                                      random_state=0)
train_dist = [0, 0]
for e in y:
    if e == 0:
        train_dist[0] +=1
    else:
        train_dist[1] += 1
train_freq = [float(e)/len(y) for e in train_dist]

dev_dist = [0, 0]
for e in y_dev:
    if e == 0:
        dev_dist[0] +=1
    else:
        dev_dist[1] += 1
dev_freq = [float(e)/len(y_dev) for e in dev_dist]

classifier = RandomForestClassifier(n_estimators=200, n_jobs=4)
parameters = {'max_depth': (5, 6, 7, 8, 9, 11),
              'min_samples_split': (10, 30, 50, 70, 100, 200)}
f1_scorer = make_scorer(f1_score, greater_is_better=True, average='weighted')
clf = RandomizedSearchCV(classifier,
                         parameters,
                         n_jobs=4,
                         n_iter=20,
                         random_state=42,
                         scoring=f1_scorer)

import sys
reload(sys)
sys.setdefaultencoding('utf8')
t0 = time()
clf.fit(X, y)
t1 = time()
t1-t0

y_pred = clf.predict(X_1)

f1_score(y_1,y_pred, average='weighted')

cm = confusion_matrix(y_1, y_pred)

pred = ['male_pred', 'female_pred' ]
true = ['male_true', 'female_true' ]

with sns.axes_style('white'):
    plt.figure()
    ax = sns.heatmap(cm, xticklabels=pred, yticklabels=true, square=True, annot=True, cmap='RdBu_r')
    #plt.show()


sig_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit' )
sig_clf.fit(X_1, y_1)
sig_cdlf_probs = sig_clf.predict_proba(X_2)
sig_score = log_loss(y_2, sig_cdlf_probs)

# FEATURE IMPPORTANCE

importances = clf.best_estimator_.feature_importances_
features = data.keys()
features_importance = pd.DataFrame(importances,
                                   index=features,
                                   columns=['Importance']).sort(columns='Importance', ascending=False)

features_importance


from sklearn.manifold import TSNE

data_tsne = TSNE(n_components=2).fit_transform(data)

x_min, x_max = data_tsne[:,0].min() - 1, data_tsne[:,1].max() + 1
y_min, y_max = data_tsne[:,0].min() - 1, data_tsne[:,1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(x_min, x_max, 0.1))

plt.figure()
plt.scatter(data_tsne[:,0], data_tsne[:,1], c=bin_labels, alpha=0.8)
plt.show()
