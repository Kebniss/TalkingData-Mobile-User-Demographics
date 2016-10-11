'''SCORE ON KAGGLE FOR THIS MODEL IS 2.43266'''

import os
import numpy as np
import pandas as pd
from os import path
from dotenv import load_dotenv, find_dotenv

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
data = data.drop(['device_model', 'device_id'], 1)


f = {}
for l in labels:
    if l in f:
        f[l] += 1
    else:
        f[l] = 1
for e in f:
    f[e] = float(f[e])/len(labels)

row = pd.DataFrame(f.values()).T
baseline = pd.concat([b]*112071)
baseline.columns = f.keys()
baseline = baseline.reset_index(drop=True)

data_t = pd.read_csv(path.join(FEATURES_DATA_DIR, 'test_dataset.csv'))
baseline['device_id'] = data_t['device_id']

baseline.to_csv(path.join(FEATURES_DATA_DIR, 'skewed_baseline.csv'), index=False)
