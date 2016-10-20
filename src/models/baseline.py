'''SCORE ON KAGGLE FOR THIS MODEL IS 2.43266'''

import os
import numpy as np
import pandas as pd
from os import path
from dotenv import load_dotenv, find_dotenv

%matplotlib inline

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

MODELS_DIR = os.environ.get("MODELS_DIR")
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

train_data = pd.read_csv(path.join(RAW_DATA_DIR, 'gender_age_train.csv'))
test_data = pd.read_csv(path.join(RAW_DATA_DIR, 'gender_age_test.csv'))

labels = train_data['group']

f = {}
for l in labels:
    if l in f:
        f[l] += 1
    else:
        f[l] = 1
for e in f:
    f[e] = float(f[e])/len(labels)

row = pd.DataFrame(f.values()).T
baseline = pd.concat([row]*test_data.shape[0])
baseline.columns = f.keys()
baseline = baseline.reset_index(drop=True)

baseline['device_id'] = test_data['device_id']

baseline.to_csv(path.join(MODELS_DIR, 'skewed_baseline.csv'), index=False)
