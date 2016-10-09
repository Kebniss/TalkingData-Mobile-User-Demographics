import os
import numpy as np
import pandas as pd
from os import path
from scripts import fillnan
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

features = pd.read_csv(path.join(FEATURES_DATA_DIR, 'final_features.csv'))
features['device_id'] = features['device_id'].astype(str)

train_ids = pd.read_csv(path.join(RAW_DATA_DIR, 'gender_age_train.csv'))
train_ids['device_id'] = train_ids['device_id'].astype(str)

train_data = train_ids.merge(features, on='device_id', how='inner')
train_data.shape

train_data = train_data.drop(['gender', 'age'], 1)
train_data.to_csv(path.join(FEATURES_DATA_DIR, 'train_dataset.csv'), index=False)
