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

test_ids = pd.read_csv(path.join(RAW_DATA_DIR, 'gender_age_test.csv'))
test_ids['device_id'] = test_ids['device_id'].astype(str)

test_data = test_ids.merge(features, on='device_id', how='left')
test_data.shape
test_data = test_data.fillna(-1)

test_data.to_csv(path.join(FEATURES_DATA_DIR, 'test_dataset.csv'), index=False)
