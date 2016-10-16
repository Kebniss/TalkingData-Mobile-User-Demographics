import os
import numpy as np
import pandas as pd
from os import path
from scripts import fillnan
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

app_data = pd.read_csv(path.join(FEATURES_DATA_DIR, 'app_features.csv'))
app_data['device_id'] = app_data['device_id'].astype(str)
app_data = app_data.drop('Unnamed: 0', 1)

cat_data = pd.read_csv(path.join(FEATURES_DATA_DIR, 'categories_features.csv'))
cat_data['device_id'] = cat_data['device_id'].astype(str)
cat_data = cat_data.drop('Unnamed: 0', 1)

app_features = app_data.merge(cat_data,
                              on='device_id',
                              how='inner',
                              suffixes=['_app', '_cat'])

specs = pd.read_csv(path.join(FEATURES_DATA_DIR, 'specs_features.csv'))
specs['device_id'] = specs['device_id'].astype(str)

app_specs = app_features.merge(specs, on='device_id', how='inner')
app_specs['phone_brand'], map_brand = pd.factorize(app_specs['phone_brand'])
pd.DataFrame(map_brand).to_csv(path.join(FEATURES_DATA_DIR, 'map_brand,csv'))

ignore_columns = ['phone_brand', 'device_model']
final_features = fillnan(app_specs, ignore_columns=ignore_columns)
final_features.info()
a=1
final_features.to_csv(path.join(FEATURES_DATA_DIR, 'final_features.csv'), index=False)
