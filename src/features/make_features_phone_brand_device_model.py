""" This script loads the raw phone_brand_device_model dataset, creates the
    features and deals with NaN values."""

import os
from os import path
import pandas as pd
import pickle as pkl
from scripts import *
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

data = pd.read_csv(path.join(RAW_DATA_DIR, 'phone_brand_device_model.csv'))
specs_table = pd.read_csv(path.join(FEATURES_DATA_DIR, 'specs_table.csv'))
model_mapping = pd.read_csv(path.join(FEATURES_DATA_DIR, 'model_mapping.csv'))
brand_mapping = pd.read_csv(path.join(FEATURES_DATA_DIR, 'brand_mapping.csv'))

data = data.drop_duplicates('device_id')

data = data.merge(brand_mapping, how='left', left_on='phone_brand',
                                      right_on='phone_brand_chinese')
data = data.merge(model_mapping, how='left', left_on='device_model',
                                      right_on='device_model_chinese')
data = data.drop(['phone_brand', 'device_model',
           'phone_brand_chinese', 'device_model_chinese'], axis=1)
data = data.drop_duplicates('device_id')
data = data.rename( columns = {'phone_brand_latin': 'phone_brand',
                               'device_model_latin': 'device_model'})

data = data.merge(specs_table[['phone_brand', 'device_model', 'price_eur']],
                 on=['phone_brand', 'device_model'],
                 how='left',
                 suffixes=['', '_R'])

data['price_eur'] = data['price_eur'].fillna(-1)

#save
data.to_csv(path.join(RAW_DATA_DIR, 'phone_brand_model_price.csv'), index=False)
