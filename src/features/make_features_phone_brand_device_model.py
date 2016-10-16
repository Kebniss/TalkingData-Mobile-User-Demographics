""" This script loads the raw phone_brand_device_model dataset, creates the
    features and deals with NaN values."""

import os.path
import pandas as pd
import pickle as pkl
from scripts import *

os.getcwd()
os.chdir('..\..')
path = os.getcwd() + '\data\\raw\phone_brand_device_model.csv'
data = pd.read_csv(path)
path = os.getcwd() + '\data\\processed\specs_table.csv'
specs_table = pd.read_csv(path)
path = os.getcwd() + '\data\\processed\model_mapping.csv'
model_mapping = pd.read_csv(path)
path = os.getcwd() + '\data\\processed\\brand_mapping.csv'
brand_mapping = pd.read_csv(path)

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

data = data.merge(specs_table,
                 on=['phone_brand', 'device_model'],
                 how='left',
                 suffixes=['', '_R'])

for c in data[['price_eur', 'screen_size', 'ram_gb', 'release_month', 'release_year', 'camera']].columns:
    data[c] = fillnan(data[c])

#save
path = os.getcwd() + '\data\\processed\\specs_features.csv'
data.to_csv(path, index=False)
