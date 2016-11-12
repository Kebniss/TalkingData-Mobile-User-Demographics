import os
import sys
import numpy as np
import pandas as pd
from os import path
from scripts import *
from scipy import sparse, io
from datetime import timedelta
from drop_nans import drop_nans
from operations_on_list import *
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, hstack
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import LabelEncoder
from get_most_recent_event import get_most_recent_event
from rolling_stats_in_window import rolling_stats_in_window
from rolling_most_freq_in_window import rolling_most_freq_in_window

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

gatrain = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_train.csv'),
                      index_col='device_id')
gatest = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_test.csv'),
                     index_col = 'device_id')

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

phone = pd.read_csv(path.join(RAW_DATA_DIR,'phone_brand_device_model.csv'))
phone = phone.drop_duplicates('device_id',keep='first')

specs_table = pd.read_csv(path.join(FEATURES_DATA_DIR, 'specs_table.csv'))
model_mapping = pd.read_csv(path.join(FEATURES_DATA_DIR, 'model_mapping.csv'))
brand_mapping = pd.read_csv(path.join(FEATURES_DATA_DIR, 'brand_mapping.csv'))

phone = phone.merge(brand_mapping, how='left', left_on='phone_brand',
                                      right_on='phone_brand_chinese')
phone = phone.merge(model_mapping, how='left', left_on='device_model',
                                      right_on='device_model_chinese')
phone = phone.merge(specs_table,
                    left_on=['phone_brand_latin', 'device_model_latin'],
                    right_on=['phone_brand', 'device_model'],
                    how='left',
                    suffixes=['', '_R'])
phone = phone.drop(['phone_brand_latin', 'device_model_latin',
                    'phone_brand_chinese', 'device_model_chinese',
                    'phone_brand_R', 'device_model_R'], axis=1)
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')


# PHONE BRAND

brandencoder = LabelEncoder().fit(phone['phone_brand'])
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']

m = phone['phone_brand'].str.cat(phone['device_model'])
brandencoder = LabelEncoder().fit(m)
phone['model'] = brandencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']

phone = phone.drop(['phone_brand', 'device_model'], 1)
phone = phone.fillna(-1)
phone = (phone.join(gatrain[['trainrow']], how='left')
              .join(gatest[['testrow']], how='left'))

phone_train = phone.dropna(subset=['trainrow']).drop('testrow',1)
phone_test = phone.dropna(subset=['testrow']).drop('trainrow',1)

assert phone_train.reset_index()['device_id'].nunique() == phone_train.shape[0]
assert sorted(phone_train.reset_index()['device_id']) == sorted(gatrain.reset_index()['device_id'])

assert phone_test.reset_index()['device_id'].nunique() == phone_test.shape[0]
assert sorted(phone_test.reset_index()['device_id']) == sorted(gatest.reset_index()['device_id'])

phone_train = phone_train.sort_values(by='trainrow').drop('trainrow', 1)
phone_test = phone_test.sort_values(by='testrow').drop('testrow', 1)

phone_train_sparse = csr_matrix(phone_train.values)
phone_test_sparse = csr_matrix(phone_test.values)

io.mmwrite(path.join(FEATURES_DATA_DIR, 'specs_phone_features_train'), phone_train)
io.mmwrite(path.join(FEATURES_DATA_DIR, 'specs_phone_features_test'), phone_test)
