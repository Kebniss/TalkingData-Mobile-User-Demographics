""" This script loads the raw phone_brand_device_model phoneset, creates the
    features and deals with NaN values."""

import os
from os import path
import pandas as pd
import pickle as pkl
from scripts import *
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

phone = pd.read_csv(path.join(RAW_DATA_DIR, 'phone_brand_device_model.csv'))
specs_table = pd.read_csv(path.join(FEATURES_DATA_DIR, 'specs_table.csv'))
model_mapping = pd.read_csv(path.join(FEATURES_DATA_DIR, 'model_mapping.csv'))
brand_mapping = pd.read_csv(path.join(FEATURES_DATA_DIR, 'brand_mapping.csv'))
gatrain = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_train.csv'),
                      index_col='device_id')
gatest = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_test.csv'),
                     index_col = 'device_id')

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

phone = phone.drop_duplicates('device_id')

phone = phone.merge(brand_mapping, how='left', left_on='phone_brand',
                                      right_on='phone_brand_chinese')
phone = phone.merge(model_mapping, how='left', left_on='device_model',
                                      right_on='device_model_chinese')
phone = phone.drop(['phone_brand', 'device_model',
           'phone_brand_chinese', 'device_model_chinese'], axis=1)
phone = phone.drop_duplicates('device_id')
phone = phone.rename( columns = {'phone_brand_latin': 'phone_brand',
                               'device_model_latin': 'device_model'})

phone = phone.merge(specs_table[['phone_brand', 'device_model', 'price_eur']],
                 on=['phone_brand', 'device_model'],
                 how='left',
                 suffixes=['', '_R'])

phone['price_eur'] = phone['price_eur'].fillna(-1)

# encoding and scaling all features to a distribution with mean = 0
phone['device_model'] = phone['phone_brand'] + phone['device_model']

brandencoder = LabelEncoder().fit(phone['phone_brand'])
modelencoder = LabelEncoder().fit(phone['device_model'])
phone['phone_brand'] = brandencoder.transform(phone['phone_brand'])
phone['device_model'] = modelencoder.transform(phone['device_model'])

brand_scale = StandardScaler().fit(phone['phone_brand'].reshape(-1,1))
model_scale = StandardScaler().fit(phone['device_model'].reshape(-1,1))
price_scale = StandardScaler().fit(phone['price_eur'].reshape(-1,1))

phone['phone_brand'] = brand_scale.transform(phone['phone_brand'].reshape(-1,1))
phone['device_model'] = model_scale.transform(phone['device_model'].reshape(-1,1))
phone['price_eur'] = price_scale.transform(phone['price_eur'].reshape(-1,1))

#save
phone.to_csv(path.join(RAW_DATA_DIR, 'dense_phone_brand_model_price.csv'), index=False)
