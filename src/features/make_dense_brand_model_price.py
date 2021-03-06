""" This script loads the raw phone_brand_device_model phone set, creates the
    features and deals with NaN values."""

import os
from os import path
import pandas as pd
import pickle as pkl
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
gatest.head()
gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

phone = phone.drop_duplicates('device_id')

# join phone to add the phone price feature
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

# not all models have the price stored, fill the NaN values
phone['price_eur'] = phone['price_eur'].fillna(-1)

phone = (phone.set_index('device_id').join(gatrain[['trainrow']], how='left')
                                     .join(gatest[['testrow']], how='left'))

# encoding and scaling all features to a distribution with mean = 0
phone['device_model'] = phone['phone_brand'] + phone['device_model']

# ecoding strings to numbers
brandencoder = LabelEncoder().fit(phone['phone_brand'])
modelencoder = LabelEncoder().fit(phone['device_model'])
phone['phone_brand'] = brandencoder.transform(phone['phone_brand'])
phone['device_model'] = modelencoder.transform(phone['device_model'])

# scale data to a distribution with 0 mean and variance 1
brand_scale = StandardScaler().fit(phone['phone_brand'].reshape(-1,1))
model_scale = StandardScaler().fit(phone['device_model'].reshape(-1,1))
price_scale = StandardScaler().fit(phone['price_eur'].reshape(-1,1))

phone['phone_brand'] = brand_scale.transform(phone['phone_brand'].reshape(-1,1))
phone['device_model'] = model_scale.transform(phone['device_model'].reshape(-1,1))
phone['price_eur'] = price_scale.transform(phone['price_eur'].reshape(-1,1))

# device_ids that belongs to gatrain's rows make the training set. Test set is viceversa
phone_train = phone.dropna(subset=['trainrow']).drop(['testrow'],1)
phone_test = phone.dropna(subset=['testrow']).drop(['trainrow'],1)

#save
phone_train.to_csv(path.join(FEATURES_DATA_DIR, 'dense_brand_model_price_train.csv'))
phone_test.to_csv(path.join(FEATURES_DATA_DIR, 'dense_brand_model_price_test.csv'))
