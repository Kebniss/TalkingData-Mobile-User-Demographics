""" This script loads the raw phone_brand_device_model phoneset, creates the
    features and deals with NaN values."""

import os
from os import path
import pandas as pd
import pickle as pkl
from scipy import sparse, io
from scipy.sparse import csr_matrix, hstack
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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

phone = pd.read_csv(path.join(RAW_DATA_DIR, 'phone_brand_device_model.csv'))
phone = phone.drop_duplicates('device_id', keep='first')

specs_table = pd.read_csv(path.join(FEATURES_DATA_DIR, 'specs_table.csv'))
model_mapping = pd.read_csv(path.join(FEATURES_DATA_DIR, 'model_mapping.csv'))
brand_mapping = pd.read_csv(path.join(FEATURES_DATA_DIR, 'brand_mapping.csv'))

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

phone = (phone.set_index('device_id').join(gatrain[['trainrow']], how='left')
                                     .join(gatest[['testrow']], how='left'))

price_scale = StandardScaler().fit(phone['price_eur'].reshape(-1,1))
phone['price_eur'] = price_scale.transform(phone['price_eur'].reshape(-1,1))

brandencoder = LabelEncoder().fit(phone['phone_brand'])
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']

# encoding and scaling all features to a distribution with mean = 0
phone['model'] = phone['phone_brand'].str.cat(phone['device_model'])
modelencoder = LabelEncoder().fit(phone['model'])
phone['model'] = modelencoder.transform(phone['model'])
gatrain['model'] = phone['model']
gatest['model'] = phone['model']


phone_train = phone.dropna(subset=['trainrow']).drop('testrow',1)
phone_test = phone.dropna(subset=['testrow']).drop('trainrow',1)

assert phone_train.reset_index()['device_id'].nunique() == phone_train.shape[0]
assert sorted(phone_train.reset_index()['device_id']) == sorted(gatrain.reset_index()['device_id'])

assert phone_test.reset_index()['device_id'].nunique() == phone_test.shape[0]
assert sorted(phone_test.reset_index()['device_id']) == sorted(gatest.reset_index()['device_id'])

# PHONE BRAND ---------------------------------------------------------------

nbrands = brandencoder.classes_.shape[0]
d = phone_train.reset_index()
Xtr_brand = csr_matrix( ( d['price_eur'], (d['trainrow'], d['brand']) ),
                             shape=(gatrain.shape[0], nbrands)
                          )

d = phone_test.reset_index()
Xte_brand = csr_matrix( ( d['price_eur'], (d['testrow'], d['brand']) ),
                             shape=(gatest.shape[0], nbrands)
                          )

# DEVICE MODEL --------------------------------------------------------------

nmodels = modelencoder.classes_.shape[0]
d = phone_train.reset_index()
Xtr_model = csr_matrix( ( d['price_eur'], (d['trainrow'], d['model']) ),
                             shape=(gatrain.shape[0], nmodels)
                          )

d = phone_test.reset_index()
Xte_model = csr_matrix( ( d['price_eur'], (d['testrow'], d['model']) ),
                             shape=(gatest.shape[0], nmodels)
                          )

# MERGE FEATURES ------------------------------------------------------------

train = hstack( (Xtr_brand, Xtr_model), format='csr')
test = hstack( (Xte_brand, Xte_model), format='csr')

#save
io.mmwrite(path.join(FEATURES_DATA_DIR, 'sparse_brand_model_price_train'), train)
io.mmwrite(path.join(FEATURES_DATA_DIR, 'sparse_brand_model_price_test'), test)
