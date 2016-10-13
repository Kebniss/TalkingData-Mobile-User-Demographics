
import os
import sys
from os import path
import numpy as np
import pandas as pd
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

PROJECT_DIR = os.environ.get("PROJECT_DIR")
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")
VISUALIZATION_DIR = os.environ.get("VISUALIZATION_DIR")

gatrain = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_train.csv'),
                      index_col='device_id')
gatest = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_test.csv'),
                     index_col = 'device_id')

phone = pd.read_csv(path.join(RAW_DATA_DIR,'phone_brand_device_model.csv'))
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

# PHONE BRAND

brandencoder = LabelEncoder().fit(phone['phone_brand'])
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]),
                         (gatrain['trainrow'], gatrain['brand']))
                       )
Xte_brand = csr_matrix((np.ones(gatest.shape[0]),
                        (gatest['testrow'], gatest['brand']))
                       )

m = phone['phone_brand'].str.cat(phone['device_model'])
brandencoder = LabelEncoder().fit(m)
phone['model'] = brandencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]),
                         (gatrain['trainrow'], gatrain['model']))
                       )
Xte_model = csr_matrix((np.ones(gatest.shape[0]),
                        (gatest['testrow'], gatest['model']))
                       )

phone_train = hstack((Xtr_brand, Xtr_model), format='csr')
io.mmwrite(path.join(FEATURES_DATA_DIR, 'phone_features_train'), phone_train)
phone_test = hstack((Xte_brand, Xte_model), format='csr')
io.mmwrite(path.join(FEATURES_DATA_DIR, 'phone_features_test'), phone_test)
