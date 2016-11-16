import os
import numpy as np
import pandas as pd
from os import path
from scipy import sparse, io
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

# MAKE SPARSE FEATURES -------------------------------------------------------
phone_s = io.mmread(path.join(FEATURES_DATA_DIR, 'sparse_brand_model_price_test'))
app_labels_s = io.mmread(path.join(FEATURES_DATA_DIR, 'sparse_cum_app_labels_test'))
distance_s = io.mmread(path.join(FEATURES_DATA_DIR, 'sparse_position_test'))

test_set = hstack((phone_s, app_labels_s, distance_s), format='csr')

io.mmwrite(path.join(FEATURES_DATA_DIR, 'sparse_test_p_al_d'), test_set)


# MAKE DENSE FEATURES --------------------------------------------------------
phone_d = pd.read_csv(path.join(FEATURES_DATA_DIR, 'dense_brand_model_price_test.csv')
                      , index_col='testrow').drop(['device_id'], 1)
app_labels_d = pd.read_csv(path.join(FEATURES_DATA_DIR, 'dense_500SVD_cum_app_labels_test.csv'))
distance_d = pd.read_csv(path.join(FEATURES_DATA_DIR, 'dense_position_test.csv')
                         , index_col='testrow').drop(['device_id'], 1)

test = (phone_d.join(app_labels_d, how='outer')
               .join(distance_d, how='outer'))

for col in test.columns:
    test[col] = test[col].fillna(test[col].mean(0))

test.to_csv(path.join(FEATURES_DATA_DIR, 'dense_test_p_al_d.csv'), index=False)
