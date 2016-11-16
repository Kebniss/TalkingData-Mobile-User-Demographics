import os
import numpy as np
import pandas as pd
from os import path
from scipy import sparse, io
from scipy.sparse import csr_matrix, hstack
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

# MAKE SPARSE FEATURES -------------------------------------------------------
phone_s = io.mmread(path.join(FEATURES_DATA_DIR, 'sparse_brand_model_price_train'))
app_labels_s = io.mmread(path.join(FEATURES_DATA_DIR, 'sparse_cum_app_labels_train'))
distance_s = io.mmread(path.join(FEATURES_DATA_DIR, 'sparse_position_train'))

train = hstack((phone_s, app_labels_s, distance_s), format='csr')

io.mmwrite(path.join(FEATURES_DATA_DIR, 'sparse_train_p_al_d'), train)


# MAKE DENSE FEATURES --------------------------------------------------------
phone_d = pd.read_csv(path.join(FEATURES_DATA_DIR, 'dense_brand_model_price_train.csv')
                      , index_col='trainrow').drop(['device_id'], 1)
app_labels_d = pd.read_csv(path.join(FEATURES_DATA_DIR, 'dense_500SVD_cum_app_labels_train.csv'))
distance_d = pd.read_csv(path.join(FEATURES_DATA_DIR, 'dense_position_train.csv')
                         , index_col='trainrow').drop(['device_id'], 1)

train = (phone_d.join(app_labels_d, how='outer')
                .join(distance_d, how='outer'))

# fill nan with average value of feature = less information value
for col in train.columns:
    train[col] = train[col].fillna(train[col].mean(0))

train.to_csv(path.join(FEATURES_DATA_DIR, 'dense_train_p_al_d.csv'), index=False)
