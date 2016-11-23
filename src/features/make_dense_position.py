""" This script loads the raw events dataset and creates the dense features"""

import os
import os.path
import numpy as np
import pandas as pd
from scipy import sparse, io
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

data = pd.read_csv(os.path.join(RAW_DATA_DIR, 'events.csv'),
                   parse_dates=['timestamp'],
                   infer_datetime_format=True)
data.columns = ['event_id', 'device_id', 'timestamp', 'lon', 'lat']

gatrain = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_train.csv'))
gatest = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_test.csv'))

# add rownumber = encoding of device_id
gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

# count the number of events per position per user
tex_lat_long_counts = (data
                       .groupby(['device_id', 'lat', 'lon'])['event_id']
                       .agg(['count'])
                       )
# select the pair lat, lon visited the most
tex_lat_long_counts = (tex_lat_long_counts.join(tex_lat_long_counts
                                                .groupby(level=0)
                                                .max()
                                                , rsuffix="_max")
                       )
tex_lat_long_counts = (tex_lat_long_counts
                       .query("count == count_max")
                       .drop("count_max", axis=1)
                       )

data = data.set_index(['device_id', 'lat', 'lon'])
data_max = tex_lat_long_counts.join(data, how='left')
# some d_id have 2+ positions with the same count last picks the one with lo and la the biggest
data_max = (data_max
            .reset_index()
            .drop_duplicates(subset='device_id', keep='last') # last avoids (lo0,la0)
            .drop('count',1)
            )

# preprocessing: scaling lat and lon
lat_scale,lon_scale = StandardScaler(), StandardScaler()
data_max['lat'] = lat_scale.fit_transform(data_max['lat'].reshape(-1, 1), [0,1])
data_max['lon'] = lon_scale.fit_transform(data_max['lon'].reshape(-1, 1), [0,1])

data_max = (data_max.merge(gatrain[['device_id','trainrow']], on='device_id', how='left')
                    .merge(gatest[['device_id','testrow']], on='device_id', how='left')
                    )

data_max = data_max.drop(['event_id', 'timestamp'],1)

# generate dense data for random forest
train_data = data_max.dropna(subset=['trainrow']).drop(['testrow'],1)
test_data = data_max.dropna(subset=['testrow']).drop(['trainrow'],1)

#save dense dataset
train_data.to_csv(os.path.join(FEATURES_DATA_DIR, 'dense_position_train.csv'), index=False)
test_data.to_csv(os.path.join(FEATURES_DATA_DIR, 'dense_position_test.csv'), index=False)
