""" This script loads the raw events dataset, creates the
    features and deals with NaN values."""

import os
import os.path
import numpy as np
import pandas as pd
from scipy import sparse, io
from scipy.sparse import csr_matrix, hstack
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

lat_long_counts = data.groupby(['device_id', 'lat', 'lon'])['event_id'].agg(['count'])
positions = lat_long_counts.reset_index(['lat', 'lon']).groupby(level=0).max()
gatrain = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_train.csv'))
gatest = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_test.csv'))

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

tex_lat_long_counts = (data
                       .groupby(['device_id', 'lat', 'lon'])['event_id']
                       .agg(['count'])
                       )
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
# keep = last avoids that when for a d_id there are two or more euqally frequent
# positions the tuple lo0,la0 is picked if it is the first value
data_max = (data_max
            .reset_index()
            .drop_duplicates(subset='device_id', keep='last') # avoids (lo0,la0)
            .drop('count',1)
            )

data_max = (data_max.merge(gatrain[['device_id','trainrow']], on='device_id', how='left')
                    .merge(gatest[['device_id','testrow']], on='device_id', how='left')
                    )

# generate sparse data for logistic classifier
lat_encoder = LabelEncoder().fit(data_max['lat'])
data_max['lat'] = lat_encoder.transform(data_max['lat'])
nlats = len(lat_encoder.classes_)

lon_encoder = LabelEncoder().fit(data_max['lon'])
data_max['lon'] = lon_encoder.transform(data_max['lon'])
nlons = len(lon_encoder.classes_)

d_lat = data_max.dropna(subset=['trainrow'])
Xtr_lat = csr_matrix((np.ones(d_lat.shape[0]), (d_lat.trainrow, d_lat.lat)),
                      shape=(gatrain.shape[0],nlats))
d_lat = data_max.dropna(subset=['testrow'])
Xte_lat = csr_matrix((np.ones(d_lat.shape[0]), (d_lat.testrow, d_lat.lat)),
                      shape=(gatest.shape[0],nlats))

d_lon = data_max.dropna(subset=['trainrow'])
Xtr_lon = csr_matrix((np.ones(d_lon.shape[0]), (d_lon.trainrow, d_lon.lon)),
                      shape=(gatrain.shape[0],nlons))
d_lon = data_max.dropna(subset=['testrow'])
Xte_lon = csr_matrix((np.ones(d_lon.shape[0]), (d_lon.testrow, d_lon.lat)),
                      shape=(gatest.shape[0],nlons))

Xtrain = hstack((Xtr_lat, Xtr_lon), format='csr')
Xtest =  hstack((Xte_lat, Xte_lon), format='csr')

#save sparse dataset
io.mmwrite(path.join(FEATURES_DATA_DIR, 'sparse_position_train'), Xtrain)
io.mmwrite(path.join(FEATURES_DATA_DIR, 'sparse_position_test'), Xtest)
