""" This script loads the raw app_events dataset, creates the
    features and deals with NaN values."""

import os
import sys
from os import path
import numpy as np
import pandas as pd
from scripts import *
from datetime import timedelta
from drop_nans import drop_nans
from operations_on_list import *
from matplotlib import pyplot as plt
from dotenv import load_dotenv, find_dotenv
from get_most_recent_event import get_most_recent_event
from rolling_stats_in_window import rolling_stats_in_window
from rolling_most_freq_in_window import rolling_most_freq_in_window

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")
VISUALIZATION_DIR = os.environ.get("VISUALIZATION_DIR")

time_data = pd.read_csv(path.join(RAW_DATA_DIR, 'events.csv'),
                        parse_dates=['timestamp'],
                        infer_datetime_format=True)

app_data = pd.read_csv(path.join(RAW_DATA_DIR, 'app_events.csv'))

# find nans
time_data = drop_nans(time_data)
app_data = drop_nans(app_data)

app_data = app_data.sort_values(by='app_id')
app_data['app_id'], map_app_id = pd.factorize(app_data['app_id'])

time_data = time_data.drop(['longitude','latitude'], 1)

# for each event_id group installed and active app_ids
installed_apps = (app_data
                  .query('is_installed == 1')
                  .groupby('event_id', as_index=False)['app_id']
                  .agg({ 'installed_apps':( lambda x: list(x) ) })
                  )

active_apps = (app_data
               .query('is_active == 1')
               .groupby('event_id', as_index=False)['app_id']
               .agg({ 'active_apps':( lambda x: list(x) ) })
               )


time_data = time_data.sort_values(by='device_id')
time_data['device_id'], map_device_id = pd.factorize(time_data['device_id'])

cols = set(app_data['app_id'])
rows = set(time_data['device_id'])

installed_matrix = np.zeros((len(rows), len(cols)))
active_matrix = np.zeros((len(rows), len(cols)))

# resample data as one per day.
data_installed = time_data.merge(installed_apps, on='event_id', how='inner')
data_installed = data_installed.drop('event_id', 1)
daily_installed = (data_installed
                   .set_index(['device_id','timestamp'])
                   .sort_index()
                   .reset_index('device_id')
                   .groupby('device_id')
                   .installed_apps
                   .resample('1D')
                   .sum()
                   .to_frame() # convert series to dataframe
                   )

daily_installed = daily_installed.reset_index('device_id')

for i in range(daily_installed.shape[0]):
    device = daily_installed.iloc[i]['device_id']
    app_list = daily_installed.iloc[i]['installed_apps']
    if isinstance(app_list, (int, long)):
        pass
    else:
        app_list = set(app_list)
        for elem in app_list:
            installed_matrix[device, elem] = 1

installed_matrix = pd.DataFrame(installed_matrix)

daily_installed['installed_apps'] = deduplicate_list(
                                    daily_installed['installed_apps'])
daily_installed['n_app_installed_daily'] = count_list_and_int(daily_installed['installed_apps'],
                                                              max_value=200)

data_active = time_data.merge(active_apps, on='event_id', how='inner')
data_active = data_active.drop('event_id', 1)
daily_active = (data_active
                   .set_index(['device_id','timestamp'])
                   .sort_index()
                   .reset_index('device_id')
                   .groupby('device_id')
                   .active_apps
                   .resample('1D')
                   .sum()
                   .to_frame() # convert series to dataframe
                )

daily_active = daily_active.reset_index('device_id')

for i in range(daily_active.shape[0]):
    device = daily_active.iloc[i]['device_id']
    app_list = daily_active.iloc[i]['active_apps']
    if isinstance(app_list, (int, long)):
        pass
    else:
        for elem in app_list:
            active_matrix[device, elem] += 1

active_matrix = pd.DataFrame(active_matrix)
active_matrix.to_csv('active_matrix.csv')

daily_active['n_app_active_daily'] = count_list_and_int(daily_active['active_apps'],
                                                        max_value=500)

daily_installed = daily_installed.drop('installed_apps', 1)
daily_active = daily_active.drop('active_apps', 1)
daily_active_most_rec = get_most_recent_event(daily_active,
                                              groupby_key='device_id',
                                              timestamp='timestamp')
daily_active_most_rec = daily_active_most_rec.reset_index('device_id')

active_matrix = active_matrix.reset_index()
active_matrix = active_matrix.rename(columns = {'index': 'device_id'})

active_apps_feat = daily_active_most_rec.merge(active_matrix,
                                               on='device_id',
                                               how='inner')

daily_installed_most_rec = get_most_recent_event(daily_installed,
                                                 groupby_key='device_id',
                                                 timestamp='timestamp')
daily_installed_most_rec = daily_installed_most_rec.reset_index('device_id')

installed_matrix = installed_matrix.reset_index()
installed_matrix = installed_matrix.rename(columns = {'index': 'device_id'})

installed_apps_feat = daily_installed_most_rec.merge(installed_matrix,
                                                     on='device_id',
                                                     how='inner')

active_apps_feat.to_csv('active_apps_feat.csv', index=False)
installed_apps_feat.to_csv('installed_apps_feat.csv', index=False)

import pickle
with open('map_device_id', 'w') as f:
    pickle.dump(map_device_id, f)

installed_apps_feat = pd.read_csv(path.join(FEATURES_DATA_DIR, 'installed_apps_feat.csv'))
active_apps_feat = pd.read_csv(path.join(FEATURES_DATA_DIR, 'active_apps_feat.csv'))
active_apps_feat = active_apps_feat.drop('Unnamed: 0',1)

from scipy import sparse
from sklearn.decomposition import RandomizedPCA

installed_apps_feat

pca_installed = RandomizedPCA(n_components=2000)
installed_not_pca = installed_apps_feat[['device_id', 'n_app_installed_daily']]
installed_apps_feat = installed_apps_feat.drop(['device_id', 'n_app_installed_daily'], 1)
installed_apps_feat = installed_apps_feat.as_matrix()
sparse_installed = sparse.csr_matrix(installed_apps_feat)
reduced_installed = pca_installed.fit_transform(installed_apps_feat)

reduced_installed = pd.DataFrame(reduced_installed)
reduced_installed = installed_not_pca.join(reduced_installed)
reduced_installed.to_csv(path.join(FEATURES_DATA_DIR, 'pca_2000_reduced_installed.csv'), index=False)
reduced_installed.shape

pca_active = RandomizedPCA(n_components=2000)
active_not_pca = active_apps_feat[['device_id', 'n_app_active_daily']]
active_apps_feat = active_apps_feat.drop(['device_id', 'n_app_active_daily'], 1)
active_apps_feat = active_apps_feat.as_matrix()
sparse_active = sparse.csr_matrix(active_apps_feat)
reduced_active = pca_active.fit_transform(active_apps_feat)
reduced_active = pd.DataFrame(reduced_active)
reduced_active = active_not_pca.join(reduced_active)
reduced_active.to_csv(path.join(FEATURES_DATA_DIR, 'pca_2000_reduced_active.csv'), index=False)
reduced_active.shape

reduced_installed = pd.read_csv(path.join(FEATURES_DATA_DIR, 'pca_2000_reduced_installed.csv'))

app_feat = reduced_active.merge(reduced_installed,
                                  on='device_id',
                                  how='inner',
                                  suffixes=['_a','_i']
                                  )

app_feat.columns
app_feat.shape
app_feat['device_id'] = app_feat['device_id'].apply(lambda x:map_device_id[x])
app_feat.to_csv(path.join(FEATURES_DATA_DIR, 'app_features.csv'), index=False)
#save
