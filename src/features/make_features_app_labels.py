""" This script loads the raw app_categories dataset, creates the
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

events = pd.read_csv(path.join(RAW_DATA_DIR, 'events.csv'),
                     parse_dates=['timestamp'],
                     infer_datetime_format=True,
                     index_col='event_id'
                     )

appevents = pd.read_csv(path.join(RAW_DATA_DIR, 'app_events.csv'),
                        dtype={'is_installed':bool, 'is_active':bool})
installed = appevents.drop('is_active', 1).query('is_installed == True')
active = appevents.drop('is_installed', 1).query('is_active == True')

applabels = pd.read_csv(os.path.join(RAW_DATA_DIR, 'app_labels.csv'))
# keep only the labels of the apps that we have events of = have been used at least once
applabels_inst = applabels.loc[applabels['app_id'].isin(installed['app_id'].unique())]
applabels_actv = applabels.loc[applabels['app_id'].isin(active['app_id'].unique())]

appencoder = LabelEncoder().fit(appevents['app_id'])
applabels_inst['app'] = appencoder.transform(applabels_inst['app_id'])
applabels_actv['app'] = appencoder.transform(applabels_actv['app_id'])

labelencoder = LabelEncoder().fit(applabels['label_id'])
applabels_inst['label'] = labelencoder.transform(applabels_inst['label_id'])
applabels_actv['label'] = labelencoder.transform(applabels_actv['label_id'])
nlabels = len(labelencoder.classes_)

inst_deviceapps = (installed.merge(events[['device_id']],
                                   how='left',
                                   left_on='event_id',
                                   right_index=True)
                            .groupby(['device_id', 'app'])['app']
                            .agg(['size'])
                     )

actv_deviceapps = (active.merge(events[['device_id']],
                                how='left',
                                left_on='event_id',
                                right_index=True)
                          .groupby(['device_id', 'app'])['app']
                          .agg(['size'])
                     )


app_category = (app_category
                .sort_values(by='app_id')
                .groupby('app_id', as_index=False)['label_id']
                .agg({ 'apps_category_list':( lambda x: list(x) ) })
                )

# find nans
time_data = drop_nans(time_data)
app_data = drop_nans(app_data)
app_category = drop_nans(app_category)

time_data = time_data.drop(['longitude','latitude'], 1)

installed_apps_cat = app_data.merge(app_category, on='app_id', how='left')
installed_apps_cat = (installed_apps_cat
                      .query('is_installed == 1')
                      .groupby('event_id', as_index=False)['apps_category_list']
                      .agg({ 'installed_apps_cat':( lambda x: list(x) ) })
                      )

active_apps_cat = app_data.merge(app_category, on='app_id', how='left')
active_apps_cat = (active_apps_cat
                   .query('is_active == 1')
                   .groupby('event_id', as_index=False)['apps_category_list']
                   .agg({ 'active_apps_cat':( lambda x: list(x) ) })
                   )


data_installed_cat = time_data.merge(installed_apps_cat, on='event_id', how='inner')
data_installed_cat['installed_apps_cat'] = (data_installed_cat['installed_apps_cat']
                                            .apply(flatten_list,1)
                                            )

data_active_cat = time_data.merge(active_apps_cat, on='event_id', how='inner')
data_active_cat['active_apps_cat'] = (data_active_cat['active_apps_cat']
                                      .apply(flatten_list,1)
                                      )


data_installed_cat = data_installed_cat.drop('event_id', 1)
daily_installed_cat = (data_installed_cat
                       .set_index(['device_id','timestamp'])
                       .sort_index()
                       .reset_index('device_id')
                       .groupby('device_id')
                       .installed_apps_cat
                       .resample('1D')
                       .sum()
                       .to_frame() # convert series to dataframe
                       )
daily_installed_cat = daily_installed_cat.reset_index('device_id')

for i in range(daily_installed_cat.shape[0]):
    device = daily_installed_cat.iloc[i]['device_id']
    app_list = daily_installed_cat.iloc[i]['installed_apps_cat']
    if isinstance(app_list, (int, long)):
        pass
    else:
        app_list = set(app_list)
        for elem in app_list:
            installed_matrix[device, elem] = 1

installed_matrix = pd.DataFrame(installed_matrix)
installed_matrix.to_csv('installed_matrix_cat.csv')
installed_matrix = installed_matrix.reset_index()
installed_matrix = installed_matrix.rename(columns = {'index': 'device_id'})

daily_installed_cat['n_app_installed_daily'] = count_list_and_int(daily_installed_cat['installed_apps_cat'],
                                                        max_value=500)

data_active_cat = data_active_cat.drop('event_id', 1)
daily_active_cat = (data_active_cat
                   .set_index(['device_id','timestamp'])
                   .sort_index()
                   .reset_index('device_id')
                   .groupby('device_id')
                   .active_apps_cat
                   .resample('1D')
                   .sum()
                   .to_frame() # convert series to dataframe
                )

daily_active_cat = daily_active_cat.reset_index('device_id')

for i in range(daily_active_cat.shape[0]):
    device = daily_active_cat.iloc[i]['device_id']
    app_list = daily_active_cat.iloc[i]['active_apps_cat']
    if isinstance(app_list, (int, long)):
        pass
    else:
        for elem in app_list:
            active_matrix[device, elem] += 1

active_matrix = pd.DataFrame(active_matrix)
active_matrix.to_csv('active_matrix_cat.csv')
active_matrix = active_matrix.reset_index()
active_matrix = active_matrix.rename(columns = {'index': 'device_id'})

daily_active_cat['n_app_active_daily'] = count_list_and_int(daily_active_cat['active_apps_cat'],
                                                        max_value=500)

recent_dly_instll_cat = get_most_recent_event(daily_installed_cat,
                                              groupby_key='device_id')
recent_dly_instll_cat = recent_dly_instll_cat.reset_index('device_id')

recent_dly_instll_cat = recent_dly_instll_cat.drop('installed_apps_cat',1)
installed_cat = recent_dly_instll_cat.merge(installed_matrix,
                                       on='device_id',
                                       how='inner')

recent_dly_actv_cat = get_most_recent_event(daily_active_cat,
                                            groupby_key='device_id')
recent_dly_actv_cat = recent_dly_actv_cat.reset_index('device_id')
recent_dly_actv_cat = recent_dly_actv_cat.drop('active_apps_cat',1)
active_cat = recent_dly_actv_cat.merge(active_matrix,
                                       on='device_id',
                                       how='inner')

categories_features = installed_cat.merge(active_cat,
                                          on='device_id',
                                          how='inner',
                                          suffixes=['_i', '_a']
                                         )
categories_features['device_id'] = categories_features['device_id'].apply(lambda x:map_device_id[x])

categories_features.to_csv(path.join(FEATURES_DATA_DIR,
                                     'categories_features.csv'),
                           index=False)