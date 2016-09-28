""" This script loads the raw app_categories dataset, creates the
    features and deals with NaN values."""

import os
import numpy as np
import pandas as pd
from scripts import *
from drop_nans import drop_nans
from operations_on_list import *
from get_most_recent_event import get_most_recent_event
from rolling_most_freq_in_window import rolling_most_freq_in_window

os.getcwd()
os.chdir('..\..')

path = os.getcwd() + '\data\\raw\events.csv'
time_data = pd.read_csv(path,
                        parse_dates=['timestamp'],
                        infer_datetime_format=True)

path = os.getcwd() + '\data\\raw\\app_events.csv'
app_data = pd.read_csv(path)

path = os.getcwd() + '\data\\raw\\app_labels.csv'
app_category = pd.read_csv(path)

# find nans
time_data = drop_nans(time_data)
app_data = drop_nans(app_data)
app_category = drop_nans(app_category)

time_data = time_data.drop(['longitude','latitude'], 1)

app_category = drop_nans(app_category)
app_category = (app_category
                .sort_values(by='app_id')
                .groupby('app_id', as_index=False)['label_id']
                .agg({ 'apps_category_list':( lambda x: list(x) ) })
                )

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

data_installed_cat = time_data.merge(installed_apps_cat,
                                     on='event_id',
                                     how='inner')

data_installed_cat['installed_apps_cat'] = (data_installed_cat
                                            .installed_apps_cat
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

daily_installed_cat['1st_cat_dly'] = most_common_in_list(
                                    daily_installed_cat['installed_apps_cat'],1)
daily_installed_cat['2nd_cat_dly'] = most_common_in_list(
                                    daily_installed_cat['installed_apps_cat'],2)
daily_installed_cat['3rd_cat_dly'] = most_common_in_list(
                                    daily_installed_cat['installed_apps_cat'],3)

data_active_cat = time_data.merge(active_apps_cat, on='event_id', how='inner')
data_active_cat['active_apps_cat'] = (data_active_cat['active_apps_cat']
                                      .apply(flatten_list,1))
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

daily_active_cat['1st_cat_dly'] = most_common_in_list(
                                        daily_active_cat['active_apps_cat'],1)
daily_active_cat['2nd_cat_dly'] = most_common_in_list(
                                        daily_active_cat['active_apps_cat'],2)
daily_active_cat['3rd_cat_dly'] = most_common_in_list(
                                        daily_active_cat['active_apps_cat'],3)
rld_most_installed_cat = rolling_most_freq_in_window(
                                        daily_installed_cat['installed_apps_cat'],
                                        col_to_roll ='installed_apps_cat',
                                        groupby_key='device_id',
                                        windows=[2, 3, 7, 10]
                                        )

recent_dly_instll_cat = get_most_recent_event(daily_installed_cat,
                                              groupby_key='device_id')
recent_dly_actv_cat = get_most_recent_event(daily_active_cat,
                                            groupby_key='device_id')

installed_cat_feat = rld_most_installed_cat.merge(recent_dly_instll_cat
                                                  .drop('installed_apps_cat', 1)
                                                  .reset_index(),
                                                  on=['device_id', 'timestamp'],
                                                  how='inner'
                                                  )

rld_most_active_cat = rolling_most_freq_in_window(
                                            daily_active_cat['active_apps_cat'],
                                            col_to_roll = 'active_apps_cat',
                                            groupby_key='device_id',
                                            windows=[2, 3, 7, 10]
                                            )

active_cat_feat = rld_most_active_cat.merge(recent_dly_actv_cat
                                            .drop('active_apps_cat', 1)
                                            .reset_index(),
                                            on=['device_id', 'timestamp'],
                                            how='inner'
                                            )

categories_features = installed_cat_feat.merge(active_cat_feat,
                                               on=['device_id', 'timestamp'],
                                               how='inner',
                                               suffixes=['_installed', '_active']
                                               )

path = os.getcwd() + '\data\\processed\\categories_features.csv'
categories_features.to_csv(path)
