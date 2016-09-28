""" This script loads the raw app_events dataset, creates the
    features and deals with NaN values."""

import os
import numpy as np
import pandas as pd
from scripts import *
from datetime import timedelta
from drop_nans import drop_nans
from operations_on_list import *
from get_most_recent_event import get_most_recent_event
from rolling_stats_in_window import rolling_stats_in_window
from rolling_most_freq_in_window import rolling_most_freq_in_window

os.getcwd()
os.chdir('..\..')

path = os.getcwd() + '\data\\raw\events.csv'
time_data = pd.read_csv(path,
                        parse_dates=['timestamp'],
                        infer_datetime_format=True)

path = os.getcwd() + '\data\\raw\\app_events.csv'
app_data = pd.read_csv(path)

# find nans
time_data = drop_nans(time_data)
app_data = drop_nans(app_data)

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

# resample data as one per day. Deal with missing data by assuming person never
# used the phone
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

#res = deduplicate_list(daily_installed['installed_apps'])
daily_installed['installed_apps'] = deduplicate_list(
                                    daily_installed['installed_apps'])
daily_installed['n_app_installed_daily'] = count_list_and_int(
                                            daily_installed['installed_apps'])

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

daily_active['n_app_active_daily'] = count_list_and_int(
                                                    daily_active['active_apps'])
daily_active['1st_app_dly'] = most_common_in_list(daily_active['active_apps'],1)
daily_active['2nd_app_dly'] = most_common_in_list(daily_active['active_apps'],2)
daily_active['3rd_app_dly'] = most_common_in_list(daily_active['active_apps'],3)

daily_installed = daily_installed.drop('installed_apps', 1)
daily_active = daily_active.drop('active_apps', 1)

# create time windows
rld_dly_installed = rolling_stats_in_window(daily_installed,
                                            groupby_key='device_id',
                                            col_to_roll='n_app_installed_daily',
                                            windows={'day':1, '2days':2,
                                                     '3days':28, '7days':7,
                                                     '10days':10})

rld_dly_active = rolling_stats_in_window(daily_active,
                                         groupby_key='device_id',
                                         ignore_columns=['1st_app_dly',
                                                         '2nd_app_dly',
                                                         '3rd_app_dly'],
                                         col_to_roll='n_app_active_daily',
                                         windows={'day':1, '2days':2,
                                                  '3days':28, '7days':7,
                                                  '10days':10}
                                         )

rld_most_active = rolling_most_freq_in_window(daily_active,
                                              groupby_key='device_id',
                                              col_to_roll='1st_app_dly',
                                              windows=[2, 3, 7, 10]
                                              )

 rld_most_active = rld_most_active.set_index(['device_id', 'timestamp'])

daily_active_1d_log.fillna(value={'nan': -1}, axis=0)

daily_installed_1d_log = get_most_recent_event(daily_installed, groupby_key='device_id', timestamp='timestamp')
daily_active_1d_log = get_most_recent_event(daily_active, groupby_key='device_id', timestamp='timestamp')
rld_dly_installed_1d_log = get_most_recent_event(rld_dly_installed, groupby_key='device_id', timestamp='timestamp')
rld_dly_active_1d_log = get_most_recent_event(rld_dly_active, groupby_key='device_id', timestamp='timestamp')

for c in daily_active_1d_log.columns:
    daily_active_1d_log[c] = fillnan(daily_active_1d_log[c])
for c in daily_installed_1d_log.columns:
    daily_installed_1d_log[c] = fillnan(daily_installed_1d_log[c])
for c in rld_dly_installed_1d_log.columns:
    rld_dly_installed_1d_log[c] = fillnan(rld_dly_installed_1d_log[c])
for c in rld_dly_active_1d_log.columns:
    rld_dly_active_1d_log[c] = fillnan(rld_dly_active_1d_log[c])
for c in rld_most_active.columns:
    rld_most_active[c] = fillnan(rld_most_active[c])

active_apps_feat = daily_active_1d_log.join(rld_most_active, how='inner')
active_apps_feat = active_apps_feat.join(rld_dly_active_1d_log, how='inner')
active_apps_feat = active_apps_feat.drop('n_app_active_daily_day_var', 1)

installed_apps_feat = daily_installed_1d_log.join(rld_dly_installed_1d_log,
                                                  how='inner')

app_feat = active_apps_feat.join(installed_apps_feat,
                                 how='inner',
                                 lsuffix='_active',
                                 rsuffix='_installed'
                                 )

path = os.getcwd() + '\data\processed\\app_id_features.csv'
#save
app_feat.to_csv(path)
