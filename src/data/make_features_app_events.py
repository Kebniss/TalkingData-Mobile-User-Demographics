""" This script loads the raw app_events dataset, creates the
    features and deals with NaN values."""

import os
import pandas as pd
import numpy as np
import pickle as pkl
from datetime import timedelta
from drop_nans import drop_nans
from deduplicate_list import deduplicate_list
from count_list_and_int import count_list_and_int
from most_common_in_list import most_common_in_list
from get_most_recent_event import get_most_recent_event
from rolling_stats_in_window import rolling_stats_in_window
from rolling_most_freq_in_window import rolling_most_freq_in_window


os.getcwd()
os.chdir('..')
os.chdir('..')
path = os.getcwd() + '\data\\raw\events.csv'
time_data = pd.read_csv(path,
                        parse_dates=['timestamp'],
                        infer_datetime_format=True)

path = os.getcwd() + '\data\\raw\\app_events.csv'
app_data = pd.read_csv(path)

path = os.getcwd() + '\data\\processed\\app_labels_list.csv'
app_category = pd.read_csv(path)
type(app_category.apps_category_list[0])

# find nans
time_data = drop_nans(time_data)
app_data = drop_nans(app_data)
app_category = drop_nans(app_category)

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

installed_apps_cat = app_data.merge(app_category, on='app_id', how='left')
#installed_apps_cat = installed_apps_cat.drop('Unnamed: 0', 1)
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

daily_active['n_app_active_daily'] = count_list_and_int(daily_active['active_apps'])
daily_active['most_freq_app_dly'] = most_common_in_list(daily_active['active_apps'], 1)

daily_installed = daily_installed.drop('installed_apps', 1).reset_index('device_id')
daily_active = daily_active.drop('active_apps', 1).reset_index('device_id')

def flatten_list(l):
    res = [item for sublist in l for item in sublist]
    return res

data_installed_cat = time_data.merge(installed_apps_cat, on='event_id', how='inner')
data_installed_cat['installed_apps_cat'] = (data_installed_cat.installed_apps_cat
                                            .apply(flatten_list,1))
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

rld_dly_installed = rolling_stats_in_window(daily_installed,
                                            groupby_key='device_id',
                                            col_to_roll='n_app_installed_daily',
                                            windows={'day':1, '2days':2,
                                                     '3days':28, '7days':7,
                                                     '10days':10})

rld_dly_active = rolling_stats_in_window(daily_active,
                                         groupby_key='device_id',
                                         ignore_columns='most_freq_app_dly',
                                         col_to_roll='n_app_active_daily',
                                         windows={'day':1, '2days':2,
                                                  '3days':28, '7days':7,
                                                  '10days':10})

rld_most_active = rolling_most_freq_in_window(daily_active,
                                              groupby_key='device_id',
                                              windows=[2, 3, 7, 10])
rld_most_active = rld_most_active.drop('timestamp',1)

# -----
daily_active_cat = daily_active_cat.reset_index('device_id')
daily_active_cat[['device_id','active_apps_cat']]
format_d = 'YYYY-MM-DD HH:MM:SS'
daily_active_cat.index = pd.to_datetime(daily_active_cat.index, format=format_d)
rld_most_active_cat = rolling_most_freq_in_window(daily_active_cat[['device_id','active_apps_cat']],
                                                   groupby_key='device_id',
                                                   windows=[2, 3, 7, 10])

rld_active = rld_daily_active.merge(rld_most_active, on='device_id', how='inner')

path = os.getcwd() + '\src\\data\\rld_active.csv'
rld_active.to_csv(path,index=False)
rld_dly_active = pd.read_csv(path)
rld_dly_active = rld_dly_active.drop('Unnamed: 1', 1)

path = os.getcwd() + '\src\\data\\rld_dly_installed.csv'
rld_dly_installed.to_csv(path)
#save
data.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/train_app_events.csv", index=False)
