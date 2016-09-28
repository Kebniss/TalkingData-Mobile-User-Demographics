""" This script loads the raw events dataset, creates the
    features and deals with NaN values."""

import os
import pandas as pd
import numpy as np
import pickle as pkl
from datetime import timedelta
from geopy.distance import great_circle
from get_most_recent_event import get_most_recent_event
from rolling_stats_in_window import rolling_stats_in_window

os.getcwd()
os.chdir('..')
os.chdir('..')
path = os.getcwd() + '\data\\raw\events.csv'
data = pd.read_csv(path,
                    parse_dates=['timestamp'],
                    infer_datetime_format=True)
data.columns = ['event_id', 'device_id', 'timestamp', 'lon', 'lat']

# find and remove nans
nans = {}
cols = data.columns
for _, col in enumerate(cols):
        nulls = data[data[col].isnull()]
        if nulls.empty:
            print 'No NaNs found.'
        else:
            print 'Found NaN values: '
            print nulls
            nans[col] = nulls
            data.drop(nulls, inplace=True)

# remove those instances that are located:
# - in the ocean around the equator (lat~0)
# - in one of the two poles (lat<-55 and lat>69)
# These values are most likely errors
data = (data
        .query("~(lon>49 and lon<95 and lat>-12 and lat<6)") # -indian
        .query("~(lon>-50 and lon<9 and lat>-2 and lat<4)") # -atlantic
        .query("~(lon>150 and lon<=180 and lat>-21 and lat<8)") # -pacific east
        .query("~(lon>=-180 and lon<-81 and lat>-21 and lat<8)") # -pacific west
        .query("lat < 69") # north pole
        .query("lat > -55") # south pole
        .set_index(['device_id', 'timestamp'])
        .sort_index()
        )
duplicates = list(data.index.duplicated()).index(True)
data.drop(data.index[duplicates], inplace=True)

data['lat'] = data['lat'].astype(float)
data['lon'] = data['lon'].astype(float)

# resample data as one per day. Deal with missing data by assuming person never
# moved and filling with previous day data\
daily_data = (data
              .reset_index('device_id')
              .groupby('device_id')
              .resample('1D') # some days might have no readings...
              .mean()
              .ffill()
              .drop('device_id', 1) # so fills them with prev day's position
              )

daily_data_grouped = (daily_data.reset_index()
                      .set_index(['timestamp'])
                      .groupby('device_id', sort=True)
                      )

# shift data of one day in order to measure the day to day distance moved
yesterday = daily_data_grouped.shift(periods=1, freq='D')
yesterday = yesterday.drop('device_id', 1)
yesterday = yesterday.reset_index().set_index(['device_id', 'timestamp'])
# today = daily_data_grouped

# create features
joined = daily_data.join(yesterday,
                   how='left',
                   lsuffix="_today",
                   rsuffix="_yesterday"
                   )

# measure daily distance and add it as a new column
joined['daily_distance'] = joined.apply(
        (lambda x: great_circle(
                            (x.lat_today, x.lon_today),
                            (x.lat_yesterday, x.lon_yesterday)
                            ).kilometers
         ), axis=1
        )

joined.drop(['lon_yesterday', 'lat_yesterday', 'event_id_yesterday',
             'lon_today', 'lat_today', 'event_id_today'
             ],
            axis=1,
            inplace=True
            )

rolled = rolling_stats_in_window(joined,
                               groupby_key='device_id',
                               aggs = ['mean', 'var', 'max'],
                               windows={'1day':1, '2days':2, '3days':3,
                                        '7days':7, '10days':10}
                               )

most_recent_data = get_most_recent_event(rolled.reset_index('device_id'), 'device_id', 'timestamp')
#save
path = os.getcwd() + '\data\processed\periodic_distances.csv'

most_recent_data.to_csv(path)
