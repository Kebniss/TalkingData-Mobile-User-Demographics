""" This script loads the raw events dataset, creates the
    features and deals with NaN values."""

import pandas as pd
import numpy as np
import pickle as pkl
from datetime import timedelta
from geopy.distance import great_circle
from rolling_stats_in_window import rolling_stats_in_window

data = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/raw/events.csv",
                    parse_dates=['timestamp'],
                    infer_datetime_format=True)

pd.__version__
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
data = data.set_index(['device_id', 'timestamp']).sort_index()

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

data = data.append(
    pd.DataFrame.from_dict(
        {'device_id': 1000,
        'timestamp': pd.to_datetime("2016-05-01 18:10:44"),
        'event_id': 1,
        'lon': 1,
        'lat': 1},
        orient='index' ).T.set_index(['device_id', 'timestamp']))

data = data.append(
    pd.DataFrame.from_dict(
        {'device_id': 1000,
        'timestamp': pd.to_datetime("2016-05-02 18:10:44"),
        'event_id': 1,
        'lon': 2,
        'lat': 2},
        orient='index' ).T.set_index(['device_id', 'timestamp']))

data = data.append(
    pd.DataFrame.from_dict(
        {'device_id': 1000,
        'timestamp': pd.to_datetime("2016-05-03 18:10:44"),
        'event_id': 1,
        'lon': 3,
        'lat': 3},
        orient='index' ).T.set_index(['device_id', 'timestamp']))

data = data.append(
    pd.DataFrame.from_dict(
        {'device_id': 1000,
        'timestamp': pd.to_datetime("2016-05-05 18:10:44"),
        'event_id': 1,
        'lon': 5,
        'lat': 5},
        orient='index' ).T.set_index(['device_id', 'timestamp']))

data = data.append(
    pd.DataFrame.from_dict(
        {'device_id': 1000,
        'timestamp': pd.to_datetime("2016-05-06 18:10:44"),
        'event_id': 'nan',
        'lon': 'nan',
        'lat': 'nan'},
        orient='index' ).T.set_index(['device_id', 'timestamp']))

data.query('device_id==1000')
# resample data as one per day. Deal with missing data by assuming person never
# moved and filling with previous day data\
data
daily_data = pd.DataFrame()
data = data.reset_index()

daily_data = pd.DataFrame()
device_ids = data['device_id'].unique()
ledata = data.head(1000).set_index('timestamp')
for device_id in device_ids:
    daily_data = daily_data.append(
        ledata
        .query("device_id == '{0}'".format(device_id))
        .resample('D').mean()
        )




cane_di_dio = data.query('device_id == 1000').reset_index('device_id')
dfgb = cane_di_dio.groupby('device_id')#.resample('1D').fillna(method='ffill')
              # some days might have no readings...
               # so fills them with prev day's position

for key, group in daily_data:
    last_elmnt = len(group) - 1
    next_day = group.ix[last_elmnt].name + timedelta(days=1)
    group = group.append(
        pd.DataFrame.from_dict(
            {'device_id': key,
            'timestamp': next_day,
            'event_id': 'nan',
            'lon': 'nan',
            'lat': 'nan'},
            orient='index' ).T.set_index([ 'timestamp']))
daily_data.get_group(-9222956879900151005)
lastRowIdx = daily_data.timestamp.idxmax()
lastRowIdx

daily_data.resample('1D')
daily_data.head()
daily_data_grouped = (daily_data.reset_index()
                      .set_index(['timestamp'])
                      .groupby('device_id', sort=True)
                      )

# shift data of one day in order to measure the day to day distance moved
yesterday = daily_data_grouped.shift(periods=1, freq='D')
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

joined.drop(['lon_yesterday', 'lat_yesterday',
             'lon_today', 'lat_today'
             ],
            axis=1,
            inplace=True
            )

joined['Kappa'] = 0
tmp=joined.reset_index('device_id')
type(tmp)
tmp.head()
temp = rolling_stats_in_window(joined.reset_index('device_id'),
                               groupby_key='device_id',
                               aggs = ['mean', 'var', 'max'],
                               windows={'week':7, 'month':28, 'year':365}
                               )
df = joined.reset_index('device_id')
grouped_df = df.groupby('device_id').resample('1D').mean()
grouped_df
#save
data.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/train_events.csv", index=False)
