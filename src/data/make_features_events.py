""" This script loads the raw events dataset, creates the
    features and deals with NaN values."""

import pandas as pd
import pickle as pkl
import os.path
from geopy.geocoders import Nominatim
from rolling_stats_in_window import rolling_stats_in_window

data = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/raw/events.csv",
                    parse_dates=['timestamp'],
                    infer_datetime_format=True)

# find nans
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

# create features

data['timestamp'].head()
df = data.set_index('timestamp')
prova = rolling_stats_in_window(df, 'device_id')

#save
data.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/train_events.csv")
