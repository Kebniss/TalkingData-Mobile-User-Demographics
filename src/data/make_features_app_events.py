""" This script loads the raw app_events dataset, creates the
    features and deals with NaN values."""

import os
import pandas as pd
import numpy as np
import pickle as pkl
from datetime import timedelta
from rolling_stats_in_window import rolling_stats_in_window

os.getcwd()
os.chdir('..')
os.chdir('..')

path = os.getcwd() + '\data\\raw\events.csv'
time_data = pd.read_csv(path,
                    parse_dates=['timestamp'],
                    infer_datetime_format=True)

path = os.getcwd() + '\data\\raw\\app_events.csv'
app_data = pd.read_csv(path)

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
            data.drop(nulls, inplace=True)

time_data = time_data.drop(['longitude','latitude'], 1)

data = time_data.merge(app_data, on='event_id', how='left')

duplicates = list(data.index.duplicated()).index(True)
data.drop(data.index[duplicates], inplace=True)

data.head()
#save
data.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/train_app_events.csv", index=False)
