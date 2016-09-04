""" This script loads the raw app_events dataset, creates the
    features and deals with NaN values."""

import pandas as pd
import pickle as pkl
import os.path

data = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/raw/app_events.csv")

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

#save
data.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/train_app_events.csv")
