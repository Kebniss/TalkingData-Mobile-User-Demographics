""" This script loads the raw events dataset, creates the
    features and deals with NaN values."""

import pandas as pd
import numpy as np
import pickle as pkl
import os.path
from geopy.distance import distance

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
data = data.set_index(['device_id', 'timestamp']).sort_index()

joined = data.join(data.shift(-1), how='left', lsuffix="_today", rsuffix="_tomorrow")

joined

joined['distance'] = joined.apply(
                        (lambda x: distance(x.latitude_today, x.longitude_today,
                                           x.latitude_tomorrow, x.longitude_tomorrow)), axis=1)

joined

# create features
sorted_data = data.sort_values(by='timestamp', kind='mergesort')
sorted_data = sorted_data.sort_values(by='device_id', kind='mergesort')
sorted_data["distance"] = np.NaN
grouped = sorted_data.groupby('device_id')


ledist = distance(0, 0, 0, 5)
ledist.
euclidean_square_dist(0, 0, 0, 5)

for k, gp in grouped:
    # measure distance
    distances = [0]
    for i in range(len(gp) - 1):
        # measuring euclidean distance
        distance = ((pow(gp.iloc[i+1]['longitude'] - gp.iloc[i]['longitude'], 2)
                    + pow(gp.iloc[i+1]['latitude'] - gp.iloc[i]['latitude'], 2))
                    ** (0.5))
        distances.append(distance)

    gp['distance'] = distances
sorted_data.head()
data['device_id'].value_counts()

#save
data.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/train_events.csv", index=False)
