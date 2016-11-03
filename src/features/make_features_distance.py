""" This script loads the raw events dataset, creates the
    features and deals with NaN values."""

import os
import os.path
import numpy as np
import pandas as pd
from scripts import *
from datetime import timedelta
from operations_on_list import *
from drop_nans import drop_nans
from geopy.distance import great_circle
from dotenv import load_dotenv, find_dotenv
from get_most_recent_event import get_most_recent_event
from rolling_stats_in_window import rolling_stats_in_window

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

data = pd.read_csv(os.path.join(RAW_DATA_DIR, 'events.csv'),
                   parse_dates=['timestamp'],
                   infer_datetime_format=True)
data.columns = ['event_id', 'device_id', 'timestamp', 'lon', 'lat']

# find and remove nans
data = drop_nans(data)

lat_long_counts = data.groupby(['device_id', 'lat', 'lon'])['event_id'].agg('count').rename('count')
positions = lat_long_counts.reset_index(['lat', 'lon']).groupby(level=0).max()

tex_lat_long_counts = (pd.DataFrame(data
                                    .groupby(['device_id', 'lat', 'lon'])['event_id']
                                    .agg('count')
                                    .rename('count')
                                    )
                       )
tex_lat_long_counts = (tex_lat_long_counts.join(tex_lat_long_counts
                                                .groupby(level=0)
                                                .max()
                                                , rsuffix="_max")
                       )
tex_lat_long_counts = (tex_lat_long_counts
                       .query("count == count_max")
                       .drop("count_max", axis=1)
                       )
data = data.set_index(['device_id', 'lat', 'lon'])
data_max = tex_lat_long_counts.join(data, how='left')
data_max = (data_max
            .reset_index()
            .drop_duplicates(subset='device_id')
            .drop('count',1)
            )
#save
data_max.to_csv(os.path.join(FEATURES_DATA_DIR, 'positions.csv'))
