import pandas as pd
import pickle as pkl
import os.path
from rolling_stats_in_window import rolling_stats_in_window

time_space = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/train_events.csv",
                    parse_dates=['timestamp'],
                    infer_datetime_format=True)
app_events = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/train_app_events.csv")

time_space.head()

data = time_space.merge(app_events, on='event_id', how='inner')
data.head()
