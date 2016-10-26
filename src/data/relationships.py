import os
import pickle
import numpy as np
import pandas as pd
from os import path
import seaborn as sns
from operator import add
from scipy import sparse, io
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
from mpl_toolkits.basemap import Basemap
%matplotlib inline

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")

train = pd.read_csv(path.join(RAW_DATA_DIR, 'gender_age_train.csv'))
events = pd.read_csv(path.join(RAW_DATA_DIR, 'events.csv'), parse_dates=['timestamp'])
app_events = pd.read_csv(path.join(RAW_DATA_DIR, 'app_events.csv'))
app_labels = pd.read_csv(path.join(RAW_DATA_DIR, 'app_labels.csv'))
phone = pd.read_csv(path.join(RAW_DATA_DIR, 'phone_brand_device_model.csv'))
train['device_id'].value_counts()

events['device_id'].value_counts()

phone['device_id'].value_counts()

events['event_id'].value_counts()

app_events['event_id'].value_counts()

app_events['app_id'].value_counts()

app_labels['app_id'].value_counts()
