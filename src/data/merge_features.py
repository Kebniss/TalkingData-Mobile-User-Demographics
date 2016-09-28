import os
import pandas as pd
import numpy as np

os.getcwd()
os.chdir('..\..')

path = os.getcwd() + '\data\processed\\app_id_features.csv'
app_data = pd.read_csv(path, skip_blank_lines=True)

path = os.getcwd() + '\data\\processed\\categories_features.csv'
cat_data = pd.read_csv(path)

app_data = app_data.drop('n_app_installed_daily_day_var',1)
cat_data = cat_data.drop('Unnamed: 0', 1)

app_features = app_data.merge(cat_data,
                              on=['device_id','timestamp'],
                              how='inner')

path = os.getcwd() + '\data\\processed\\specs_features.csv'
specs = pd.read_csv(path)

app_specs = app_features.merge(specs, on='device_id', how='inner')

path = os.getcwd() + '\data\\processed\\daily_distances.csv'
dist = pd.read_csv(path)
