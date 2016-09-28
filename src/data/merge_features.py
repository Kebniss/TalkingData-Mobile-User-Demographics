import os
import numpy as np
import pandas as pd
from scripts import fillnan

os.getcwd()
os.chdir('..\..')

path = os.getcwd() + '\data\processed\\app_id_features.csv'
app_data = pd.read_csv(path, skip_blank_lines=True)
app_cols = app_data.columns
string_cols = [col for col in list(app_cols)
               if any([s in col for s in ['_id', '_most_used', '_app_dly']])]
app_data[string_cols] = app_data[string_cols].astype(str)

path = os.getcwd() + '\data\\processed\\categories_features.csv'
cat_data = pd.read_csv(path)
cat_data['device_id'] = cat_data['device_id'].astype(str)

app_data = app_data.drop('n_app_installed_daily_day_var',1)
cat_data = cat_data.drop('Unnamed: 0', 1)

app_features = app_data.merge(cat_data,
                              on=['device_id','timestamp'],
                              how='inner')
app_features = app_features.drop('timestamp', 1)

path = os.getcwd() + '\data\\processed\\specs_features.csv'
specs = pd.read_csv(path)
specs['device_id'] = specs['device_id'].astype(str)

app_specs = app_features.merge(specs, on='device_id', how='inner')

path = os.getcwd() + '\data\\processed\\distances.csv'
dist = pd.read_csv(path)
dist = dist.drop('timestamp', 1)
dist['device_id'] = dist['device_id'].astype(str)

fin_out = app_specs.merge(dist,
                         on='device_id',
                         how='outer')

string_cols.extend(['phone_brand', 'device_model'])

final_features = fillnan(fin_out, ignore_columns=string_cols)

final_features.info()

path = os.getcwd() + '\data\processed\\final_features.py'
final_features.to_csv(path, index=False)
