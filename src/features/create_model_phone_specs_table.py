import os.path
import pandas as pd
import pickle as pkl

path = os.getcwd() + '/data/processed/Mod_phone_brand_device_model.csv'
data = pd.read_csv(path)

specs_table = data.drop_duplicates(['phone_brand', 'device_model'])
specs_table = specs_table.drop('device_id', 1)

path = os.getcwd() + '/data/processed/specs_table.csv'
specs_table.to_csv(path, index=False)
