import os
import os.path
import pandas as pd
import pickle as pkl
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")

data = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Mod_phone_brand_device_model.csv')

specs_table = data.drop_duplicates(['phone_brand', 'device_model'])
specs_table = specs_table.drop('device_id', 1)

path = os.getcwd() + '/data/processed/specs_table.csv'
specs_table.to_csv(path, index=False)
