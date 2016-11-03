import os
import os.path
import pandas as pd
import pickle as pkl
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")

mod_data = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Mod_phone_brand_device_model.csv')

latin = mod_data[["phone_brand", "device_model"]]
chinese = data[["phone_brand", "device_model"]]

chinese_to_latin = chinese.join(latin, lsuffix="_chinese", rsuffix="_latin")

brand_chinese_to_latin = (chinese_to_latin[[col for col in chinese_to_latin.columns
                                            if "brand" in col]]
                            .drop_duplicates())
model_chinese_to_latin = (chinese_to_latin[[col for col in chinese_to_latin.columns
                                            if "model" in col]]
                            .drop_duplicates())


path = os.getcwd() + "\data/processed/brand_mapping.csv"
brand_chinese_to_latin.to_csv(path, index=False)

path = os.getcwd() + "\data/processed/model_mapping.csv"
model_chinese_to_latin.to_csv(path, index=False)
