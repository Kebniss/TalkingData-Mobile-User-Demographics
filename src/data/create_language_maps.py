import os.path
import pandas as pd
import pickle as pkl

os.getcwd()
os.chdir('..\..')
path = os.getcwd() + '\data\\processed\Mod_phone_brand_device_model.csv'
mod_data = pd.read_csv(path)

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
