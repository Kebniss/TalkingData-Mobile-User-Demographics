import pandas as pd
import pickle as pkl

mod_data = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/raw/Mod_phone_brand_device_model.csv")
data = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/raw/phone_brand_device_model.csv")

latin = mod_data[["phone_brand", "device_model"]]
chinese = data[["phone_brand", "device_model"]]

chinese_to_latin = chinese.join(latin, lsuffix="_chinese", rsuffix="_latin")

brand_chinese_to_latin = (chinese_to_latin[[col for col in chinese_to_latin.columns if "brand" in col]]
                            .drop_duplicates())
model_chinese_to_latin = (chinese_to_latin[[col for col in chinese_to_latin.columns if "model" in col]]
                            .drop_duplicates())

brand_chinese_to_latin.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/brand_mapping.csv", index=False)
model_chinese_to_latin.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/model_mapping.csv", index=False)
