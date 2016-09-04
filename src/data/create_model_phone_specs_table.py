import pandas as pd
import pickle as pkl

data = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/raw/Mod_phone_brand_device_model.csv")

specs_table = data.drop_duplicates(['phone_brand', 'device_model'])

specs_table.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/raw/specs_table.csv")
