""" This script loads the raw phone_brand_device_model dataset, creates the
    features and deals with NaN values."""

import pandas as pd
import pickle as pkl
import os.path

data = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/raw/phone_brand_device_model.csv")
phone_specs = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/specs_table.csv")
model_mapping = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/model_mapping.csv")
brand_mapping = pd.read_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/TalkingData-Mobile-User-Demographics/data/processed/brand_mapping.csv")

data = data.merge(brand_mapping, how='left', left_on='phone_brand',
                                      right_on='phone_brand_chinese')
data = data.merge(model_mapping, how='left', left_on='device_model',
                                      right_on='device_model_chinese')
data.drop(['phone_brand_chinese', 'device_model_chinese'], axis=1, inplace=True)

# Le new comment

# find nans
nans = {}
cols = data.columns
for _, col in enumerate(cols):
        nulls = data[data[col].isnull()]
        if nulls.empty:
            print 'No NaNs found.'
        else:
            print 'Found NaN values: '
            print nulls
            nans[col] = nulls

# create features
data = data.merge(phone_specs.ix[:, 2:], left_on=['phone_brand_latin', 'device_model_latin'],
                        right_on=['phone_brand', 'device_model'], how='left',
                        copy=False, suffixes=('', '_right'))
data.drop(['phone_brand_right', 'device_model_right'], axis=1)
#save

data.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/data/processed/train_phone_info.csv")
