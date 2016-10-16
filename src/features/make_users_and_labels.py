""" This script loads the raw training data, separates them in
    users and labels and saves them in two separate files"""
import pandas as pd
import pickle as pkl
import os.path

demographic = pd.read_csv(
            "/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/data/raw/gender_age_train.csv")
users = pd.DataFrame(demographic['device_id'])
users.columns = ['device_id']
labels = pd.DataFrame(demographic['group'])
labels.columns = ['labels']

users.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/data/processed/train_users.csv")
labels.to_csv("/Users/ludovicagonella/Documents/Projects/kaggle-talkingdata-mobile/data/processed/train_labels.csv")
