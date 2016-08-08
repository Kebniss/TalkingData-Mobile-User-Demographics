""" This script loads the original training data and separates them in instances
and labels and saves them in two separate files"""
import pandas as pd

data = pd.read_csv("data/gender_age_train.csv")
labels = pd.DataFrame(data['group'])
labels.columns = ['labels']
instances = data.drop('group', axis=1)

labels.to_csv('./data/train_labels.csv', index=False)
instances.to_csv('./data/train_instances.csv', index=False)
