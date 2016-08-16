""" This script loads the original training data and separates them in instances
and labels and saves them in two separate files"""
import pandas as pd
import pickle as pkl

data = pd.read_csv("data/gender_age_train.csv")
labels = pd.DataFrame(data['group'])
labels.columns = ['labels']
instances = data.drop('group', axis=1)

# converting all features to numbers to be compatible with numpy
instances['gender'], mapping_gender = pd.factorize(instances['gender'])
mapping_gender
instances['device_id'], mapping_dev_id = pd.factorize(instances['device_id'])
mapping_dev_id

labels.to_csv('./data/train_labels.csv', index=False)
instances.to_csv('./data/train_instances.csv', index=False)
with open('data/factorize_mappings.pkl', 'w') as f:
    pkl.dump([mapping_gender, mapping_dev_id], f)


test = pd.read_csv("data/gender_age_test.csv")
test.head()
