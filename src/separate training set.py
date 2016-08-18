""" This script loads the original training data and separates them in instances
and labels and saves them in two separate files"""
import pandas as pd
import pickle as pkl

demographic = pd.read_csv("data/gender_age_train.csv")
labels = pd.DataFrame(demographic['group'])
labels.columns = ['labels']
instances = demographic.drop('group', axis=1)

phone_info = pd.read_csv("data/phone_brand_device_model.csv")
app_info = pd.read_csv("data/app_labels.csv")
app_events = pd.read_csv("data/app_events.csv")
label_categories = pd.read_csv("data/label_categories.csv")
events = pd.read_csv("data/events.csv")

# join all the datasets to create the raw training datasets
data = pd.merge(instances, events, how='left', on='device_id')
data = data.merge(phone_info, how='left', on='device_id')
data = data.merge(app_events, how='left', on='event_id')
data = data.merge(app_info, how='left', on='app_id')
data = data.merge(label_categories, how='left', on='label_id')

# converting all features to numbers to be compatible with numpy
data['gender'], map_gender = pd.factorize(data['gender'])
data['phone_brand'], map_phone_brand = pd.factorize(data['phone_brand'])
data['device_model'], map_device_model = pd.factorize(data['device_model'])
data['category'], map_category = pd.factorize(data['category'])


data.head(10)
len(data)
# save processed data
labels.to_csv('./data/train_labels.csv', index=False)
instances.to_csv('./data/train_instances.csv', index=False)
with open('data/factorize_mappings.pkl', 'w') as f:
    pkl.dump([mapping_gender], f)


test = pd.read_csv("data/gender_age_test.csv")
test.head()
