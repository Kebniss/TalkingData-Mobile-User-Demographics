""" This script loads the original training data and separates them in
instances and labels and saves them in two separate files"""
import pandas as pd
import pickle as pkl

demographic = pd.read_csv("data/gender_age_train.csv")
labels = pd.DataFrame(demographic['group'])
labels.columns = ['labels']
instances = demographic.drop('group', axis=1)

test = pd.read_csv("data/gender_age_test.csv")
test.head()
phone_info['device_model'].value_counts()



phone_info = pd.read_csv("data/phone_brand_device_model.csv")
app_info = pd.read_csv("data/app_labels.csv")
app_events = pd.read_csv("data/app_events.csv")
label_categories = pd.read_csv("data/label_categories.csv")
events = pd.read_csv("data/events.csv", parse_dates=['timestamp'],
                     infer_datetime_format=True, dayfirst=True)

events.head()['timestamp'].apply(lambda x: x.year if x else 0)

label_categories



# check if there are nan values in the datasets
datasets = [instances, phone_info, events, app_info, app_events,
            label_categories]
names = ['instances', 'phone_info', 'events', 'app_info', 'app_events',
         'label_categories']
nans = {}
for i, dataset in enumerate(datasets):
    cols = dataset.columns
    for _, col in enumerate(cols):
        nulls = dataset[dataset[col].isnull()]
        if nulls.empty:
            print 'No NaNs found.'
        else:
            print 'Found NaN values: '
            print nulls
            nans[(names[i], col)] = nulls
            dataset.drop(dataset.index[nulls.index], inplace=True)


phone_info.pivot(index='device_id', columns='phone_brand', values='device_model')
# check if there are multiple values for the keys used to merge the datasets
instances['device_id'].value_counts().head()
phone_info['device_id'].value_counts().head(550)
phone_info.shape
events['device_id'].value_counts().head()
events['event_id'].value_counts().head()
app_events['event_id'].value_counts().head()
app_events['app_id'].value_counts().head()
app_info['app_id'].value_counts().head()
app_info['label_id'].value_counts().head()
label_categories['label_id'].value_counts().head()

# join LEFT all the datasets to create the raw training datasets
data_l = pd.merge(instances, events, how='left', on='device_id')
data_l = data_l.merge(phone_info, how='left', on='device_id')
data_l = data_l.merge(app_events, how='left', on='event_id')
data_l = data_l.merge(app_info, how='left', on='app_id')
data_l = data_l.merge(label_categories, how='left', on='label_id')

# join all the datasets to find their INTERSECTION
data_i = pd.merge(instances, events, how='inner', on='device_id')
data_i = data_i.merge(phone_info, how='inner', on='device_id')
data_i = data_i.merge(app_events, how='inner', on='event_id')
data_i = data_i.merge(app_info, how='inner', on='app_id')
data_i = data_i.merge(label_categories, how='inner', on='label_id')

# compare the two joins
print "Amount of data when left joined: ", data_l.shape
print "Amount of data when inner joined: ", data_i.shape
print "data_l.head():"
print data_l.head()
print "data_l.head():"
print data_i.head()
print "data_l.describe()"
print data_l.describe()
print "data_i.describe()"
print data_i.describe()

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
