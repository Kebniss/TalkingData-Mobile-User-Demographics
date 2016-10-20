import os
import pickle
import numpy as np
import pandas as pd
from os import path
import seaborn as sns
from operator import add
from scipy import sparse, io
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import LabelEncoder
%matplotlib inline

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")

train = pd.read_csv(path.join(RAW_DATA_DIR, 'gender_age_train.csv'))
events = pd.read_csv(path.join(RAW_DATA_DIR, 'events.csv'), parse_dates=['timestamp'])
app_events = pd.read_csv(path.join(RAW_DATA_DIR, 'app_events.csv'))
app_labels = pd.read_csv(path.join(RAW_DATA_DIR, 'app_labels.csv'))

train = train.sort_values(by='age')
events = events.drop(['longitude', 'latitude', 'timestamp'], 1)
app_ids = np.concatenate((app_events['app_id'].unique(), app_labels['app_id'].unique() ),0)
app_enc = LabelEncoder()
app_enc.fit(app_ids)
app_events['app_id'] = app_enc.transform(app_events['app_id'])
app_labels['app_id'] = app_enc.transform(app_labels['app_id'])

labelled_app_events = (train.merge(events, how='left', on='device_id')
                            .merge(app_events, how='left', on='event_id')
                            .merge(app_labels, how='left', on='app_id')
                            )

labelled_app_events = labelled_app_events.fillna(-1)

# Installed apps --------------------------------------------------------------

installed_events = labelled_app_events.query("is_installed == 1.0")
most_installed = installed_events['label_id'].value_counts().head(30)
print most_installed
least_installed = installed_events['label_id'].value_counts(ascending=True).head(30)
print least_installed

female_most_installed = []
male_most_installed = []
for app in most_installed.index:
    app_rows = installed_events[ installed_events['label_id'] == app]
    female_most_installed.append(len(app_rows.query('gender == "F"')))
    male_most_installed.append(len(app_rows.query('gender == "M"')))

f_users = list(female_most_installed/most_installed)
m_users = list(male_most_installed/most_installed)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 most installed categories the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

ind = np.arange(len(most_installed))  # the x locations for the groups
width = 0.35

p1 = plt.bar(ind, male_most_installed, width, color="#1292db")
p2 = plt.bar(ind, female_most_installed, width, color="#ff69b4", bottom=male_most_installed)

plt.ylabel('Number of events')
plt.title('Difference in use of the ten most frequent labels between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()

female_least_installed = []
male_least_installed = []
for app in least_installed.index:
    app_rows = installed_events[ installed_events['label_id'] == app]
    female_least_installed.append(len(app_rows.query('gender == "F"')))
    male_least_installed.append(len(app_rows.query('gender == "M"')))

f_users = list(female_least_installed/least_installed)
m_users = list(male_least_installed/least_installed)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 least installed categories the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

ind = np.arange(len(least_installed))  # the x locations for the groups
width = 0.35

p1 = plt.bar(ind, male_least_installed, width, color="#1292db")
p2 = plt.bar(ind, female_least_installed, width, color="#ff69b4", bottom=male_least_installed)

plt.ylabel('Number of events')
plt.title('Difference in use of the ten least frequent labels between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()

# Active apps ----------------------------------------------------------------
active_events = labelled_app_events.query("is_active == 1.0")
most_active = active_events['label_id'].value_counts().head(30)
print most_active
least_active = active_events['label_id'].value_counts(ascending=True).head(30)
print least_active

female_most_active = []
male_most_active = []
for app in most_active.index:
    app_rows = active_events[ active_events['label_id'] == app]
    female_most_active.append(len(app_rows.query('gender == "F"')))
    male_most_active.append(len(app_rows.query('gender == "M"')))

f_users = list(female_most_active/most_active)
m_users = list(male_most_active/most_active)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 most active apps the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

ind = np.arange(len(most_active))  # the x locations for the groups
width = 0.35

p1 = plt.bar(ind, male_most_active, width, color="#1292db")
p2 = plt.bar(ind, female_most_active, width, color="#ff69b4", bottom=male_most_active)

plt.ylabel('Number of events')
plt.title('Difference in use of the ten most frequent apps between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()

female_least_installed = []
male_least_installed = []
for app in least_installed.index:
    app_rows = installed_events[ installed_events['label_id'] == app]
    female_least_installed.append(len(app_rows.query('gender == "F"')))
    male_least_installed.append(len(app_rows.query('gender == "M"')))

f_users = list(female_least_installed/least_installed)
m_users = list(male_least_installed/least_installed)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 least installed categories the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

ind = np.arange(len(least_installed))  # the x locations for the groups
width = 0.35

p1 = plt.bar(ind, male_least_installed, width, color="#1292db")
p2 = plt.bar(ind, female_least_installed, width, color="#ff69b4", bottom=male_least_installed)

plt.ylabel('Number of events')
plt.title('Difference in use of the ten least frequent labels between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()

# In this case we see something interesting in least active and installed apps.
# Here sthere are some categories that have a clear majority of users of one sex.
# The number of events is very small in all of these cases but it can help
# classify some cases.
# The most frequents on the other hand bring more
# information on how the users use the devices. However as the plots shows the
# ratio female/male using the apps is practically constant between all the apps.
# The frequency analysis confirms this observation and highlights how this ratio
# is the same of the labels

# --- FEMALE ACTIVITY ---------------------------------------------------------
active_fem = active_events.query("gender == 'F'")
most_active_F = active_fem['label_id'].value_counts().head(30)

female_most_active_F = []
male_most_active_F = []
for app in most_active_F.index:
    app_rows = active_events[ active_events['label_id'] == app]
    female_most_active_F.append(len(app_rows.query('gender == "F"')))
    male_most_active_F.append(len(app_rows.query('gender == "M"')))

tot = map(add, female_most_active_F, male_most_active_F)
total = pd.Series(tot, index=most_active_F.index)
f_users = list(female_most_active_F/total)
m_users = list(male_most_active_F/total)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 most active apps the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

p1 = plt.bar(ind, male_most_active_F, width, color="#1292db")
p2 = plt.bar(ind, female_most_active_F, width, color="#ff69b4", bottom=male_most_active_F)

plt.ylabel('Number of events')
plt.title('Difference in use of the 30 most frequent apps among females between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()


# --- MALE ACTIVITY -----------------------------------------------------------
active_mal = active_events.query("gender == 'M'")
most_active_M = active_mal['label_id'].value_counts().head(30)

female_most_active_M = []
male_most_active_M = []
for app in most_active_M.index:
    app_rows = active_events[ active_events['label_id'] == app]
    female_most_active_M.append(len(app_rows.query('gender == "F"')))
    male_most_active_M.append(len(app_rows.query('gender == "M"')))

tot = map(add, female_most_active_M, male_most_active_M)
total = pd.Series(tot, index=most_active_M.index)
f_users = list(female_most_active_M/total)
m_users = list(male_most_active_M/total)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 most active apps the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

p1 = plt.bar(ind, male_most_active_M, width, color="#1292db")
p2 = plt.bar(ind, female_most_active_M, width, color="#ff69b4", bottom=male_most_active_M)

plt.ylabel('Number of events')
plt.title('Difference in use of the 30 most frequent apps among females between M and F')
plt.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.show()


# NUMBER OF APP INSTALLED ----------------------------------------------------
gender = (installed_events[['device_id', 'gender']]
          .drop_duplicates(subset='device_id', keep='first')
          .set_index('device_id')
          )
tmp = installed_events[['label_id', 'device_id']]
count = tmp.groupby('device_id').agg('count').sort_values(by='label_id', ascending=False)
tmp_2=tmp.groupby('device_id', as_index=False)['label_id'].agg({'installed_list':(lambda x: list(x))})

tmp_2['unique_installed_count'] = tmp_2['installed_list'].apply(set).apply(len)
tmp_2 = tmp_2.set_index('device_id')

c_g = gender.join(tmp_2['unique_installed_count'], how='inner')

sns.distplot(c_g['unique_installed_count'], hist=False);
#here age is not normally distributaed between 20 - 40 are the dominate age

sns.kdeplot(c_g.unique_installed_count[c_g['gender']=="M"], label="Male")
sns.kdeplot(c_g.unique_installed_count[c_g['gender']=="F"], label="Female")
plt.legend()


# NUMBER OF ACTIVE APPS ------------------------------------------------------

tmp = active_events[['label_id', 'device_id']]
count = tmp.groupby('device_id').agg('count').sort_values(by='label_id', ascending=False)
tmp_2=tmp.groupby('device_id', as_index=False)['label_id'].agg({'active_list':(lambda x: list(x))})

tmp_2['unique_active_count'] = tmp_2['active_list'].apply(set).apply(len)
tmp_2 = tmp_2.set_index('device_id')

c_g = gender.join(tmp_2['unique_active_count'], how='inner')

sns.distplot(c_g['unique_active_count'], hist=False)
#here age is not normally distributaed between 20 - 40 are the dominate age

sns.kdeplot(c_g.unique_active_count[c_g['gender']=="M"], label="Male")
sns.kdeplot(c_g.unique_active_count[c_g['gender']=="F"], label="Female")
