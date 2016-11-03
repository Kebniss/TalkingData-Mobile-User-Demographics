import os
import pickle
import numpy as np
import pandas as pd
from os import path
import seaborn as sns
from operator import add
from scipy import sparse, io
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
from mpl_toolkits.basemap import Basemap
%matplotlib inline

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")

train = pd.read_csv(path.join(RAW_DATA_DIR, 'gender_age_train.csv'))
events = pd.read_csv(path.join(RAW_DATA_DIR, 'events.csv'), parse_dates=['timestamp'])
app_events = pd.read_csv(path.join(RAW_DATA_DIR, 'app_events.csv'))

train = train.sort_values(by='age')
events = events.drop(['longitude', 'latitude'], 1)
app_events['app_id'], map_ids = pd.factorize(app_events['app_id'])

labelled_app_events = (train.merge(events, how='left', on='device_id')
                            .merge(app_events, how='left', on='event_id')
                            )
labelled_app_events = labelled_app_events.fillna(-1)

sns.distplot(train['age'], hist=False)
#here age is not normally distributaed between 20 - 40 are the dominate age

sns.kdeplot(train.age[train['gender']=="M"], label="Male")
sns.kdeplot(train.age[train['gender']=="F"],  label="Female")
plt.legend()

# Age distribution for male and female Female at old age are using more mobile devices then males

print("Male age group count")
ax = sns.countplot(x="group", data=train[train['gender']=="M"])

print("Female age group count")
ax = sns.countplot(x="group", data=train[train['gender']=="F"])


# DAY ------------------------------------------------------------------------
import calendar
labelled_app_events['week_day'] = labelled_app_events.timestamp.apply(lambda x: calendar.day_name[x.weekday()])
ax = sns.countplot(x="week_day", data=labelled_app_events)

days = np.sort(labelled_app_events['week_day'].unique())
ind = np.arange(len(days))  # the x locations for the groups
width = 0.35

male_days = []
female_days = []
for day in days:
    day_rows = labelled_app_events[ labelled_app_events['week_day'] == day]
    female_days.append(len(day_rows.query('gender == "F"')))
    male_days.append(len(day_rows.query('gender == "M"')))

p1 = plt.bar(ind, male_days, width, color="#1292db")
p2 = plt.bar(ind, female_days, width, color="#ff69b4", bottom=male_days)

plt.ylabel('Number of events')
plt.title('Difference in use of the ten most frequent apps between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()

tot = map(add, female_days, male_days)
total = pd.Series(tot, index=range(len(male_days)))
f_users = list(female_days/total)
m_users = list(male_days/total)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "During the whole day the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users, m_users)

labelled_app_events['hours'] = labelled_app_events['timestamp'].apply(lambda x: x.hour)
ax = sns.countplot(x='hours', data=labelled_app_events)

hours = np.sort(labelled_app_events['hours'].unique())
ind = np.arange(len(hours))  # the x locations for the groups
width = 0.35

male_hours = []
female_hours = []
for hour in hours:
    hour_rows = labelled_app_events[ labelled_app_events['hours'] == hour]
    female_hours.append(len(hour_rows.query('gender == "F"')))
    male_hours.append(len(hour_rows.query('gender == "M"')))

p1 = plt.bar(ind, male_hours, width, color="#1292db")
p2 = plt.bar(ind, female_hours, width, color="#ff69b4", bottom=male_hours)
plt.ylabel('Number of events')
plt.title('Difference in use of the ten most frequent apps between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()

tot = map(add, female_hours, male_hours)
total = pd.Series(tot, index=range(len(male_hours)))
f_users = list(female_hours/total)
m_users = list(male_hours/total)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "During the whole day the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users, m_users)

age_h_f = (labelled_app_events[labelled_app_events['gender'] == 'F']
           .groupby(['hours', 'group'])
           .agg('count')
           )
age_h_m = (labelled_app_events[labelled_app_events['gender'] == 'M']
           .groupby(['hours', 'group'])
           .agg('count')
           )
age_h = (labelled_app_events
           .groupby(['hours', 'group'])
           .agg('count')
           )
age_h = age_h['device_id'].reset_index()
groups = age_h.pivot(index='hours', columns='group')
groups.plot(kind='bar', stacked=True)

perc = []
for i in range(groups.shape[1]):
    perc.append(groups.ix[i]/groups.ix[i].sum(0))

groups.columns = groups.columns.get_level_values(1)
plt.figure()
for col in groups.columns:
    groups[col].plot(cmap='Paired')
