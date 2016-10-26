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

age_h = age_h.groupby('hours')['device_id'].agg({'count_per_group':( lambda x: list(x))})

age_h.plot( x='hours', kind='bar', stacked=True, figsize=(24,12))
age_h[pd.isnull(age_h.device_id)]
groups = age_h.pivot(index='hours', columns='group')
groups.plot(kind='bar', stacked=True)

df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])


# Installed apps --------------------------------------------------------------

installed_events = labelled_app_events.query("is_installed == 1.0")
most_installed = installed_events['app_id'].value_counts().head(30)
print most_installed
least_installed = installed_events['app_id'].value_counts(ascending=True).head(20)
print least_installed

female_most_installed = []
male_most_installed = []
for app in most_installed.index:
    app_rows = installed_events[ installed_events['app_id'] == app]
    female_most_installed.append(len(app_rows.query('gender == "F"')))
    male_most_installed.append(len(app_rows.query('gender == "M"')))

f_users = list(female_most_installed/most_installed)
m_users = list(male_most_installed/most_installed)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 most installed apps the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

ind = np.arange(len(most_installed))  # the x locations for the groups
width = 0.35

p1 = plt.bar(ind, male_most_installed, width, color="#1292db")
p2 = plt.bar(ind, female_most_installed, width, color="#ff69b4", bottom=male_most_installed)

plt.ylabel('Number of events')
plt.title('Difference in use of the ten most frequent apps between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()


# Active apps ----------------------------------------------------------------
active_events = labelled_app_events.query("is_active == 1.0")
most_active = active_events['app_id'].value_counts().head(30)
print most_active
least_active = active_events['app_id'].value_counts(ascending=True).head(20)
print least_active

female_most_active = []
male_most_active = []
for app in most_active.index:
    app_rows = active_events[ active_events['app_id'] == app]
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

# First both the least installed and active apps bring null information as they
# are specific for one user only. The most frequents on the other hand bring more
# information on how the users use the devices. However as the plots shows the
# ratio female/male using the apps is practically constant between all the apps.
# The frequency analysis confirms this observation and highlights how this ratio
# is the same of the labels

# --- FEMALE ACTIVITY ---------------------------------------------------------
active_fem = active_events.query("gender == 'F'")
most_active_F = active_fem['app_id'].value_counts().head(30)

female_most_active_F = []
male_most_active_F = []
for app in most_active_F.index:
    app_rows = active_events[ active_events['app_id'] == app]
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
most_active_M = active_mal['app_id'].value_counts().head(30)

female_most_active_M = []
male_most_active_M = []
for app in most_active_M.index:
    app_rows = active_events[ active_events['app_id'] == app]
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


# In the above graphs I isolated the 30 most common apps per gender in order to
# see if there was any app that hinted more to a sex than another. Fromthe results
# it seems that for females most used app the distribution is the same of the
# joint distribution. In the males graph some apps seem to have a predominance of
# one sex over anotyher. Overall it does not seem that these data bring much
# information. There might be some less used app that has a predomminant sex that
# uses it but it will be useful to discriminate only a small amount of people.

# NUMBER OF APP INSTALLED ----------------------------------------------------
gender = (installed_events[['device_id', 'gender']]
          .drop_duplicates(subset='device_id', keep='first')
          .set_index('device_id')
          )
tmp = installed_events[['app_id', 'device_id']]
count = tmp.groupby('device_id').agg('count').sort_values(by='app_id', ascending=False)
tmp_2=tmp.groupby('device_id', as_index=False)['app_id'].agg({'installed_list':(lambda x: list(x))})

tmp_2['unique_installed_count'] = tmp_2['installed_list'].apply(set).apply(len)
tmp_2 = tmp_2.set_index('device_id')

c_g = gender.join(tmp_2['unique_installed_count'], how='inner')

sns.distplot(c_g['unique_installed_count'], hist=False);
#here age is not normally distributaed between 20 - 40 are the dominate age

sns.kdeplot(c_g.unique_installed_count[c_g['gender']=="M"], label="Male")
sns.kdeplot(c_g.unique_installed_count[c_g['gender']=="F"], label="Female")
plt.legend()

# NUMBER OF ACTIVE APPS ------------------------------------------------------

tmp = active_events[['app_id', 'device_id']]
count = tmp.groupby('device_id').agg('count').sort_values(by='app_id', ascending=False)
tmp_2=tmp.groupby('device_id', as_index=False)['app_id'].agg({'active_list':(lambda x: list(x))})

tmp_2['unique_active_count'] = tmp_2['active_list'].apply(set).apply(len)
tmp_2 = tmp_2.set_index('device_id')

c_g = gender.join(tmp_2['unique_active_count'], how='inner')

sns.distplot(c_g['unique_active_count'], hist=False);
#here age is not normally distributaed between 20 - 40 are the dominate age

sns.kdeplot(c_g.unique_active_count[c_g['gender']=="M"], label="Male")
sns.kdeplot(c_g.unique_active_count[c_g['gender']=="F"], label="Female")
plt.legend()


# the graphs above highlight that there is a little but notable difference
# between how many apps male and females have installed
