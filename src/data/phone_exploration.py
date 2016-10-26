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
phone = pd.read_csv(path.join(RAW_DATA_DIR, 'phone_brand_device_model.csv'))

phone_t = train.merge(phone, how='left', on='device_id')

most_pop_brand = phone_t['phone_brand'].value_counts().head(20)
print most_pop_brand
least_pop_brand = phone_t['phone_brand'].value_counts(ascending=True).head(30)
print least_pop_brand

female_brands = []
male_brands = []
for brand in most_pop_brand.index:
    app_rows = phone_t[ phone_t['phone_brand'] == brand]
    female_brands.append(len(app_rows.query('gender == "F"')))
    male_brands.append(len(app_rows.query('gender == "M"')))

f_users = list(female_brands/most_pop_brand)
m_users = list(male_brands/most_pop_brand)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 most installed categories the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

ind = np.arange(len(most_pop_brand))  # the x locations for the groups
width = 0.35

p1 = plt.bar(ind, male_brands, width, color="#1292db")
p2 = plt.bar(ind, female_brands, width, color="#ff69b4", bottom=male_brands)

plt.ylabel('Number of events')
plt.title('Difference in use of the ten most frequent labels between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()


# FEMALE --------------------------------------------------------------------

brand_fem = phone_t.query("gender == 'F'")
most_pop_brand = brand_fem['phone_brand'].value_counts().head(20)
print most_pop_brand
least_pop_brand = brand_fem['phone_brand'].value_counts(ascending=True).head(30)
print least_pop_brand

female_brands = []
male_brands = []
for brand in most_pop_brand.index:
    app_rows = phone_t[ phone_t['phone_brand'] == brand]
    female_brands.append(len(app_rows.query('gender == "F"')))
    male_brands.append(len(app_rows.query('gender == "M"')))

tot = map(add, female_brands, male_brands)
total = pd.Series(tot, index=most_pop_brand.index)
f_users = list(female_brands/total)
m_users = list(male_brands/total)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 most installed categories the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

ind = np.arange(len(most_pop_brand))  # the x locations for the groups
width = 0.35

p1 = plt.bar(ind, male_brands, width, color="#1292db")
p2 = plt.bar(ind, female_brands, width, color="#ff69b4", bottom=male_brands)
plt.ylabel('Number of events')
plt.title('Difference in use of the ten most frequent labels between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()


# MALE -----------------------------------------------------------------------

brand_male = phone_t.query("gender == 'F'")
most_pop_brand = brand_male['phone_brand'].value_counts().head(20)
print most_pop_brand
least_pop_brand = brand_male['phone_brand'].value_counts(ascending=True).head(30)
print least_pop_brand

female_brands = []
male_brands = []
for brand in most_pop_brand.index:
    app_rows = phone_t[ phone_t['phone_brand'] == brand]
    female_brands.append(len(app_rows.query('gender == "F"')))
    male_brands.append(len(app_rows.query('gender == "M"')))

tot = map(add, female_brands, male_brands)
total = pd.Series(tot, index=most_pop_brand.index)
f_users = list(female_brands/total)
m_users = list(male_brands/total)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 most installed categories the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

ind = np.arange(len(most_pop_brand))  # the x locations for the groups
width = 0.35

p1 = plt.bar(ind, male_brands, width, color="#1292db")
p2 = plt.bar(ind, female_brands, width, color="#ff69b4", bottom=male_brands)
plt.ylabel('Number of events')
plt.title('Difference in use of the ten most frequent labels between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()

# DEVICE --------------------------------------------------------------------

phone_t['brand_model'] = phone_t['phone_brand'].str.cat(phone_t['device_model'])

most_pop_brand = phone_t['brand_model'].value_counts().head(20)
print most_pop_brand
least_pop_brand = phone_t['brand_model'].value_counts(ascending=True).head(30)
print least_pop_brand

female_brands = []
male_brands = []
for brand in most_pop_brand.index:
    app_rows = phone_t[ phone_t['brand_model'] == brand]
    female_brands.append(len(app_rows.query('gender == "F"')))
    male_brands.append(len(app_rows.query('gender == "M"')))

f_users = list(female_brands/most_pop_brand)
m_users = list(male_brands/most_pop_brand)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 most installed categories the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

ind = np.arange(len(most_pop_brand))  # the x locations for the groups
width = 0.35

p1 = plt.bar(ind, male_brands, width, color="#1292db")
p2 = plt.bar(ind, female_brands, width, color="#ff69b4", bottom=male_brands)
plt.ylabel('Number of events')
plt.title('Difference in use of the ten most frequent device models between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()


# FEMALE --------------------------------------------------------------------

brand_fem = phone_t.query("gender == 'F'")
most_pop_brand = brand_fem['brand_model'].value_counts().head(20)
print most_pop_brand
least_pop_brand = brand_fem['brand_model'].value_counts(ascending=True).head(30)
print least_pop_brand

female_brands = []
male_brands = []
for brand in most_pop_brand.index:
    app_rows = phone_t[ phone_t['brand_model'] == brand]
    female_brands.append(len(app_rows.query('gender == "F"')))
    male_brands.append(len(app_rows.query('gender == "M"')))

tot = map(add, female_brands, male_brands)
total = pd.Series(tot, index=most_pop_brand.index)
f_users = list(female_brands/total)
m_users = list(male_brands/total)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 most installed categories the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

ind = np.arange(len(most_pop_brand))  # the x locations for the groups
width = 0.35

p1 = plt.bar(ind, male_brands, width, color="#1292db")
p2 = plt.bar(ind, female_brands, width, color="#ff69b4", bottom=male_brands)
plt.ylabel('Number of events')
plt.title('Difference in use of the ten most frequent labels between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()


# MALE -----------------------------------------------------------------------

brand_male = phone_t.query("gender == 'F'")
most_pop_brand = brand_male['brand_model'].value_counts().head(20)
print most_pop_brand
least_pop_brand = brand_male['brand_model'].value_counts(ascending=True).head(30)
print least_pop_brand

female_brands = []
male_brands = []
for brand in most_pop_brand.index:
    app_rows = phone_t[ phone_t['brand_model'] == brand]
    female_brands.append(len(app_rows.query('gender == "F"')))
    male_brands.append(len(app_rows.query('gender == "M"')))

tot = map(add, female_brands, male_brands)
total = pd.Series(tot, index=most_pop_brand.index)
f_users = list(female_brands/total)
m_users = list(male_brands/total)

f_users = [ '%.3f' % elem for elem in f_users]
m_users = [ '%.3f' % elem for elem in m_users]

print "For the first 5 most installed categories the percentage of users are: "
print "- females: {}\n- males: {}".format(f_users[:5], m_users[:5])

ind = np.arange(len(most_pop_brand))  # the x locations for the groups
width = 0.35

p1 = plt.bar(ind, male_brands, width, color="#1292db")
p2 = plt.bar(ind, female_brands, width, color="#ff69b4", bottom=male_brands)
plt.ylabel('Number of events')
plt.title('Difference in use of the ten most frequent labels between M and F')
plt.legend((p2[0], p1[0]), ('Women', 'Men'))
plt.show()

# ADD NEW FEATURES -----------------------------------------------------------

specs_table = pd.read_csv(path.join(FEATURES_DATA_DIR, 'specs_table.csv'))
model_mapping = pd.read_csv(path.join(FEATURES_DATA_DIR, 'model_mapping.csv'))
brand_mapping = pd.read_csv(path.join(FEATURES_DATA_DIR, 'brand_mapping.csv'))

phone_t = phone_t.drop_duplicates('device_id')

phone_t = phone_t.merge(brand_mapping, how='left', left_on='phone_brand',
                                      right_on='phone_brand_chinese')
phone_t = phone_t.merge(model_mapping, how='left', left_on='device_model',
                                      right_on='device_model_chinese')
phone_t = phone_t.drop(['phone_brand', 'device_model',
           'phone_brand_chinese', 'device_model_chinese'], axis=1)
phone_t = phone_t.drop_duplicates('device_id')
phone_t = phone_t.rename( columns = {'phone_brand_latin': 'phone_brand',
                               'device_model_latin': 'device_model'})

phone_specs = phone_t.merge(specs_table,
                 on=['phone_brand', 'device_model'],
                 how='left',
                 suffixes=['', '_R'])
phone_specs = phone_specs.fillna(-1)
phone_specs = phone_specs[phone_specs['price_eur'] != -1]
v = phone_specs['price_eur'].value_counts()
v.iloc[0] > sum(v.iloc[1:])/2

sns.distplot(phone_specs['price_eur'])


sns.kdeplot(phone_specs.price_eur[phone_specs['gender'] == 'M'], label='Male')
sns.kdeplot(phone_specs.price_eur[phone_specs['gender'] == 'F'], label='Female')
plt.legend()

sns.kdeplot(phone_specs.screen_size[phone_specs['gender'] == 'M'], label='Male')
sns.kdeplot(phone_specs.screen_size[phone_specs['gender'] == 'F'], label='Female')
plt.legend()

sns.kdeplot(phone_specs.ram_gb[phone_specs['gender'] == 'M'], label='Male')
sns.kdeplot(phone_specs.ram_gb[phone_specs['gender'] == 'F'], label='Female')
plt.legend()

sns.kdeplot(phone_specs.release_year[phone_specs['gender'] == 'M'], label='Male')
sns.kdeplot(phone_specs.release_year[phone_specs['gender'] == 'F'], label='Female')
plt.legend()

sns.kdeplot(phone_specs.camera[phone_specs['gender'] == 'M'], label='Male')
sns.kdeplot(phone_specs.camera[phone_specs['gender'] == 'F'], label='Female')
plt.legend()
