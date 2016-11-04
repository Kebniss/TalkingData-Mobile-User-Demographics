""" This script loads the raw app_events dataset, creates the
    features and deals with NaN values."""

import os
import sys
from os import path
import numpy as np
import pandas as pd
from scripts import *
from scipy import sparse, io
from datetime import timedelta
from drop_nans import drop_nans
from operations_on_list import *
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, hstack
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from get_most_recent_event import get_most_recent_event
from rolling_stats_in_window import rolling_stats_in_window
from rolling_most_freq_in_window import rolling_most_freq_in_window

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

gatrain = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_train.csv'),
                      index_col='device_id')
gatest = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_test.csv'),
                     index_col = 'device_id')

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

events = pd.read_csv(path.join(RAW_DATA_DIR, 'events.csv'),
                     parse_dates=['timestamp'],
                     infer_datetime_format=True,
                     index_col='event_id')

appevents = pd.read_csv(path.join(RAW_DATA_DIR, 'app_events.csv'),
                        dtype={'is_installed':bool, 'is_active':bool})

applabels = pd.read_csv(os.path.join(RAW_DATA_DIR, 'app_labels.csv'))

appencoder = LabelEncoder().fit(appevents['app_id'])
appevents['app'] = appencoder.transform(appevents['app_id'])
napps = len(appencoder.classes_)

installed = appevents.drop('is_active',1).query('is_installed == 1')
active = appevents.drop('is_installed',1).query('is_active == 1')

installed_deviceapps = ( installed.merge(events[['device_id']],
                                         how='left',
                                         left_on='event_id',
                                         right_index=True)
                                  .groupby(['device_id', 'app'])['app']
                                  .agg(['count'])
                       )

installed_deviceapps = ( installed_deviceapps.join(gatrain[['trainrow']], how='left')
                                             .join(gatest['testrow'], how='left')
                                             .reset_index()
                        )
installed_deviceapps.head()

# separate train and test subset and create sparse matrixes
d = installed_deviceapps.dropna(subset=['trainrow'])
total_installed = d.groupby('device_id')['count'].agg(['sum'])
d = d.set_index('device_id')
for d_id in total_installed.index.values:
    d.loc[d_id,'count'] = d.ix[d_id]['count']/total_installed.ix[d_id].values
d = d.reset_index('device_id')
Xtr_app_inst = csr_matrix( ( d['count'], (d['trainrow'], d['app']) ),
                             shape=(gatrain.shape[0], napps)
                          )
d = installed_deviceapps.dropna(subset=['testrow'])
total_installed = d.groupby('device_id')['count'].agg(['sum'])
d = d.set_index('device_id')
for d_id in total_installed.index.values:
    d.loc[d_id,'count'] = d.ix[d_id]['count']/total_installed.ix[d_id].values
d = d.reset_index('device_id')
Xte_app_inst = csr_matrix( (d['count'], (d['testrow'], d['app'])),
                            shape=(gatest.shape[0], napps)
                          )

active_deviceapps = ( active.merge(events[['device_id']],
                                   how='left',
                                   left_on='event_id',
                                   right_index=True)
                            .groupby(['device_id', 'app'])['app']
                            .agg(['count'])
                    )

active_deviceapps = ( active_deviceapps.join(gatrain[['trainrow']], how='left')
                                       .join(gatest['testrow'], how='left')
                                       .reset_index()
                     )
active_deviceapps.head()

d = active_deviceapps.dropna(subset=['trainrow'])
total_active = d.groupby('device_id')['count'].agg(['sum'])
d = d.set_index('device_id')
for d_id in total_active.index.values:
    d.loc[d_id,'count'] = d.ix[d_id]['count']/total_active.ix[d_id].values
d = d.reset_index('device_id')
Xtr_app_actv = csr_matrix( ( d['count'], (d['trainrow'], d['app']) ),
                        shape=(gatrain.shape[0], napps)
                     )
d = active_deviceapps.dropna(subset=['testrow'])
total_active = d.groupby('device_id')['count'].agg(['sum'])
d = d.set_index('device_id')
for d_id in total_active.index.values:
    d.loc[d_id,'count'] = d.ix[d_id]['count']/total_active.ix[d_id].values
d = d.reset_index('device_id')
Xte_app_actv = csr_matrix( (d['count'], (d['testrow'], d['app'])),
                            shape=(gatest.shape[0], napps)
                            )

# APP LABELS

# keep only the labels of the apps that we have events of = have been used at least once
applabels_inst = applabels.loc[applabels['app_id'].isin(installed['app_id'].unique())]
applabels_actv = applabels.loc[applabels['app_id'].isin(active['app_id'].unique())]

applabels_inst.loc[:,'app'] = appencoder.transform(applabels_inst['app_id'])
applabels_actv.loc[:,'app'] = appencoder.transform(applabels_actv['app_id'])

labelencoder = LabelEncoder().fit(applabels['label_id'])
applabels_inst.loc[:,'label'] = labelencoder.transform(applabels_inst['label_id'])
applabels_actv.loc[:,'label'] = labelencoder.transform(applabels_actv['label_id'])
nlabels = len(labelencoder.classes_)

inst_devicelabels = (installed_deviceapps[['device_id', 'app']]
                     .merge(applabels_inst[['app', 'label']])
                     .groupby(['device_id','label'])['app']
                     .agg(['count'])
                     )
inst_devicelabels = (inst_devicelabels.join(gatrain[['trainrow']], how='left')
                                      .join(gatest[['testrow']], how='left')
                                      .reset_index()
                                      )
inst_devicelabels.head()

d = inst_devicelabels.dropna(subset=['trainrow'])
total_installed = d.groupby('device_id')['count'].agg(['sum'])
d = d.set_index('device_id')
for d_id in total_installed.index.values:
    d.loc[d_id,'count'] = d.ix[d_id]['count']/total_installed.ix[d_id].values
d = d.reset_index('device_id')
Xtr_label_inst = csr_matrix( (d['count'], (d['trainrow'], d['label'])),
                             shape=(gatrain.shape[0], nlabels)
                             )

d = inst_devicelabels.dropna(subset=['testrow'])
total_installed = d.groupby('device_id')['count'].agg(['sum'])
d = d.set_index('device_id')
for d_id in total_installed.index.values:
    d.loc[d_id,'count'] = d.ix[d_id]['count']/total_installed.ix[d_id].values
d = d.reset_index('device_id')
Xte_label_inst = csr_matrix( (d['count'], (d['testrow'], d['label'])),
                             shape=(gatest.shape[0], nlabels)
                             )

actv_devicelabels = (active_deviceapps[['device_id', 'app']]
                     .merge(applabels_actv[['app', 'label']])
                     .groupby(['device_id','label'])['app']
                     .agg(['count'])
                     )
actv_devicelabels = (actv_devicelabels.join(gatrain[['trainrow']], how='left')
                                      .join(gatest[['testrow']], how='left')
                                      .reset_index()
                                      )
actv_devicelabels.head()


d = actv_devicelabels.dropna(subset=['trainrow'])
total_active = d.groupby('device_id')['count'].agg(['sum'])
d = d.set_index('device_id')
for d_id in total_active.index.values:
    d.loc[d_id,'count'] = d.ix[d_id]['count']/total_active.ix[d_id].values
d = d.reset_index('device_id')
Xtr_label_actv = csr_matrix( (d['count'], (d['trainrow'], d['label'])),
                             shape=(gatrain.shape[0], nlabels)
                             )

d = actv_devicelabels.dropna(subset=['testrow'])
total_active = d.groupby('device_id')['count'].agg(['sum'])
d = d.set_index('device_id')
for d_id in total_active.index.values:
    d.loc[d_id,'count'] = d.ix[d_id]['count']/total_active.ix[d_id].values
d = d.reset_index('device_id')
Xte_label_actv = csr_matrix( (d['count'], (d['testrow'], d['label'])),
                             shape=(gatest.shape[0], nlabels)
                             )

train_apps = hstack( (Xtr_app_actv, Xtr_app_inst, Xtr_label_actv, Xtr_label_inst),
                     format='csr')
test_apps = hstack( (Xte_app_actv, Xte_app_inst, Xte_label_actv, Xte_label_inst),
                    format='csr')

io.mmwrite(path.join(FEATURES_DATA_DIR, 'cum_app_features_train'), train_apps)
io.mmwrite(path.join(FEATURES_DATA_DIR, 'cum_app_features_test'), test_apps)

svd = TruncatedSVD(n_components=500)
svd.fit(train_apps)
train_apps = svd.transform(train_apps)
train_apps = csr_matrix(train_apps)
io.mmwrite(path.join(FEATURES_DATA_DIR, '500SVD_cum_app_features_train'), train_apps)

test_apps = csr_matrix(svd.transform(test_apps))
io.mmwrite(path.join(FEATURES_DATA_DIR, '500SVD_cum_app_features_test'), test_apps)
