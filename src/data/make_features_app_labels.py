""" This script loads the raw app_categories dataset, creates the
    features and deals with NaN values."""

import os
import pandas as pd
import numpy as np
from drop_nans import drop_nans

os.getcwd()
os.chdir('..')
os.chdir('..')

path = os.getcwd() + '\data\\raw\\app_labels.csv'
app_category = pd.read_csv(path)

app_category = drop_nans(app_category)
app_category = (app_category
                .sort_values(by='app_id')
                .groupby('app_id', as_index=False)['label_id']
                .agg({ 'apps_category_list':( lambda x: list(x) ) })
                )
type(app_category.apps_category_list[0])

path = os.getcwd() + '\data\\processed\\app_labels_list.csv'
app_category.to_csv(path)
