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

train['age'].describe()
age_dist = train.sort_values(by='age')['age'].value_counts()
sns.distplot(train['age'], hist=False)
