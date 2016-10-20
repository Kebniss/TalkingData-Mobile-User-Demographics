import os
import pickle
import numpy as np
import pandas as pd
from os import path
import seaborn as sns
from scipy import sparse, io
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
%matplotlib inline

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

gatrain = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_train.csv'),
                      index_col='device_id')
phone
