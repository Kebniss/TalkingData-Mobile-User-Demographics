import os
import numpy as np
import pandas as pd
from os import path
from scipy import sparse, io
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

phone = io.mmread(path.join(FEATURES_DATA_DIR, 'phone_features_test')).tocsr()
app_l = io.mmread(path.join(FEATURES_DATA_DIR, 'app_features_test')).tocsr()

test_set = hstack((phone, app_l), format='csr')

io.mmwrite(path.join(FEATURES_DATA_DIR, 'test_set_dd'), test_set)
