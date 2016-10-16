
import os
import pickle
import numpy as np
import pandas as pd
from os import path
from scipy import sparse, io
from sklearn.metrics import log_loss
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

MODELS_DIR = os.environ.get("MODELS_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")
PREDICTIONS_DIR = os.environ.get("PREDICTIONS_DIR")
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR")


data = io.mmread(path.join(FEATURES_DATA_DIR, 'test_set_dd_enh'))
gatest = pd.read_csv(os.path.join(RAW_DATA_DIR,'gender_age_test.csv'),
                     index_col = 'device_id')
data = data.tocsr()

with open(path.join(MODELS_DIR,'logistic_0003c_multinomial_lbfgs_specs_feat.pkl'), 'rb') as f:
    model = pickle.load(f)

pred = model.predict_proba(data)

with open(path.join(FEATURES_DATA_DIR, 'targetencoder.pkl'), 'rb') as f:
    targetencoder = pickle.load(f)

labels = targetencoder.inverse_transform(model.classes_)
pred = pd.DataFrame(pred)
pred.columns = labels
pred['device_id'] = gatest.index

pred.to_csv(path.join(PREDICTIONS_DIR, 'logistic_003c_multinomial_lbfgs_specs_feat.csv'), index=False)
