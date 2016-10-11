import os
import pickle
import numpy as np
import pandas as pd
from os import path
from sklearn.metrics import log_loss
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics import make_scorer, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

MODELS_DIR = os.environ.get("MODELS_DIR")
FEATURES_DATA_DIR = os.environ.get("FEATURES_DIR")

data = pd.read_csv(path.join(FEATURES_DATA_DIR, 'test_dataset.csv'))
dev_id = data['device_id']
data = data.drop(['device_model', 'device_id'], 1)

with open(path.join(MODELS_DIR,'gbm_12c_3d_300est_clb.pkl'), 'rb') as f:
    model = pickle.load(f)

pred = model.predict_proba(data)

pred = pd.DataFrame(pred)
pred.columns = model.classes_
pred['device_id'] = dev_id

pred.to_csv(path.join(FEATURES_DATA_DIR, 'predictions.csv'), index=False)
tmp = pd.read_csv(path.join(FEATURES_DATA_DIR, 'predictions.csv'))
tmp.shape
