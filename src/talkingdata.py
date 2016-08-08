"""This is the capstone project for the Udacity Nanodegree in Machine Learning
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

X = pd.read_csv("data/train_instances.csv")
y = pd.read_csv("data/train_labels.csv")

X.describe()
X.shape
y.shape
y.head(10)
X.head(10)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.33,
                                                    random_state=0)
