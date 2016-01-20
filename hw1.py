# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

from pandas import DataFrame,Series
import pandas as pd
from sklearn import linear_model
import sklearn.metrics
import numpy as np

#########################
# READ train.csv
# SPLIT into train and test
#########################
raw = pd.read_csv("train.csv")
raw_train = raw.iloc[:len(raw.index)*9/10,:]
raw_test = raw.iloc[len(raw.index)*9/10:,:]

#########################
# LOG-TRANS features
# DIFFERENCE pairwise logarithmically-transformed features
# TRAIN a plain logistic model
#########################

# Log transformation
def log_trans(x):
    return np.log(x+1)  # +1 to take care of 0 values

# Train
y_train = raw_train.iloc[:,0]

X_train_A = raw_train.iloc[:,1:12].as_matrix()
X_train_B = raw_train.iloc[:,12:].as_matrix()
X_train = log_trans(X_train_A) - log_trans(X_train_B)

model_logit = linear_model.LogisticRegression(fit_intercept = False)
model_logit.fit(X_train,y_train)

# Predict
y_test = raw_test.iloc[:,0]

X_test_A = raw_test.iloc[:,1:12].as_matrix()
X_test_B = raw_test.iloc[:,12:].as_matrix()
X_test = log_trans(X_test_A) - log_trans(X_test_B)

y_pred = model_logit.predict(X_test)

# Benchmark Confusion Matrix
sklearn.metrics.confusion_matrix(y_test,y_pred)

