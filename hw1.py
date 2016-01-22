# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

from pandas import DataFrame,Series
import pandas as pd
import numpy as np

from sklearn import linear_model,ensemble,svm,grid_search
import sklearn.metrics

import matplotlib.pyplot as plt

#########################
# READ train.csv
# SPLIT into train and test
#########################
raw = pd.read_csv("train.csv")
raw_train = raw.iloc[:len(raw.index)*9/10,:]
raw_test = raw.iloc[len(raw.index)*9/10:,:]

#########################
# Benchmark 

# LOG-TRANS features
# DIFFERENCE pairwise logarithmically-transformed features
# TRAIN a plain logistic model
#########################

# Log transformation
def log_trans(x):
    return np.log(x+1)  # +1 to take care of 0 values

# Train
features = raw_train.iloc[:,1:12].columns

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


#########################
# Confusion Matrix
#########################
cm_lgt = sklearn.metrics.confusion_matrix(y_test,y_pred)
plt.matshow(cm_lgt)
plt.colorbar()

#########################
# AUC score for ROC curve

# In a symetric case where
# TPR and TNR are of equal interest
# we can just use accuracy score
# or AUC score of ROC curve
#########################

# Accuracy Score
accuracy_lgt = sklearn.metrics.accuracy_score(y_test,y_pred)

# AUC score of ROC curve
roc_auc_lgt = sklearn.metrics.roc_auc_score(y_test,y_pred)

## Plot ROC curve
tpr_lgt = 1.0 * cm_lgt[0,0]/(cm_lgt[0,0] + cm_lgt[0,1])
fpr_lgt = 1.0 * cm_lgt[1,0]/(cm_lgt[1,0] + cm_lgt[1,1])
plt.plot([0,fpr_lgt,1],[0,tpr_lgt,1],'ro-')
plt.plot([0,1],[0,1],'b--')
plt.axis([0,1,0,1])
plt.show()


#########################
# Todo:
# 1.Gradient Boosting --> importance of predictors
# 2.SVM (w/ grid_search to optimize parameters)
#########################
model_gb = sklearn.ensemble.GradientBoostingClassifier()
model_gb.fit(X_train,y_train)
y_pred_gb = model_gb.predict(X_test)

## Feature Importance Plotting
bar_width = 0.5
xtick_place = np.arange(0+bar_width/2,11+bar_width/2,1)

plt.bar(range(0,11),model_gb.feature_importances_, bar_width)
plt.xticks(xtick_place,features,rotation = 'vertical')
plt.xlim(-bar_width,10+2*bar_width)
plt.title("Gradient Boosting: Feature Importance",size=14)

sklearn.metrics.accuracy_score(y_test,y_pred_gb)

