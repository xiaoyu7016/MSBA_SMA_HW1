# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

from pandas import DataFrame,Series
import pandas as pd
import numpy as np

from sklearn import preprocessing,linear_model,ensemble,svm,grid_search
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

y_pred_lgt = model_logit.predict(X_test)

"""
#########################
# Confusion Matrix
#########################
cm_lgt = sklearn.metrics.confusion_matrix(y_test,y_pred_lgt)
plt.matshow(cm_lgt)
plt.colorbar()
"""

#########################
# AUC score for ROC curve

# In a symetric case where
# TPR and TNR are of equal interest
# we can just use accuracy score
# or AUC score of ROC curve
#########################

# REPORT Accuracy Score
accuracy_lgt = sklearn.metrics.accuracy_score(y_test,y_pred_lgt)

# AUC score of ROC curve
roc_auc_lgt = sklearn.metrics.roc_auc_score(y_test,y_pred_lgt)

## Plot ROC curve
tpr_lgt = 1.0 * cm_lgt[0,0]/(cm_lgt[0,0] + cm_lgt[0,1])
fpr_lgt = 1.0 * cm_lgt[1,0]/(cm_lgt[1,0] + cm_lgt[1,1])
plt.plot([0,fpr_lgt,1],[0,tpr_lgt,1],'ro-')
plt.plot([0,1],[0,1],'b--')
plt.axis([0,1,0,1])
plt.show()


#########################
# Gradient Boosting --> importance of predictors
#########################
model_gb = sklearn.ensemble.GradientBoostingClassifier(loss = 'exponential')
model_gb.fit(X_train,y_train)
y_pred_gb = model_gb.predict(X_test)
  ## Param:
   # loss = 'exponential'   - AdaBoost
   # max_depth = 3
   # learning_rate = 0.1
   # subsample = 1   - Global (Not Stochastic)
   # max_features = 11


## REPORT Accuracy Score
accuracy_gb = sklearn.metrics.accuracy_score(y_test,y_pred_gb) # 80.18%
sklearn.metrics.roc_auc_score(y_test,y_pred_gb)  # 80.17%


## PLOT Feature Importance
bar_width = 0.5
xtick_place = np.arange(0+bar_width/2,11+bar_width/2,1)

plt.bar(range(0,11),model_gb.feature_importances_, bar_width)
plt.xticks(xtick_place,features,rotation = 'vertical')
plt.xlim(-bar_width,10+2*bar_width)
plt.title("Gradient Boosting: Feature Importance",size=14)

"""
## OPTIMIZE PARAMS for Gradient Boosting w/ girdsearchcv
## For some reason accuracy gets slightly worse after optimization
## WEIRD =-=

params = {
          'n_estimators' : [100, 150, 200],
          'learning_rate' : [0.05, 0.08, 0.1, 0.2], 
          'max_depth' : [2,3,4,5],
}

estimator_gb = sklearn.ensemble.GradientBoostingClassifier(loss = 'exponential')
model_gb = sklearn.grid_search.GridSearchCV(estimator_gb,param_grid = params)

model_gb.fit(X_train,y_train)
y_pred_gb = model_gb.predict(X_test)

sklearn.metrics.accuracy_score(y_test,y_pred_gb) ## 79.82%
sklearn.metrics.roc_auc_score(y_test,y_pred_gb)  ## 79.81%
## WHAT THE HACK =-=
"""



#########################
# SELECT Top 6 features:
  # follower_count, following_count, listed_count, retweets_received, network_feature_1, network_feature_2
  # Intuitively, retweets_received seems important 
  # ==> Cut off at 6 features
#########################
top_6 = [0,1,2,4,8,9]
X_train_6 = X_train[:,top_6] 
X_test_6 = X_test[:,top_6]

#########################
# Logistic
#########################
model_logit_6 = linear_model.LogisticRegression(fit_intercept = False)
model_logit_6.fit(X_train_6,y_train)
y_pred_lgt_6 = model_logit_6.predict(X_test_6)

## REPORT Accuracy Score
accuracy_lgt_6 = sklearn.metrics.accuracy_score(y_test,y_pred_lgt_6) # 79.82%
roc_auc_lgt_6 = sklearn.metrics.roc_auc_score(y_test,y_pred_lgt_6)   # 79.83%

#########################
# Gradient Boosting (w/ param optimization)
#########################
params = {
          'n_estimators' : [100, 200, 300, 400, 500],
          'learning_rate' : [0.05, 0.08, 0.1, 0.2], 
          'max_depth' : [2,3,4,5],
         }

estimator_gb_6 = sklearn.ensemble.GradientBoostingClassifier(loss = 'exponential')
model_gb_6 = sklearn.grid_search.GridSearchCV(estimator_gb_6,param_grid = params)
model_gb_6.fit(X_train_6,y_train)
  ## Param:
   # loss = 'exponential'   - AdaBoost
   # n_estimators = 100
   # max_depth = 3
   # learning_rate = 0.1
   # subsample = 1   - Global (Not Stochastic)
   # max_features = 11

y_pred_gb_6 = model_gb.predict(X_test_6)

## REPORT Accuracy Score
accuracy_gb_6 = sklearn.metrics.accuracy_score(y_test,y_pred_gb_6) # 79.27%
roc_auc_gb_6 = sklearn.metrics.roc_auc_score(y_test,y_pred_gb_6)   # 79.27%


#########################
# SVM (w/ param optimization)
#########################
params_svm = {
              'kernel' : ('linear','rbf'),
              'C' : [1,10,100]
             }
X_train_6_n = sklearn.preprocessing.normalize(X_train_6,axis=0)
X_test_6_n = sklearn.preprocessing.normalize(X_test_6,axis=0)

estimator_svm_6 = sklearn.svm.SVC()
model_svm_6 = sklearn.grid_search.GridSearchCV(estimator_svm_6,params_svm)

model_svm_6.fit(X_train_6_n, y_train)
y_pred_svm_6 = model_svm_6.predict(X_test_6_n)

accuracy_svm_6 = sklearn.metrics.accuracy_score(y_test,y_pred_svm_6) # 79.45%
roc_auc_svm_6 = sklearn.metrics.roc_auc_score(y_test,y_pred_svm_6)   # 79.46%

#########################
# Conclusion:

# Top 6 features: 
  # follower_count, following_count, listed_count, retweets_received, network_feature_1, network_feature_2

# Best Model:
  # Logistc w/ log-tranformed features
  # Accuracy: 79.82%
#########################
