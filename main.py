#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:56:45 2019

@author: gusimsek
"""

import pandas as pd
import numpy as np
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

X_train = pd.read_csv('/Users/gusimsek/Desktop/anomaly/UCI HAR Dataset/train/X_train.txt',sep = ",")

X_train = X_train.drop(X_train.columns[0], axis=1).values
#%%
Y_train = pd.read_csv('/Users/gusimsek/Desktop/anomaly/UCI HAR Dataset/train/y_train.txt').values.ravel()

#%%
X_test = pd.read_csv('/Users/gusimsek/Desktop/anomaly/UCI HAR Dataset/test/X_test.txt',sep = ",")

#%%
X_test = X_test.drop(X_test.columns[0], axis=1).values
#%%
Y_test = pd.read_csv('/Users/gusimsek/Desktop/anomaly/UCI HAR Dataset/test/y_test.txt').values.ravel()
#%%
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)
#%%
score = np.sum([(_x == _y) for _x, _y in zip(clf.predict(X_test), Y_test)]) / len(X_test)

#%%
deneme_anomaly = np.zeros(561).reshape(1,-1)
#%%
a = np.full((1,561), 1)
#%%
probs_xtest = clf.predict_proba(X_test)
probs_anomaly = clf.predict_proba(a)

#%%