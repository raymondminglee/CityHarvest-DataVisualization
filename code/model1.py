# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 23:12:17 2019

@author: ray80
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('data_model_3.csv', delimiter=',', index_col=0)
df = df.drop(['Unnamed: 1'], axis=1)
X = df[['Total population', '  Male', '  Female', 'per', 'un']]
Y = df['served']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
Y_test = Y_test.values
Y_train = Y_train.values

clf = svm.SVR(kernel='rbf', C=1e9, gamma=0.1)
clf.fit(X_train, Y_train)
result = clf.predict(X_test)
accuracy_score(Y_te, result)
