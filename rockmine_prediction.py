# -*- coding: utf-8 -*-
"""Importing Modules"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collectinng"""

data = pd.read_csv("sonar data.csv", header = None)

data.head()

data.shape

data.describe() #describe statistical measures

data[60].value_counts() #display counts for each class

data.groupby(60).mean()

"""Removing Class Attribute"""

X = data.drop(columns = 60, axis = 1)
Y = data[60]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 1)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training"""

model = LogisticRegression()
model.fit(X_train, Y_train)

"""Model Evaluation"""

#accuracy of training data
X_train_prediction= model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(training_data_accuracy)

#accuracy of testing data
X_test_prediction= model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(testing_data_accuracy)

"""Predictive System"""

input_dataR = (0.0311,0.0491,0.0692,0.0831,0.0079,0.0200,0.0981,0.1016,0.2025,0.0767,0.1767,0.2555,0.2812,0.2722,0.3227,0.3463,0.5395,0.7911,0.9064,0.8701,0.7672,0.2957,0.4148,0.6043,0.3178,0.3482,0.6158,0.8049,0.6289,0.4999,0.5830,0.6660,0.4124,0.1260,0.2487,0.4676,0.5382,0.3150,0.2139,0.1848,0.1679,0.2328,0.1015,0.0713,0.0615,0.0779,0.0761,0.0845,0.0592,0.0068,0.0089,0.0087,0.0032,0.0130,0.0188,0.0101,0.0229,0.0182,0.0046,0.0038)
input_dataM = (0.0249,0.0119,0.0277,0.0760,0.1218,0.1538,0.1192,0.1229,0.2119,0.2531,0.2855,0.2961,0.3341,0.4287,0.5205,0.6087,0.7236,0.7577,0.7726,0.8098,0.8995,0.9247,0.9365,0.9853,0.9776,1.0000,0.9896,0.9076,0.7306,0.5758,0.4469,0.3719,0.2079,0.0955,0.0488,0.1406,0.2554,0.2054,0.1614,0.2232,0.1773,0.2293,0.2521,0.1464,0.0673,0.0965,0.1492,0.1128,0.0463,0.0193,0.0140,0.0027,0.0068,0.0150,0.0012,0.0133,0.0048,0.0244,0.0077,0.0074)

# converting to numpy array
input_data_array = np.asarray(input_dataR) # prediction for Rock
input_data_array = np.asarray(input_dataM) # prediction for Mine

# reshape the array
input_data_reshape = input_data_array.reshape(1, -1)
prediction = model.predict(input_data_reshape)

if prediction[0] == 'R':
  print("The object is Rock")
else:
  print('The object is Mine')

