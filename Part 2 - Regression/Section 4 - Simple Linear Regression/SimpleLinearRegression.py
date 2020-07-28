#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:24:05 2020

@author: kanzakiken
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
# X = dataset.iloc[:, 0].values.reshape(30,1)
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state = 0)

# =============================================================================
# # feature scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)
# =============================================================================

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results
y_pred = regressor.predict(X_test)  # 要预测当然要用predict方法咯

# visualising the Training set results
plt.scatter(X_train, y_train, c='red', label='Training Set')  # 散点图
plt.plot(X_train, regressor.predict(X_train), c='blue', label='predict')  # 直线
plt.legend()
plt.title('Salary VS Experience(Training Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# visualising the Test set results
plt.scatter(X_test, y_test, c='red', label='Testing Set')
# plt.scatter(X_test, y_pred, c='blue', label='predicting')

plt.plot(X_test, regressor.predict(X_test), c='blue', label='predict')
plt.legend()
plt.title('Salary VS Experience(Test Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()










