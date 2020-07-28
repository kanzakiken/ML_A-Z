#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:32:43 2020

@author: kanzakiken
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)
linear_reg_2 = LinearRegression()
linear_reg_2.fit(X_poly, y)

# visualising the Linear Regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color='r')
plt.plot(X_grid, linear_reg.predict(X_grid), c='blue', label='Linear Regression')
plt.plot(X_grid, linear_reg_2.predict(poly_reg.fit_transform(X_grid)), c='green', label='Polynomial Regression')
plt.legend()
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

print('Linear_R^2', linear_reg.score(X, y))
print('Polynomial_R^2:', linear_reg_2.score(X_poly, y))

# Predicting a new result with Linear Regression
print(linear_reg.predict([[6.5]]))
# Predicting a new result with Polynomial Regression
print(linear_reg_2.predict(poly_reg.fit_transform([[4]])))