# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# from sklearn.cross_validation import train_test_split 
# No module named 'sklearn.cross_validation'

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

X = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough').fit_transform(X)
# y = LabelEncoder().fit_transform(y)

# Avoiding the Dummy Variable Trap
X = X[:, 1:].astype('float')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_train, y_train))
print(regressor.score(X_test, y_test))
# Predicting the Test set result
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X_train = np.append(arr = np.ones((40,1)), values=X_train, axis=1)

X_opt = X_train[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
print(regressor_OLS.summary())






