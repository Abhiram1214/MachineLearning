#!/usr/bin/env python
# multiple regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing data
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


#Encoding categorical data for x
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenconder_x = LabelEncoder()
x[:, 3] = labelenconder_x.fit_transform(x[:,3])

    #creating dummy variable
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()

#avoiding the dummy variable trap
x=x[:, 1:] #it will be taken care automatically


#splitting the datasets into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)

#fitting multiple linear regression to our training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#predicting the test set results
y_pred = regressor.predict(x_test)

#building an optimal model through backward elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)

x_opt = x[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

#final optimized value (less p value)
x_opt = x[:, [0,3]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()
