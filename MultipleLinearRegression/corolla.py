#Consider only the below columns and prepare a prediction model for predicting Price.

#Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
#3 6 8 12 13 15 16 17

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("ToyotaCorolla.csv", encoding="latin1", engine='python')
x = dataset.iloc[:, [3, 6, 8, 12, 13, 15, 16, 17]].values
y = dataset.iloc[:, 2].values

#splitting dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

#simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predict the test test
y_pred = regressor.predict(x_test)

#building an optimal model through backward elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((1436,1)).astype(int), values = x, axis = 1)

x_opt = x[:, [0,1,2,3,4,5,6,7]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()
# pvalue = 0.269 

x_opt = x[:, [0,1,2,3,5,6,7]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()