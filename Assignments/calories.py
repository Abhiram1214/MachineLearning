# Calories_consumed-> predict weight gained using calories consumed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("calories_consumed.csv")
x = dataset.iloc[:, -1:].values
y = dataset.iloc[:, 0].values


# splitting the training set with a test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)


#Linear Regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#predicting the test results
y_pred = regressor.predict(x_test)

#Visualizing the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

