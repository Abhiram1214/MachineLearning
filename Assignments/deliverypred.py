#Delivery_time -> Predict delivery time using sorting time 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#importing data
dataset = pd.read_csv("delivery_time.csv")
x = dataset.iloc[:, -1:].values
y = dataset.iloc[:, 0].values

plt.scatter(x,y)



#splitting the datasets into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state = 0)

#linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#predicting the test results
y_pred = regressor.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)



#visualize the training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train,regressor.predict(x_train), color="blue")
plt.title("Predict delivery time based on sorting time")
plt.xlabel("Sorting time")
plt.ylabel("Delivery time")
plt.show()

#visualize test results
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Predict delivery time based on sorting time(test data)")
plt.xlabel("Sorting time")
plt.ylabel("Delivery time")
plt.show()

