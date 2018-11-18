import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting linear regression to the dataset
# we are creating this model to simply compare the results with polynomial reg
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

#we need to fit the polynomial regression into linear reg model
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


#visualizing the linear regression set
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title("truth or bluff (Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show()


#visualizing the polynomial reg set
#x_grid = np.arange(mix(x), max(x), 0.1)
#x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(x, y, color='red')
plt.plot(x, lin_reg2.predict(x_poly), color='blue')
plt.title("truth or bluff (polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show()


#predict using linear regression
lin_reg.predict(6.5)

#predict using Polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))

#The Bias vs Variance trade-off
#Bias refers to the error due to the model’s simplistic assumptions in fitting the data. 
#A high bias means that the model is unable to capture the patterns in the data and this results in under-fitting.
#Variance refers to the error due to the complex model trying to fit the data. 
#High variance means the model passes through most of the data points and it results in over-fitting the data.
#Therefore to achieve a good model that performs well both on the train and unseen data, a trade-off is made.