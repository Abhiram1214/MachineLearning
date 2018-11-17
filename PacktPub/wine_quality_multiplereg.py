import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sn
import matplotlib.pyplot as plt

# Reading the data
data = pd.read_csv('winequality-white.csv', sep = ';')

# Missing data detection
msno.matrix(data,figsize=(10,2))

# Distribution
fig, axes = plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(15, 20)
sn.boxplot(data=data,orient="v",ax=axes[0])
sn.boxplot(data=data,y="quality",orient="v",ax=axes[1])

# Correlation analasys
corrMatt = data.corr()
sn.heatmap(corrMatt, annot=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

#adding extra column - y=b0+b1x => column for b0
x = np.append(arr = np.ones((x.shape[0], 1)), values = x, axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y)

#scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Linear Regression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
predictions = regressor.predict(x_test)


#metrics
from sklearn.metrics import r2_score
r2_score(y_test, predictions) #0.24989

#Backward Elimination
#1) select a significance level = 0.05
#2) fit the model with all the independent variable
#3) choose independent variable with the highest p-value
#4) Remove the independent variable
import statsmodels.formula.api as sm
x_opt = x[:, [0,1,2,4,6,8,9,10,11]] #remove 3, 5 and 7
regreesor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regreesor_ols.summary()














