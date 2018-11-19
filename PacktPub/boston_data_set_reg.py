'''
#Supervised - Simple Linear Regression
import random 
import numpy as np

x = 10 * np.random.rand(100)
y = 3 * x + np.random.rand(100)

plt.scatter(x, y)

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)

#Arrange the data into a feature Matrix & target array as a vector
X = x.reshape(-1,1)
X.shape

#fit your model to data
model.fit(X, y)

model.coef_ 
model.intercept_

x_fit = np.linspace(-1, 11) # designing the regrssion line

X_fit = x_fit.reshape(-1,1)

y_fit = model.predict(X_fit)

plt.scatter(x,y)
plt.plot(x_fit, y_fit)


'''

#-------------------------------------------HOUSING DATA----------------------------------
'''

import numpy as np
import pandas as pd
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

boston = load_boston()
df = pd.DataFrame(boston.data)
df.head()

col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        'TAX', 'PTRATIO', 'B', 'LSTAT']

 
df.columns = col_names

#adding MEDV to columns
df['MEDV'] = boston['target']
df.head()
df.describe()


#EDA - Exploratory Data Analysis
df.describe()



#visualiztion
sns.pairplot(df, size=1.2)


col_study = ['PTRATIO', 'B', 'LSTAT', 'MEDV']
sns.pairplot(df[col_study])



#correlation analaysis and feature selection
sns.heatmap(df.corr(), annot=True)
df.corr()

'''




#---------------------------------------
import numpy as np
import pandas as pd
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt



from sklearn.datasets import load_boston

boston = load_boston()
df = pd.DataFrame(boston.data)
df.head()

col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        'TAX', 'PTRATIO', 'B', 'LSTAT'] 
df.columns = col_names

#adding MEDV to columns
df['MEDV'] = boston['target']
df.head()

#-----For RM---------
X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X,y)
model.coef_ 
#array([9.10210898]) - positive slope

model.intercept_
#-34.67062077643857

sns.regplot(X,y)
plt.xlabel("Number of rooms per dwelling")
plt.ylabel("Median value of the owner occupied homes in $100000's")

sns.jointplot(x='RM', y='MEDV', data=df, kind='reg', size=10)

#-------For LSTAT--------
X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values
model.fit(X,y)
sns.regplot(X,y)
plt.xlabel("Lower staatus of the population")
plt.ylabel("Median value of the owner occupied homes in $100000's")

#based on our data visualization-... we can see that the linear regression is not a correct fit
#MEDV is capped at 500000k and when working with LTSAT, you see that the data is flattened 
#between 110k and 150K.. no affect when x increases


#----------Robust Regression----------------
#RANSAC Regressor....linear regression in its current state is prone to outliers.
#if outliers increase the corelation coefficient decreases
#check stats_applet- http://digitalfirst.bfwpub.com/stats_applet/stats_applet_5_correg.html
 
#---------RANSAC ALGORITHM-----------
#RANSAC is an iterative algorithm for the robust estimation of 
#parameters from a subset of inliers from the complete data set. 
#https://scikit-learn.org/stable/modules/linear_model.html#ransac-random-sample-consensus



#-----for RM----
X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values

from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor()

ransac.fit(X,y)


inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_x = np.arange(3,10,1)
line_y_ransac = ransac.predict(line_x.reshape(-1,1))

sns.set(style="darkgrid", context="notebook")
plt.scatter(X[inlier_mask], y[inlier_mask], 
            c='blue', marker='o', label='inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], 
            c='brown', marker='s', label='outliers')
plt.plot(line_x, line_y_ransac, color='red')
plt.xlabel("AVG rooms per dwelling")
plt.ylabel("Median value of the owner occupied homes in $100000's")
plt.legend()

ransac.estimator_.coef_
#array([8.76796372])

ransac.estimator_.intercept_
#-31.692591588101912


#---------for LSTAT-------
X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values

ransac.fit(X,y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inliers)

line_x = np.arange(0,50,1)
line_y_ransac = ransac.predict(line_x.reshape(-1,1))

sns.set(style="darkgrid", context="notebook")
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='blue', marker='o', label="inliers")
plt.scatter(X[outlier_mask], y[outlier_mask],
            c="green", marker='s', label="outliers")
plt.plot(line_x, line_y_ransac, color="red")
plt.xlabel("LSTAT")
plt.ylabel("Median value of the owner occupied homes in $100000's")
plt.legend()


#--------------Performance Evaluation of Regression model-----------------

from sklearn.model_selection import train_test_split

X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

#------Method1: Residual analysis----------------------

plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label = "training data")
#y_train_pred - y_train = error
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='s', label ="test data")
plt.xlabel("Predicted values")
plt.ylabel("Errors")
plt.legend()
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='k')
plt.xlim([-10, 50])

#-------method2: Mean Squared Error (MSE)-----------------
from sklearn.metrics import mean_squared_error

mean_squared_error(y_train, y_train_pred)
#36.523966406959666

mean_squared_error(y_test, y_test_pred)
#46.33630536002592 

#error gone up.

#----------Method3: Co-efficient of determination--------------

from sklearn.metrics import r2_score
r2_score(y_train, y_train_pred)
#0.571031588576562
r2_score(y_test, y_test_pred)
#0.43095672846187616

#bigger the better.. values between 0 and 1.
#1 perfect explanation...0 not so much



#--------near perfect model...point of reference-------

generate_random = np.random.RandomState(0)
X = 10 * generate_random.rand(1000)
y = 3 * X + np.random.randn(1000)

plt.scatter(x, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train.reshape(-1,1), y_train)

y_train_pred = model.predict(X_train.reshape(-1,1))
y_test_pred = model.predict(X_test.reshape(-1,1))

#Method 1

plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label = "training data")
#y_train_pred - y_train = error
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label ="test data")
plt.xlabel("Predicted values")
plt.ylabel("Errors")
plt.legend()
plt.hlines(y=0, xmin=-3, xmax=33, lw=2, color='k')
plt.xlim([-5, 35])

#Method 2
from sklearn.metrics import mean_squared_error

mean_squared_error(y_train, y_train_pred)
#1.0295598483906578

mean_squared_error(y_test, y_test_pred)
#41.0434871745260852

#error gone up. Lesser the better

#----------Method3: Co-efficient of determination--------------

from sklearn.metrics import r2_score
r2_score(y_train, y_train_pred)
#0.9864735003238377
r2_score(y_test, y_test_pred)
#0.9864069519436973

#higher the better









































