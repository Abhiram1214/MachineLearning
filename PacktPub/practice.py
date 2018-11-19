import numpy as np
import pandas as pd
#import missingno as msno
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

boston_dataset = load_boston()
boston_dataset.DESCR

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head
#Median is the target...so we need to add it manually.
boston['MEDV'] = boston_dataset.target

#verify there are no null
boston.isnull().sum()


#Data Correlation
corrMatt = boston.corr()
sn.heatmap(corrMatt, annot=True) #seems to be correlation between RAD and TAX


#Based on the above observations we will use RM and LSTAT as our features. Using a scatter plot let’s see how these features vary with MEDV.
plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("MEDV")
    
# now we need to concatnate LSTAT and RM into x and target into y
    
x = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
y = boston['MEDV']
    
    
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)


# model evaluation for training set
predictions_y_train = regression.predict(x_train)
r2_score(y_train, predictions_y_train) 
rmse_training = (np.sqrt(mean_squared_error(y_train, predictions_y_train)))
print(rmse_training) #5.3656571342244215


# model evaluation for test set

predictions_y_test = regression.predict(x_test)
r2_score(y_test, predictions_y_test) #0.54
rmse = (np.sqrt(mean_squared_error(y_test, predictions_y_test)))
print(rmse) #6.114172522817783












