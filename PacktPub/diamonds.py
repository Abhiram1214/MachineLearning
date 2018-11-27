
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import collections

diamond = pd.read_csv("diamonds.csv")
data.describe()

#EDA
diamond.dtypes
# some are in float and some are in int
diamond.info()
diamond.describe()
#Columns x (length), y (width), z (height) have some zero values. 0 values for length, width or depth does not make sense hence we remove all such rows.

diamond = diamond.drop(diamond.loc[diamond.x <= 0].index)
diamond = diamond.drop(diamond.loc[diamond.y <= 0].index)
diamond = diamond.drop(diamond.loc[diamond.z <= 0].index)


diamond = diamond.drop(['Unnamed: 0'], axis = 1) # not working

sns.heatmap(diamond.corr(), annot=True)
#According to the correlation matrix price is highly correlated with the following features.
#Carat of Diamond
#Length of Diamond
#Width of Diamond

#Visualize
print("Mean Diamond Carat = " + str(np.mean(diamond.carat)))
#Mean Diamond Carat = 0.7976982566765384
sns.distplot(diamond.carat)

sns.countplot(y=diamond.cut)

print("Mean Diamond Depth Value = " + str(np.mean(diamond.depth)))
#Mean Diamond Depth Value = 61.7495140949547

sns.countplot(diamond.color)

#Feature Encoding is an important step before starting Regression. 
#We have to encode the values such that better feature value has higher numeric value, i.e Ideal = 4 whereas Good = 1. 
#This is important because these values will play important part in regression as the classifier would consider larger value having more impact on the final price as compared to a smaller value.

diamond_cut = {'Fair':0,
               'Good':1,
               'Very Good':2, 
               'Premium':3,
               'Ideal':4}

diamond_color = {'J':0,
                 'I':1, 
                 'H':2,
                 'G':3,
                 'F':4,
                 'E':5,
                 'D':6}

diamond_clarity = {'I1':0,
                   'SI2':1,
                   'SI1':2,
                   'VS2':3,
                   'VS1':4,
                   'VVS2':5,
                   'VVS1':6,
                   'IF':7}

diamond.cut = diamond.cut.map(diamond_cut);
diamond.clarity = diamond.clarity.map(diamond_clarity);
diamond.color = diamond.color.map(diamond_color);



from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

X = diamond.drop(['price'],1)
y = diamond['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Linear Regression

classifier = LinearRegression()
classifier.fit(X_train, y_train)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, verbose=1) 

print('Linear regression accuracy ', classifier.score(X_test, y_test))
#Linear regression accuracy  0.9008633668194134
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))
#mean = 0.8944687565860239, std = 0.02024368489155924



#Ridge Regression
classifier = Ridge(normalize=True)
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2',verbose = 1)
print('Ridge regression accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))



#Lasso Regression
classifier = Lasso(normalize=True)
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2',verbose = 1)
print('Lasso accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))


#Elastic Net Regression
classifier = ElasticNet()
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2',verbose = 1)
print('Elastic Net accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))


#KNeighbors Regression
from sklearn.neighbors import KNeighborsRegressor
classifier = KNeighborsRegressor(n_neighbors=3)
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2',verbose = 1)
print('KNeighbors accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))



#MLP Regression
from sklearn.neural_network import MLPRegressor
classifier = MLPRegressor(hidden_layer_sizes=(14, ), learning_rate_init = 0.1)
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2',verbose = 1)
print('MLP accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))



#Gradient Boosting Regression
from sklearn.ensemble import GradientBoostingRegressor
classifier = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls',verbose = 1)
classifier.fit(X_train,y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, scoring = 'r2')
print('Gradient Boosting Regression accuracy: ', classifier.score(X_test,y_test))
print(accuracies)
print("mean = {0}, std = {1}".format(np.mean(accuracies), np.std(accuracies)))

















