import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.4, random_state=0)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

regression = svm.SVR(kernel = 'linear', C=1).fit(X_train, y_train)
regression.score(X_test, y_test)
# 0.6672554157940424

from sklearn.model_selection import cross_val_score
regression = svm.SVR(kernel = 'linear', C=1)
scores = cross_val_score(regression, boston.data, boston.target, cv=5)
print("meand {}, std {}".format(scores.mean(), scores.std() * 2))
#meand 0.4537929318961176, std 0.5777955694968605
#STD is bad


scores = cross_val_score(regression, boston.data, boston.target, cv=5, scoring="neg_mean_squared_error")
print("meand {}, std {}".format(scores.mean(), scores.std() * 2))
#meand -33.69392280665913, std 44.710851267548776

from sklearn.model_selection import KFold

X = ['a','b','c','d']
kf = KFold(n_splits=2)

for train, test in kf.split(X):
    print(train, test)
    
    
#startified K-Fold
from sklearn.model_selection import StratifiedKFold
X= np.ones(10)
y = [0,0,0,0,1,1,1,1,1,1]

skf = StratifiedKFold(n_splits=3)

for train, test in skf.split(X, y):
    print(train, test)