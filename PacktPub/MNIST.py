
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

mnist
len(mnist['data'])

#visualization
X, y = mnist['data'], mnist['target']
temp = X[30000]
_temp = temp.reshape(28,28)
plt.imshow(_temp)

#find 4
np.where(y==4)

temp = X[24754]
_temp = temp.reshape(28, 28)
plt.imshow(_temp)

#splitting training and test data
num_split = 60000
X_train, X_test, y_train, y_test = X[:num_split], X[num_split:], y[:num_split], y[num_split:]

#shuffling a dataset
shuffle = np.random.permutation(num_split)
X_train, y_train = X_train[shuffle], y_train[shuffle]

#training a BINARY classifier
#to simplify our problem, we will make it a two-class problem
#we need to first convert our targets to a zero or non zero

y_train_0 = (y_train==0)
y_test_0 = (y_test == 0)

#now we need to pick classifiers and tune their hyper parameters

#SGD classifier
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(random_state=0)
clf.fit(X_train, y_train_0)
    
#prediction
clf.predict(X[1000].reshape(1, -1))



#we dont know how well our model czn predict...so, we use cross vlidation
#measuring accuracy
#Skfold..with split, we get 30 percent from non zero and 70 from zero

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
clf = SGDClassifier(random_state=0)


skfolds = StratifiedKFold(n_splits=3, random_state=100)

for train_index, test_index in skfolds.split(X_train, y_train_0):
    clone_clf = clone(clf)
    X_train_fold = X_train[train_index]
    y_train_folds = (y_train_0[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_0[test_index])
    
    clone_clf.fit(X_train_fold, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print("{0:.4f}".format(n_correct / len(y_pred)))


#whats hapeening here is we discted the data int 3 eual protions
#ex: A,B will be training and C will test
#net A,C training abd B test
#next, B,C training and A test

#0.9739
#0.9732
#0.9881

#another way of doing cross validation
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X_train, y_train_0, cv=3, scoring='accuracy')
#array([0.9739013 , 0.97325   , 0.98814941])  -- same as above



## Danger of Blindly Applying Evaluator As a Performance Measure

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(clf, X_train, y_train_0, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_0, y_train_pred)

from sklearn.metrics import precision_score, recall_score
precision_score(y_train_0, y_train_pred) # 5618 / (574 + 5618)

recall_score(y_train_0, y_train_pred) # 5618 / (305 + 5618)

from sklearn.metrics import f1_score
f1_score(y_train_0, y_train_pred)












