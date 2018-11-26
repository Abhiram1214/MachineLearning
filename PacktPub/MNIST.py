
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




























