
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(-10, 10, num=1000)
plt.plot(x, 1 / (1 + np.exp(-x)))
plt.title("Sigmoid functon") #signmoid

# to convert the value to binary, we will eith round the value or provide a cut off point.
 
tmp = [0, 0.4, 0.6, 0.8, 1.0]
np.round(tmp)

#cut off
np.array(tmp) > 0.7

dataset = [[-2.0011, 0],
           [-1.4654, 0],
           [0.0965, 0],
           [1.3881, 0],
           [3.0641, 0],
           [7.6275, 1],
           [5.3324, 1],
           [6.9225, 1],
           [8.6754, 1],
           [7.6737, 1]]

coef = [-0.806605464, 0.2573316]

for row in dataset:
    yhat = 1.0 / (1.0 + np.exp(-coef[0] - coef[1] * row[0]))
    print("yhat {0:.4f}, yhat {1}".format(yhat, round(yhat)))
    

#now in real world, we wont have the real coefficient
    
#Estimate coefficient
#maximum likelihood
# Stochastic Gradient Descent

#using sklearn to estimate coefficient
from sklearn.linear_model import LogisticRegression

X = np.array(dataset)[:,0:1]
y = np.array(dataset)[:, 1]

clf_LR = LogisticRegression(C=2, penalty = 'l1', tol = 0.01)
#c = 1 failed..
clf_LR.fit(X,y)
clf_LR.predict(X)

clf_LR.predict_proba(X)


#excercise
dataset2 = [[ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.4,  0. ],
            [ 0.3,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.1,  0. ],
            [ 1.4,  1. ],
            [ 1.5,  1. ],
            [ 1.5,  1. ],
            [ 1.3,  1. ],
            [ 1.5,  1. ],
            [ 1.3,  1. ],
            [ 1.6,  1. ],
            [ 1. ,  1. ],
            [ 1.3,  1. ],
            [ 1.4,  1. ]]



X = np.array(dataset2)[:,0:1]
y = np.array(dataset2)[:, 1]

log_2 = LogisticRegression(penalty='l2', C=1.0, tol=0.01)
log_2.fit(X,y)
y_pred = log_2.predict(X)


np.column_stack((y_pred, y))

































































