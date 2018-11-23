
#Data Preprocessing
# Standardization/Mean Removal
# Min-Max or Scaling Features to a range
# Normalization
# Binarization

import numpy as np
import pandas as pd
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


from sklearn import preprocessing

X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0.,1.,-1.]])

#mean is removed. Data is centered to Zero. This is to remove BIAS
X_scaled = preprocessing.scale(X_train)
X_scaled.mean(axis=0)
X_scaled.std(axis=0)

#Mean is Zero and STD = 1

from sklearn.preprocessing import StandardScaler
scaler = preprocessing.StandardScaler().fit(X_train)

scaler.mean_
scaler.scale_

scaler.transform(X_train)

plt.hist(X_train)

#now we can use the transform on the new dataset

X_test = [[-1., 1., 0.]]
scaler.transform(X_test)


#variable with high variance dominte the objective function and prevent the model/estimator
# from learning from other variables/features


#----------Min-Max scaling------------------

#scaling features to li b/w the min max values..usually they are zero and 1..

X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0.,1.,-1.]])

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)

# now the data is scaled b/w 0 and 1.. o is the minimum and 1 is the max
#very small standaard deviations of features and preserving zero entries

X_test = np.array([[-3., -1., 4.]])
X_test_mimax = min_max_scaler.transform(X_test)


























