#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset

dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#----------------------------------
#Encoding categorical data for x
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenconder_x = LabelEncoder()
x[:, 0] = labelenconder_x.fit_transform(x[:, 0])

#creating dummy variable
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

##Encoding categorical data for y
labelenconder_y = LabelEncoder()
y = labelenconder_y.fit_transform(y)
#---------------------------------

# splitting the training set with a test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)


#Feature scaling
#xstand = (x-mean(x))/std(x)
#xnorm = (x-min(x))/(max(x)-min(x))

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)












