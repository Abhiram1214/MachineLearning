
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset1 = pd.read_csv("Data.csv")
x_axis = dataset1.iloc[:,0:3].values
y_axis = dataset1.iloc[:,3].values

# missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x_axis[:,1:3])
x_axis[:, 1:3] = imputer.transform(x_axis[:, 1:3])

#dealing with categorical value
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
x_axis[:, 0] = label_encoder.fit_transform(x_axis[:, 0])

#dummy variables
onehot_encoder = OneHotEncoder(categorical_features=[0])
x_axis = onehot_encoder.fit_transform(x_axis).toarray()

label_encoder_y = LabelEncoder()
y_axis = label_encoder_y.fit_transform(y_axis)


#spliting the train_set from test_set
from sklearn.model_selection import train_test_split
x_axis_train, x_axis_test, y_axis_train, y_axis_test = train_test_split(x_axis, y_axis, test_size=0.2,random_state=0)


