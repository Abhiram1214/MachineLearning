import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#import dataset

dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values


sns.set(style="whitegrid")
sns.relplot(x=dataset.Age, y=dataset.EstimatedSalary, hue='size', data=dataset);

# splitting the training set with a test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Fitting Logistic Regression to our training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

#Predicting test set results
y_pred = classifier.predict(x_test)

y_pred == y_test

#making the confusion matrix
from sklearn.metrics import confusion_matrix


