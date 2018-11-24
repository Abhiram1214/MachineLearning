#Lets import modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("kc_house_data.csv")
data.describe()


#check for misssing value
data.isnull().sum()
sns.heatmap(df.isnull())

#check for corelation
cor_mat = df.corr()
cor_mat[np.abs(cor_mat) < 0.6] = 0
sns.heatmap(cor_mat, annot=True)

import statsmodels.api as sm
import statsmodels.formula.api as smf

eigenvalues, eigenvectors = np.linalg.eig(df.corr())
pd.Series(eigenvalues).sort_values()

np.abs(pd.Series(eigenvectors[:,19])).sort_values(ascending=False)

print(data.columns[4], data.columns[11], data.columns[12], df.columns[3])
#bathrooms grade sqft_above bedrooms

data.info()

sns.countplot(x='bedrooms', data=data, order=data['bedrooms'].value_counts().index)

df = pd.DataFrame(data)


X = df[['bathrooms','sqft_living','grade', 'sqft_above']].values

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 3)

reg = LinearRegression()
reg.fit(X, y)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)


#for training:
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_train_pred))  
#Mean Squared Error: 57865260331.49756

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
#Root Mean Squared Error: 249866.55193128675

#For testing:
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))  
#Mean Squared Error: 57865260331.49756

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
#Root Mean Squared Error: 240551.99091152323

plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label = "-training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', label = "test data")
plt.xlabel("Predicted values")
plt.ylabel("Errors")
plt.legend()
plt.hlines(y=0, xmin=-500000, xmax=2500000, lw=2, color='k')






































