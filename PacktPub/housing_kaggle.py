#Lets import modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("kc_house_data.csv")
data.describe()

df = pd.DataFrame(data)
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




X = df[['bathrooms','sqft_living','grade', 'sqft_above']].values
y = df['price'].values

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
print("R2 score is ", metrics.r2_score(y_train, y_train_pred))
# R2 score is  0.5422114498196673

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


#Mean Squared Error: 57865260331.49756
# R2 score is  0.5422114498196673








#----checking with polynomial

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures()
X_train_poly = poly_reg.fit_transform(X_train)
X_test_poly = poly_reg.fit_transform(X_test)

poly_reg.fit(X_train_poly, y_train)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_train_poly, y_train.reshape(-1,1))

y_train_poly_pred = lin_reg_2.predict(X_train_poly)
y_test_poly_pred = lin_reg_2.predict(X_test_poly)


print("R2 score is ", metrics.r2_score(y_train, y_train_poly_pred))
#   R2 score is  0.6141761191926798

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_train_poly_pred)))
#Root Mean Squared Error: 229387.74555858175

X_fit = np.arange(X.min(), X.max(), 1)[:np.newaxis]

plt.scatter(X_train, y_train, c='blue', marker='o', label = "training data")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='red')

plt.scatter(y_test_poly_pred, y_test_poly_pred - y_test, c='red', marker='s', label = "test data")
plt.plot(X_fit, y_train_poly_pred)
plt.xlabel("Predicted values")
plt.ylabel("Errors")
plt.legend()
plt.hlines(y=0, xmin=-500000, xmax=2500000, lw=2, color='k')


#-------------Feature Scaling using Standard Scaler

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


reg = LinearRegression()
reg.fit(X, y)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)


#for training:
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_train_pred))  
#Mean Squared Error: 57865260331.49756
#after scaling Mean Squared Error: 1520196218896.5513

print("R2 score is ", metrics.r2_score(y_train, y_train_pred))
# R2 score is  0.5422114498196673
#R2 score is  -10.146751692407944

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
#Root Mean Squared Error: 249866.55193128675
#Root Mean Squared Error: 1232962.3752964044

#For testing:
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))  
#Mean Squared Error: 57865260331.49756

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
#Root Mean Squared Error: 240551.99091152323





















