import numpy as np
import pandas as pd
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

boston = load_boston()

df = pd.DataFrame(boston.data)
col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
       'TAX', 'PTRATIO', 'B', 'LSTAT']
df.columns = col_names

df['MEDV'] = boston.target

# we will check the correlation
sns.heatmap(df.corr(), annot=True)
#LSTAT and RM are inversly correlated ?

#Linear model

X = df.iloc[:, 5].values.reshape(-1,1)
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)



plt.scatter(y_pred_train, y_pred_train - y_train, c='blue', marker='o', label = "training data")
#y_train_pred - y_train = error
plt.scatter(y_pred_test, y_pred_test - y_test, c='orange', marker='*', label ="test data")
plt.xlabel("Predicted values")
plt.ylabel("Errors")
plt.legend()
plt.hlines(y=0, xmin=-3, xmax=33, lw=2, color='k')
plt.xlim([-5, 35])



from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

mean_squared_error(y_train, y_pred_train)
r2_score(y_train, y_pred_train)
