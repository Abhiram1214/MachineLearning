import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Reading the data
data = pd.read_csv('winequality-white.csv', sep = ';')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Adding extra Columnt
X = np.append(arr = np.ones((X.shape[0], 1)), values = X, axis = 1)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear regressiong
regressor = LinearRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

r2_score(y_test, predictions)

# Backwards elimination
import statsmodels.formula.api as sm
X_opt = X[:, [0, 1, 2, 4, 6, 8, 9, 10, 11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Displaying the results
import matplotlib.pylab as plt
plt.scatter(y_test, predictions, c='g')
plt.xlabel('True Quality')
plt.ylabel('Predicted Quality')
plt.title('Predicted Quality Against True Quality ')
plt.show()

