import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#feature scaling
#usually it will be taken care of.. but feature scaling in not included in SVR
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = y.reshape((len(y), 1))
y = sc_y.fit_transform(y)


#fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)


#predict the employee salary
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
#we got the scaled prediction of the salary..so, to get the orignal scal of the salary
#we inverse the transform method

#visualizing the linear regression set
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title("truth or bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show()

