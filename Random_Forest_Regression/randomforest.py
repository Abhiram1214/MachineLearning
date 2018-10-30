import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0) #try with n_estimators=100
regressor.fit(x,y)

y_pred = regressor.predict(6.5)


#visualizing the Random regression set for higher resoultion and smooth curve
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title("truth or bluff (Random Forest Regression)")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show()
