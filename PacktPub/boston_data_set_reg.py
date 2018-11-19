'''
#Supervised - Simple Linear Regression
import random 
import numpy as np

x = 10 * np.random.rand(100)
y = 3 * x + np.random.rand(100)

plt.scatter(x, y)

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)

#Arrange the data into a feature Matrix & target array as a vector
X = x.reshape(-1,1)
X.shape

#fit your model to data
model.fit(X, y)

model.coef_ 
model.intercept_

x_fit = np.linspace(-1, 11) # designing the regrssion line

X_fit = x_fit.reshape(-1,1)

y_fit = model.predict(X_fit)

plt.scatter(x,y)
plt.plot(x_fit, y_fit)


'''

#-------------------------------------------HOUSING DATA----------------------------------
'''

import numpy as np
import pandas as pd
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

boston = load_boston()
df = pd.DataFrame(boston.data)
df.head()

col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        'TAX', 'PTRATIO', 'B', 'LSTAT']

 
df.columns = col_names

#adding MEDV to columns
df['MEDV'] = boston['target']
df.head()
df.describe()


#EDA - Exploratory Data Analysis
df.describe()



#visualiztion
sns.pairplot(df, size=1.2)


col_study = ['PTRATIO', 'B', 'LSTAT', 'MEDV']
sns.pairplot(df[col_study])



#correlation analaysis and feature selection
sns.heatmap(df.corr(), annot=True)
df.corr()

'''




#---------------------------------------
import numpy as np
import pandas as pd
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt



from sklearn.datasets import load_boston

boston = load_boston()
df = pd.DataFrame(boston.data)
df.head()

col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        'TAX', 'PTRATIO', 'B', 'LSTAT'] 
df.columns = col_names

#adding MEDV to columns
df['MEDV'] = boston['target']
df.head()

X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X,y)
model.coef_ 
#array([9.10210898]) - positive slope

model.intercept_
#-34.67062077643857

sns.regplot(X,y)
plt.xlabel("Number of rooms per dwelling")
plt.ylabel("Median value of the owner occupied homes in $100000's")

sns.jointplot(x='RM', y='MEDV', data=df, kind='reg', size=10)

#-------For LSTAT--------
X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values
model.fit(X,y)
sns.regplot(X,y)
plt.xlabel("Lower staatus of the population")
plt.ylabel("Median value of the owner occupied homes in $100000's")

#based on our data visualization-... we can see that the linear regression is not a correct fit
#MEDV is capped at 500000k and when working with LTSAT, you see that the data is flattened 
#between 110k and 150K.. no affect when x increases











