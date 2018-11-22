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

#-----For RM---------
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


#----------Robust Regression----------------
#RANSAC Regressor....linear regression in its current state is prone to outliers.
#if outliers increase the corelation coefficient decreases
#check stats_applet- http://digitalfirst.bfwpub.com/stats_applet/stats_applet_5_correg.html
 
#---------RANSAC ALGORITHM-----------
#RANSAC is an iterative algorithm for the robust estimation of 
#parameters from a subset of inliers from the complete data set. 
#https://scikit-learn.org/stable/modules/linear_model.html#ransac-random-sample-consensus



#-----for RM----
X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values

from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor()

ransac.fit(X,y)


inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_x = np.arange(3,10,1)
line_y_ransac = ransac.predict(line_x.reshape(-1,1))

sns.set(style="darkgrid", context="notebook")
plt.scatter(X[inlier_mask], y[inlier_mask], 
            c='blue', marker='o', label='inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], 
            c='brown', marker='s', label='outliers')
plt.plot(line_x, line_y_ransac, color='red')
plt.xlabel("AVG rooms per dwelling")
plt.ylabel("Median value of the owner occupied homes in $100000's")
plt.legend()

ransac.estimator_.coef_
#array([8.76796372])

ransac.estimator_.intercept_
#-31.692591588101912


#---------for LSTAT-------
X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values

ransac.fit(X,y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inliers)

line_x = np.arange(0,50,1)
line_y_ransac = ransac.predict(line_x.reshape(-1,1))

sns.set(style="darkgrid", context="notebook")
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='blue', marker='o', label="inliers")
plt.scatter(X[outlier_mask], y[outlier_mask],
            c="green", marker='s', label="outliers")
plt.plot(line_x, line_y_ransac, color="red")
plt.xlabel("LSTAT")
plt.ylabel("Median value of the owner occupied homes in $100000's")
plt.legend()


#--------------Performance Evaluation of Regression model-----------------

from sklearn.model_selection import train_test_split

X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

#------Method1: Residual analysis----------------------

plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label = "training data")
#y_train_pred - y_train = error
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='s', label ="test data")
plt.xlabel("Predicted values")
plt.ylabel("Errors")
plt.legend()
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='k')
plt.xlim([-10, 50])

#-------method2: Mean Squared Error (MSE)-----------------
from sklearn.metrics import mean_squared_error

mean_squared_error(y_train, y_train_pred)
#36.523966406959666

mean_squared_error(y_test, y_test_pred)
#46.33630536002592 

#error gone up.

#----------Method3: Co-efficient of determination--------------

from sklearn.metrics import r2_score
r2_score(y_train, y_train_pred)
#0.571031588576562
r2_score(y_test, y_test_pred)
#0.43095672846187616

#bigger the better.. values between 0 and 1.
#1 perfect explanation...0 not so much



#--------near perfect model...point of reference-------

generate_random = np.random.RandomState(0)
X = 10 * generate_random.rand(1000)
y = 3 * X + np.random.randn(1000)

plt.scatter(x, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train.reshape(-1,1), y_train)

y_train_pred = model.predict(X_train.reshape(-1,1))
y_test_pred = model.predict(X_test.reshape(-1,1))

#Method 1

plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label = "training data")
#y_train_pred - y_train = error
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label ="test data")
plt.xlabel("Predicted values")
plt.ylabel("Errors")
plt.legend()
plt.hlines(y=0, xmin=-3, xmax=33, lw=2, color='k')
plt.xlim([-5, 35])

#Method 2
from sklearn.metrics import mean_squared_error

mean_squared_error(y_train, y_train_pred)
#1.0295598483906578

mean_squared_error(y_test, y_test_pred)
#41.0434871745260852

#error gone up. Lesser the better

#----------Method3: Co-efficient of determination--------------

from sklearn.metrics import r2_score
r2_score(y_train, y_train_pred)
#0.9864735003238377
r2_score(y_test, y_test_pred)
#0.9864069519436973

#higher the better





#------------ Multiple regression with stat models-------------



import numpy as np
import pandas as pd
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.datasets import load_boston

boston_data = load_boston()

df = pd.DataFrame(boston_data.data, columns = boston_data.feature_names)
df.head()
df.shape

X = df
y = boston_data.target


#-----Stat models------------

import statsmodels.api as sm
import statsmodels.formula.api as smf


X_constant = sm.add_constant(X)
#we will add constant to avoid BIAS..if we dont use coefficients estimation will include BIAS and intercept as well.
pd.DataFrame(X_constant)

model = sm.OLS(y, X_constant)
lr = model.fit()
lr.summary()

#-------------

form_lr = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', data=df)
mlr = form_lr.fit()
mlr.summary()
#R-squared: 0.741



mod_lr = smf.ols(formula = 'y ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT', data=df)
mod = mod_lr.fit()
mod.summary()



#----------------Identifying key features is by \
#1) Standardizing vaiables
#2) R^2

#---------------------Feature Extraction-----------

#------Correlation matrix----
# useful to identify the collinearity between predictors
# colinearity is usually present in the model itself.. so, not to worry much about it..but good to know


pd.options.display.float_format = '{:, .4f}'.format
corr_matrix = df.corr()
corr_matrix[np.abs(corr_matrix) < 0.6] = 0
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
#sns.heatmap(df.corr(), annot=True)

#---Detecting Collinearity with Eigen Vectors-----

eigenvalues, eigenvetors = np.linalg.eig(df.corr())
pd.Series(eigenvalues).sort_values()
#Index 8 is very close to zero or small when compared to others.. small values represents presence of collinearity

np.abs(pd.Series(eigenvetors[:,8])).sort_values(ascending=False)
#note that 9,8,2 are very high loading when compared to the rest
#they cause multicolinearity problems

print(df.columns[2], df.columns[8], df.columns[9])
#INDUS RAD TAX  -- these are causing multi colinearity problem


#-------------Feature Importance-------------

#Standardization  --  easy one
#R^2

#now we need to check two things
#1)direction of the coefficient
#2) impact of the variable/factor on the model

#to perform point two, we need to standardze the model..because 
plt.hist(df['TAX'])
#its rather large..
plt.hist(df['NOX'])
#tax is gonna drown up NOX...so we standardize the variable

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X,y)

result = pd.DataFrame(list(zip(model.coef_, df.columns)), columns=['coefficient','name']).set_index('name')
np.abs(result).sort_values(by='coefficient', ascending=False)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

scaler = StandardScaler()
Stand_coef_reg_model = make_pipeline(scaler, model)

#now recheck the coefficients

Stand_coef_reg_model.fit(X,y)
result = pd.DataFrame(list(zip(Stand_coef_reg_model.steps[1][1].coef_, df.columns)), columns=['coefficient','name']).set_index('name')
np.abs(result).sort_values(by='coefficient', ascending=False)
#LSTAT is more significat and AGE is least significant
# all the variables are standarized b/w -3 to +3 and we can explain the variablitiy better
#thank the rest

#Using R^2 to identify keyfeatures
#compare R^2 of model against R^2 of model without features
#Significant change in R^2 signifies the importance of the feature

from sklearn.metrics import r2_score

linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', data=df)
bechmark = linear_reg.fit()
r2_score(y, bechmark.predict(df))
#0.7406077428649427

#without LSTAT
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B', data=df)
lr_without_lstat = linear_reg.fit()
r2_score(y, lr_without_lstat.predict(df))
#0.6839521119105445 ----- drop in R^2


#withou AGE
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT', data=df)
lr_without_AGE = linear_reg.fit()
r2_score(y, lr_without_AGE.predict(df))
#0.7406060387904339


#without DIS
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + RAD + TAX + PTRATIO + B + LSTAT', data=df)
lr_without_DIS = linear_reg.fit()
r2_score(y, lr_without_DIS.predict(df))
#0.7117535455461315

#without RM
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', data=df)
lr_without_RM = linear_reg.fit()
r2_score(y, lr_without_RM.predict(df))
#0.6969264517537718


#------OLS using Gradient Descent----------------

# instead of using linear algebra to calculate estimates..ML model uses Gradient Descent


# our goal is to minimize the cost(j(theta))

#For cost function j(theta) = (theta)^2

import numpy as np
import pandas as pd
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt




theta = 3
alpha = 0.1
dat = []

for oo in range(0,10):
    res = alpha * 2 * theta #upta rule
    print("{0:.4f} {1:4f}".format(theta, res))
    dat.append([theta, theta**2]) # cost function
    theta = theta - res   #update theta


tmp = pd.DataFrame(dat)
tmp


plt.plot(np.linspace(-2, 4, 100), np.linspace(-2,4, 100)**2) #Theta^2
plt.scatter(tmp.iloc[:,0], tmp.iloc[:,1], marker = 'X')
plt.xlabel("theta")
plt.ylabel("J(Theta)")


#for cost function j(theta) = (theta)^4 + (theta)^2

theta = 3
alpha = 0.01
dat = []

for oo in range(0,10):
    res = alpha * (4 * theta ** 3 + 2 * theta)    #upta rule
    print("{0:.4f} {1:.4f}".format(theta, res))
    dat.append([theta, theta ** 4 + theta ** 2]) # cost function
    theta = theta - res   #update theta


tmp = pd.DataFrame(dat)
tmp

x_grid = np.linspace(-2, 4, 100)
plt.plot(x_grid, x_grid ** 4 + x_grid **2) #Theta^2
plt.scatter(tmp.iloc[:,0], tmp.iloc[:,1], marker = 'X')
plt.xlabel("theta")
plt.ylabel("J(Theta)")



#applying gradient descent to Boston Housing Data-----------------



import numpy as np
import pandas as pd
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.datasets import load_boston

boston_data = load_boston()

df = pd.DataFrame(boston_data.data, columns = boston_data.feature_names)
df.head()
df.shape

X = df[['LSTAT']].values
y = boston_data.target


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y.reshape(-1,1)).flatten()


alpha = 0.0001
w_ = np.zeros(1 + X_std.shape[1])
cost_ = []
n_ = 100

for i in range(n_):
    y_pred = np.dot(X_std, w_[1:]) + w_[0]
    errors = (y_std - y_pred)
    
    w_[1:] += alpha * X_std.T.dot(errors)
    w_[0] += alpha * errors.sum()
    
    cost = (errors**2).sum() / 2.0
    cost_.append(cost)
    


plt.plot(range(1, n_+1), cost_)
plt.xlabel("SSE")
plt.ylabel("Epoch")






#-----------------------Regularized Regrassion-------------

#Linear Regression

import numpy as np
import pandas as pd
#import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

np.random.seed(42)
n_samples = 100
rng = np.random.randn(n_samples) * 10
y_gen = 0.5 * rng + 2 * np.random.randn(n_samples)

lr = LinearRegression()
lr.fit(rng.reshape(-1,1), y_gen)
model_pred = lr.predict(rng.reshape(-1,1))

plt.scatter(rng, y_gen)
plt.plot(rng, model_pred)
print("Coefficients Estimate: ", lr.coef_)
#Coefficients Estimate:  [0.47134857]

#now with an outlier
idx = rng.argmax()
y_gen[idx] = 200
#just adding 200 at 31 index

o_lr = LinearRegression()
o_lr.fit(rng.reshape(-1,1), y_gen)
model_pred = o_lr.predict(rng.reshape(-1,1))

plt.scatter(rng, y_gen)
plt.plot(rng, model_pred)
print("Coefficients Estimate: ", o_lr.coef_)
#Coefficients Estimate:  [0.92796845]....
#should be close to 0.5 as in y_gen = 0.5 * rng




#Ridge Regression with outlier
from sklearn.linear_model import Ridge
ridge_mod = Ridge(alpha=1, normalize=True)
ridge_mod.fit(rng.reshape(-1,1), y_gen)
ridge_model_pred = ridge_mod.predict(rng.reshape(-1,1))

plt.scatter(rng, y_gen)
plt.plot(rng, ridge_model_pred)
print("Coefficients Estimate: ", ridge_mod.coef_)
#Coefficients Estimate:  [0.46398423]
#It ignored the outlier(it might have given it some weight)




#Lasso Regression
from sklearn.linear_model import Lasso
lasso_mod = Lasso(alpha=0.4, normalize=True)
lasso_mod.fit(rng.reshape(-1,1), y_gen)
lasso_mod_pred = lasso_mod.predict(rng.reshape(-1,1))

plt.scatter(rng, y_gen)
plt.plot(rng, lasso_mod_pred)
print("Coefficients Estimate: ", lasso_mod.coef_)
#Coefficients Estimate:  [0.48530263]



#Elastic Net
from sklearn.linear_model import ElasticNet
en_mod = ElasticNet(alpha=0.02, normalize=True)
en_mod.fit(rng.reshape(-1,1), y_gen)
en_mod_pred = en_mod.predict(rng.reshape(-1,1))

plt.scatter(rng, y_gen)
plt.plot(rng, en_mod_pred)
print("Coefficients Estimate: ", en_mod.coef_)
#Coefficients Estimate:  [0.4584509]


#Ridge doent zero out coefficents...it includes all in the model or none
#Lasso does both parameter shrinkage and variable selection automatically
#if some covariates are very high, prefer ElasticNet instead of LASSO







