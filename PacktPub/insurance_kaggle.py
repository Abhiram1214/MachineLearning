
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import collections

insurance = pd.read_csv("insurance.csv")
insurance.describe()

insurance.isnull().sum()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

insurance.sex = encoder.fit_transform(insurance.sex)
insurance.smoker = encoder.fit_transform(insurance.smoker)
insurance.region = encoder.fit_transform(insurance.region)


sns.heatmap(insurance.corr(), annot=True)
#coreelation is observed with smoking


#visualization
#for smokers
sns.distplot(insurance[(insurance.smoker == 1)]["charges"],color='c')
#for non smokers 
sns.distplot(insurance[(insurance.smoker == 0)]['charges'], color = 'b')

#suprisingly non smokers spend more on insurance then smokers do
insurance[(insurance.smoker == 0)]['bmi'].sum()
#smokers = 8781763.521839999
#non smokers = 8974061.468919002
sns.distplot(insurance.age, color = 'g')

sns.catplot(x="smoker", kind="count",hue = 'sex', palette="rainbow", data=insurance[(insurance.age==18)])
sns.lmplot(x="age", y="charges", hue="smoker", data=insurance, palette = 'inferno_r', size = 7)


#with BMI
sns.distplot(insurance.bmi)
sns.distplot(insurance[(insurance.bmi > 30)]['charges'])
sns.distplot(insurance[(insurance.bmi < 30)]['charges'])
# people with BMI > 30 spend more money on treattment 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error

x = insurance.drop(['charges'], axis = 1)
y = insurance.charges

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state = 0)

regression = LinearRegression()
regression.fit(x_train, y_train)

y_train_pred = regression.predict(x_train)
y_test_pred = regression.predict(x_test)

print(regression.score(x_test,y_test))
#0.7998747145449959


# polynomial

quad_poly = PolynomialFeatures(degree=3)

x_quad = quad_poly.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_quad, y, test_size=0.2, random_state = 0)

ln = LinearRegression()
ln.fit(x_train, y_train)

y_train_poly_pred = ln.predict(x_train)
y_test_poly_pred = ln.predict(x_test)

print(ln.score(x_test,y_test))
#with poly 2 = 0.8648637939194134
# with poly 3 = 0.8740116040081882








