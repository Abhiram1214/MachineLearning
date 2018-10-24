#simple linear regresion
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#import dataset


dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# splitting the training set with a test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state=0)


#Linear regression willtake care of feature scaling

#Fitting Simple Linear regression on the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(x_test)

#Visualizing the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
'''


#simple linear regresion

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#import dataset

def initialize_dataset():
    global x
    global y
    global dataset
    
    dataset = pd.read_csv("Salary_Data.csv")
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    

# splitting the training set with a test set
def intialize_train_and_test_sets():
    global x_train
    global x_test
    global y_train
    global y_test
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state=0)


#Linear regression willtake care of feature scaling

#Fitting Simple Linear regression on the training set

def regression_technique():
    global regressor
    global y_pred
    
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    
    #Predicting the test set results
    y_pred = regressor.predict(x_test)


#Visualizing the training set results
def visualization():
    plt.scatter(x_train, y_train, color = 'red')
    plt.plot(x_train, regressor.predict(x_train), color = 'blue')
    plt.title('Salary vs Experience (training set')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()
    
    #Visualizing the test set results
    plt.scatter(x_test, y_test, color = 'red')
    plt.plot(x_train, regressor.predict(x_train), color = 'blue')
    plt.title('Salary vs Experience (test set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()


initialize_dataset()
intialize_train_and_test_sets()
regression_technique()
visualization()

