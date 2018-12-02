# it is the line that allows for the largest margin between two classes
#svm will place the line in the margin.. it will locate and optmizes the hyperplane in a way it maximizes the distance of the two classes


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

data = datasets.load_iris()

df = pd.DataFrame(data = data['data'], columns = data['feature_names'])
df['species'] = data['target']
df.head()



target_names = {0:'setosa',
                   1:'versicolor',
                   2:'virginica'
                  }

df.species = df.species.map(target_names);

#update columns names 
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

col = ['petal_length', 'petal_width']
X = df.loc[:, col]

y = df['species']

from sklearn import svm
clf = svm.SVC(kernel = 'linear', C=1)
clf.fit(X, y)
