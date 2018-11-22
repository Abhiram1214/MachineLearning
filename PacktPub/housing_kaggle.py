#Lets import modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("kc_house_data.csv")
data.describe()

df = pd.DataFrame(data)

#check for misssing value
df.isnull().sum()
sns.heatmap(df.isnull())

#check for corelation
cor_mat = df.corr()
cor_mat[np.abs(cor_mat) < 0.6] = 0
sns.heatmap(cor_mat, annot=True)
s