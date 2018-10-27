
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("crime_data.csv")
x = dataset.iloc[:, [1,4]].values

#use elbow techniques to identify the number of required clusters
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10,
                    max_iter=300, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.show()

#number of clusters = 3
kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10,
                    max_iter=300, random_state=0)
y_clust = kmeans.fit_predict(x)