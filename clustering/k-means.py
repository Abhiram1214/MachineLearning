
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib qt
#%reset -f
#import dataset

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3,4]].values


#using elbow method to determine the optimum number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("the elbow method")
plt.xlabel("number of cluster")
plt.ylabel("wcss")
plt.show()

#apply k-means algo with cluster 5
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300,random_state=0)
y_kmeans = kmeans.fit_predict(x)


#visualizing the clusters
plt.scatter(x[y_kmeans==0, 0], x[y_kmeans==0, 1],s=100, c='red', label = 'careful')
plt.scatter(x[y_kmeans==1, 0], x[y_kmeans==1, 1],s=100, c='blue', label = 'standard')
plt.scatter(x[y_kmeans==2, 0], x[y_kmeans==2, 1],s=100, c='green', label = 'Target')
plt.scatter(x[y_kmeans==3, 0], x[y_kmeans==3, 1],s=100, c='cyan', label = 'careless')
plt.scatter(x[y_kmeans==4, 0], x[y_kmeans==4, 1],s=100, c='magenta', label = 'sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='centroids')
plt.title("clusters of clients")
plt.xlabel("Anaual income is $")
plt.ylabel("spending score [1-100]")
plt.legend()
plt.show()

