import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%reset -f
#import dataset

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3,4]].values

#using denrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method= 'ward'))
plt.title("Dendrogram")
plt.xlabel("customers")
plt.ylabel("Euclidian dist")
plt.show()

#fitting hierarchical clustering to our mall data set
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
y_hc = hc.fit_predict(x) #it will and creartes a vector of clusters mapping which cx belongs to whcih cluster

#viusalizing the cluster

#visualizing the clusters
plt.scatter(x[y_hc ==0, 0], x[y_hc ==0, 1],s=100, c='red', label = 'careful')
plt.scatter(x[y_hc ==1, 0], x[y_hc==1, 1],s=100, c='blue', label = 'standard')
plt.scatter(x[y_hc ==2, 0], x[y_hc==2, 1],s=100, c='green', label = 'Target')
plt.scatter(x[y_hc ==3, 0], x[y_hc ==3, 1],s=100, c='cyan', label = 'careless')
plt.scatter(x[y_hc ==4, 0], x[y_hc ==4, 1],s=100, c='magenta', label = 'sensible')
plt.title("clusters of clients")
plt.xlabel("Anaual income is $")
plt.ylabel("spending score [1-100]")
plt.legend()
plt.show()

