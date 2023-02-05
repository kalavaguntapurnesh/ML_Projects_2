#importing the libraries like pandas, numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#reading the data of CSV File..
data_set = pd.read_csv("Dataset_master.xlsx - Amazon.com Clusturing Model (Pr.csv")
X = data_set.iloc[:, [3, 4]].values
#using skicit learn library which consists of K-Means Clustering
from sklearn.cluster import KMeans
wcss = []
#WCSS is called Within Cluster Sum of Squares is the sum of square of distance between a data point and a centroid in a cluster
for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 21)
  #n_clusters means the number of clusters
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
#plotting the data using matplot
plt.plot(range(1, 11), wcss)
plt.title("WCSS via Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
#we are dividing the data into 4 clusters
y_means = kmeans.fit_predict(X)
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, color = 'red', label = 'Cluster-1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, color = 'magenta', label = 'Cluster-2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, color = 'blue', label = 'Cluster-3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, color = 'cyan', label = 'Cluster-4')
plt.title("Amazon User Segmentation")
plt.xlabel("Income of Users")
plt.ylabel("Purchase Rating of Users")
plt.legend()
plt.show()
