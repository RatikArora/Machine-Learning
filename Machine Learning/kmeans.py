import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd

# Load the dataset
X = pd.read_csv('data/driver_data.csv')

# Extract features
x1 = X['Distance_Feature'].values
x2 = X['Speeding_Feature'].values

# Convert data to numpy array
X = np.array(list(zip(x1, x2)))

# Plot dataset
plt.figure(figsize=(8, 6))
plt.scatter(x1, x2, c='blue', label='Data points')
plt.xlabel('Distance Feature')
plt.ylabel('Speeding Feature')
plt.title('Dataset')
plt.legend()
plt.grid(True)
plt.show()

# KMeans clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Plot KMeans clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow', label='Clusters')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x', label='Centroids')
plt.xlabel('Distance Feature')
plt.ylabel('Speeding Feature')
plt.title('KMeans Clustering')
plt.legend()
plt.grid(True)
plt.show()

# Gaussian Mixture Model (EM)
gmm = GaussianMixture(n_components=2)
gmm.fit(X)
em_predictions = gmm.predict(X)

# Plot EM clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=em_predictions, cmap='viridis', label='Clusters')
plt.xlabel('Distance Feature')
plt.ylabel('Speeding Feature')
plt.title('Expectation-Maximization Clustering')
plt.legend()
plt.grid(True)
plt.show()
