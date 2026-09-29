from sklearn.datasets import make_moons 
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)
#check dataset
# plt.scatter(X[:, 0], X[:, 1])
# plt.xlabel('feature 1')
# plt.ylabel('feature 2')
# plt.tight_layout()
# plt.show()

km = KMeans(n_clusters=2,
            random_state=0)
y_km = km.fit_predict(X)

f, ax = plt.subplots(1, 2)

ax[0].scatter(X[y_km == 0, 0],
              X[y_km == 0, 1],
              color='blue',
              s=40,
              marker='o',
              label='cluster 1')

ax[0].scatter(X[y_km == 1, 0],
              X[y_km == 1, 1],
              color='orange',
              s=40,
              marker='s',
              label='cluster 2')

ax[0].set_title('clusterization on kmeans')
ax[0].set_xlabel('feature 1')
ax[0].set_ylabel('feature 2')

ac = AgglomerativeClustering(n_clusters=2,
                            metric='euclidean',
                            linkage='complete')
y_ac = ac.fit_predict(X)

ax[1].scatter(X[y_ac == 0, 0],
              X[y_ac == 0, 1],
              color='blue',
              s=40,
              marker='o',
              label='cluster 1')

ax[1].scatter(X[y_ac == 1, 0],
              X[y_ac == 1, 1],
              color='orange',
              s=40,
              marker='s',
              label='cluster 2')

ax[1].set_title('clusterization on kmeans')
ax[1].set_xlabel('feature 1')
ax[1].set_ylabel('feature 2')

ax[1].set_title('agglomerative clusterization')
ax[1].set_xlabel('feature 1')
ax[1].set_ylabel('feature 2')

plt.legend(loc='best')
plt.tight_layout()
plt.show()