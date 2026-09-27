from data import X
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set_style('whitegrid')

km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-4,
            random_state=0)

y_km = km.fit_predict(X) 

plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='green',
            marker='s', edgecolor='black',
            label='cluster 1')

plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='yellow',
            marker='o', edgecolor='black',
            label='cluster 2')

plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='blue',
            marker='^', edgecolor='black',
            label='cluster 3')

plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=350, color='red',
            marker='*', edgecolor='black', 
            label='centroids'
            )

plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend(scatterpoints=1, loc='best')
plt.tight_layout()
plt.show()