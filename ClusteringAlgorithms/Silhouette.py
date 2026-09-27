from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples 
from matplotlib import cm
import matplotlib.pyplot as plt  
import numpy as np 

from data import X 


km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-4,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]

silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0
y_ticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), 
            c_silhouette_vals,
            height=1.0,
            edgecolor='none',
            color=color)
    y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_vals_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_vals_avg, 
            color='red',
            linestyle='--')
plt.yticks(y_ticks, cluster_labels + 1)
plt.xlabel('silhouette_coefficient')
plt.ylabel('cluster')
plt.tight_layout()
plt.show()

