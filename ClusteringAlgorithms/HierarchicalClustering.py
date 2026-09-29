import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette


np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = [f'ID{i}' for i in range(1, 6)]
X = np.random.random_sample([5, 3]) * 10 
df = pd.DataFrame(X, columns=variables, index=labels)

row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
# print(row_dist)

row_clusters = linkage(df.values, 
                       method='complete',
                       metric='euclidean')

# _df = pd.DataFrame(row_clusters,
#                    columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
#                    index = [f'cluster {(i + 1)}' for i in range(row_clusters.shape[0])])

# set_link_color_palette(['black'])
# row_dendr = dendrogram(row_clusters, 
#                        labels=labels)
#                     #    ,color_threshold=np.inf)

# plt.tight_layout()
# plt.ylabel('euclidean distance')
# plt.show()
