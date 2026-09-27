from data import X 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
import seaborn as sns 


sns.set_style('whitegrid')

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
    
plt.plot(range(1,11), distortions, marker='o', color='blue')
plt.xlabel('amount clusters')
plt.ylabel('distortion')
plt.xticks(range(1,11))
plt.tight_layout()    
plt.show()