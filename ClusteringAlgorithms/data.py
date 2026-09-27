from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_style('whitegrid')


X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)


plt.scatter(X[:, 0], 
            X[:, 1],
            c='red',
            marker='o',
            edgecolor='white',
            s=50)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()