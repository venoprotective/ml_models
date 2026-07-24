from kNN import kNN
from sklearn import tree
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).parent.parent))
from irisDataSet import *
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score

kModel = kNN()
kModel.fit(X_train_std, y_train)

y_pred = kModel.predict(X_combined_std)

plot_decision_regions(X_combined_std, y_combined, classifier=kModel, test_idx=range(105,150))
plt.show()
print(accuracy_score(y_combined, y_pred))