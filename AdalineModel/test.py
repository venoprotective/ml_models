from AdaptiveLinearNeuron import AdalineGD 
import sys
from pathlib import Path  
sys.path.append(str(Path(__file__).parent.parent))
from plot_decision_regions import *
from irisDataSet import *
import matplotlib.pyplot as plt 
from LinearRegressionGD import LinearRegressionGD
import os 


columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 
           'Central Air', 'Total Bsmt SF', 'SalePrice']

import kagglehub

# Download latest version
path = kagglehub.dataset_download("shashanknecrothapa/ames-housing-dataset")

print("Path to dataset files:", path)
path = os.path.join(path, os.listdir(path)[0])
df = pd.read_csv(path, usecols=columns)
df['Central Air'] = df['Central Air'].map({'Y' : 1, 'N' : 0})
# print(df)
# print(df)
# print(df.isnull().sum()) 
# Overall Qual     0
# Overall Cond     0
# Total Bsmt SF    1
# Central Air      0
# Gr Liv Area      0
# SalePrice        0
df = df.dropna(axis=0)
# print(df.isnull().sum())
# Overall Qual     0
# Overall Cond     0
# Total Bsmt SF    0
# Central Air      0
# Gr Liv Area      0
# SalePrice        0

import matplotlib.pyplot as plt 
# from mlxtend.plotting import scatterplotmatrix # pip install mlxtend

# scatterplotmatrix(df.values, 
#                   names=df.columns)
# plt.tight_layout()
# plt.show()
# from mlxtend.plotting import heatmap 
import numpy as np 
# cm = np.corrcoef(df.values.T)
# # print(cm)
# hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
# plt.tight_layout()
# plt.show()

X = df[['Gr Liv Area']].values
y = df['SalePrice'].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD(eta=0.1)
lr.fit(X_std, y_std) 

# plt.plot(range(1, lr.n_iter + 1), lr.losses_)
# plt.ylabel('mse')
# plt.xlabel('epochs')
# plt.show()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    
# lin_regplot(X_std, y_std, lr)
# plt.xlabel('gr liv area')
# plt.ylabel('saleprice')
# plt.show()

# feature_std = sc_x.transform(np.array([[2500]]))
# f_predict = lr.predict(feature_std)
# target_reverted = sc_y.inverse_transform(f_predict.reshape(-1, 1))
# print(target_reverted.flatten())

from sklearn.linear_model import LinearRegression 

slr = LinearRegression()
slr.fit(X, y)
lin_regplot(X, y, slr)
plt.show()
# test adaline with gradient descent 
# comparison of learning speed

# ada0 = AdalineGD(n_iter=15, eta=0.1)
# ada0.fit(X, y)

# ada1 = AdalineGD(n_iter=15, eta=0.0001)
# ada1.fit(X, y)

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
# ax[0].plot(range(1, len(ada0.losses_) + 1), np.log10(ada0.losses_) ,marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('log mse')
# ax[0].set_title('adaline eta=0.1')
# ax[1].plot(range(1, len(ada1.losses_) + 1), ada1.losses_ ,marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('mse')
# ax[1].set_title('adaline eta=0.0001')
# plt.show()


# testing after standardization 
# ada = AdalineGD(n_iter=20, eta=0.5)
# ada.fit(X_std_adaline, y_perceptron)
# plot_decision_regions(X_std_adaline, y_perceptron, classifier=ada)
# plt.title('Adaline with GD')
# plt.xlabel('длина чашел.')
# plt.ylabel('длина лепестка')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()
# # measurement MSE 
# plt.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.tight_layout()
# plt.show()