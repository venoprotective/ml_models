import pandas as pd 
import numpy as np 


np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = [f'ID{i}' for i in range(1, 6)]
X = np.random.random_sample([5, 3]) * 10 
df = pd.DataFrame(X, columns=variables, index=labels)

