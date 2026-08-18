import pyprind 
from preprocess import *
from sklearn.feature_extraction.text import HashingVectorizer 
from sklearn.linear_model import SGDClassifier
import pandas as pd 
import numpy as np 


vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log_loss', random_state=1)
doc_stream = stream_docs(path='data.csv')

pbar = pyprind.ProgBar(20)
classes = np.array([0, 1])
for _ in range(20):
    X_train, y_train = get_minibatch_size(doc_stream, size=1000)
    if not X_train: break 
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()
    
# print(pd.read_csv('test.csv').shape)
# print(pd.read_csv('train.csv').shape)

X_test, y_test = get_minibatch_size(doc_stream, 5000)
X_test = vect.transform(X_test)

print(f'accuracy {clf.score(X_test, y_test):.3f}')

