import numpy as np 
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    '''
        n_estimators : int 
        amount estimators 
        
        learning_rate : flaot
        speed(len step model scale)
        
        random_state : int
        random_state 
        
    '''
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = 0
        self.weights = []
        self.models = [] # stumps

    def normalize(self, weights):
        return weights / sum(weights)
    
    def update_weights(self, weights, y, tree_predictions, a_j_coeff):    
         return weights[y != tree_predictions] * np.exp(a_j_coeff)
            
    def fit(self, X, y):
        n_samples = np.unique(y)
        weights = np.ones(n_samples) / n_samples
        self.amount_classes = len(np.unique(y))
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=3, random_state=self.random_state)
            tree = tree.fit(X, y, sample_weight=weights)
            tree_predictions = tree.predict(X)
            epsilon = sum(self.weights[tree_predictions != y])
            a_j_coeff = self.learning_rate * np.log((1 - epsilon) / epsilon)
            
            self.models.append(tree)
            self.weights.append(a_j_coeff)
            weights = self.update_weights(weights, y, tree_predictions, a_j_coeff)
            weights = self.normalize(weights)
    
    def predict(self, X):
        return self.majority_weighted_votes(X)
    
    def majority_weighted_votes(self, X):
        votes = np.zeros((len(X), self.amount_classes)) 
        for j in range(self.n_estimators):
            aj = self.weights[j]
            Cj = self.models[j]
            predictions = Cj.predict(X)
            votes[np.array(range(len(X))),predictions] += aj
    
        return np.argmax(votes, axis=1)

          
        