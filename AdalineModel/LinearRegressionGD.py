import numpy as np

class LinearRegressionGD:
    def __init__(self, n_iter=50, eta=0.01, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    # def activation(self, X):
    #     '''Вычисление линейной активации'''
    #     return X

    def net_input(self, X):
        '''Вычисление фактического ввода'''
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        '''Возврат метки класса'''
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
        return self.net_input(X)
    
    def fit(self, X, y):
        '''Подгонка к обучающим данным
        ------------------------------
        MSE = (1/n) * Σ( y - output )²
        ∂MSE/∂w = -(2/n) * X^T * (y - output)
        w_new = w_old - η * ∂MSE/∂w = w_old + (2η/n) * X^T * errors
        '''
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64()
        self.losses_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # output = self.activation(net_input)
            output = net_input
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        
        return self