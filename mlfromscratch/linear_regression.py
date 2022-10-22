import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import l1_regularization, l2_regularization

class LinearRegression():
    def __init__(self, n_iters=1000, learning_rate=0.01):
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.weights = None
        self.train_errors = []
    
    def init_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        self.weights = np.random.uniform(-limit, limit, (n_features, ))
                
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.init_weights(X.shape[1])
        
        for _ in tqdm(range(self.n_iters)):
            y_pred = X.dot(self.weights)
            self.train_errors.append(mean_squared_error(y, y_pred))
            grad_w = np.mean((y_pred - y) * X.T, axis=1)
            self.weights -= self.learning_rate * grad_w
    
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.weights)

class LassoRegression():
    def __init__(self, n_iters=1000, learning_rate=0.01):
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.regularization = l1_regularization(alpha=0.01)
        self.weights = None
        self.train_errors = []
    
    def init_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        self.weights = np.random.uniform(-limit, limit, (n_features, ))
                
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.init_weights(X.shape[1])
        
        for _ in tqdm(range(self.n_iters)):
            y_pred = X.dot(self.weights)
            self.train_errors.append(mean_squared_error(y, y_pred) + self.regularization(self.weights))
            grad_w = np.mean((y_pred - y) * X.T, axis=1) + self.regularization.grad(self.weights)
            self.weights -= self.learning_rate * grad_w
    
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.weights)

if __name__ == "__main__":
    X, y = make_regression(n_samples=1000, n_features=5, noise=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))
    
    plt.plot(model.train_errors)
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.show()