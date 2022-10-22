import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from activation import Sigmoid


class LogisticRegression():
    def __init__(self, learning_rate=.1, gradient_descent=True):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        limit = 1 / np.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for i in tqdm(range(n_iterations)):
            # Make a new prediction
            h = self.sigmoid(X.dot(self.param))
            grad_2 =  (h - y).dot(X)
            self.param -= self.learning_rate * grad_2

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred
    

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data
    y = digits.target
    y[y < 5] = 0
    y[y >= 5] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Accuracy:", np.mean(y_pred == y_test))
    
    # import logistic_regression
    from sklearn.linear_model import LogisticRegression
    model_1 = LogisticRegression()
    model_1.fit(X_train, y_train)
    
    y_pred = model_1.predict(X_test)
    print("Accuracy:", np.mean(y_pred == y_test))
    