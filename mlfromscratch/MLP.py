import numpy as np
import matplotlib.pyplot as plt

from activation import Sigmoid, ReLu, Softmax
from utils import CrossEntropy, to_catrgorical, normalize

class MLP():
    def __init__ (self, n_hidden, n_iters, learning_rate):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()
        self.losses = []
        self.weights = None
    
    def init_weights(self, n_features, n_classes):
        # Initialize weights between -1/sqrt(n_features) and 1/sqrt(n_features)
        limit = 1 / np.sqrt(n_features)
        self.weights = {
            'W1': np.random.uniform(-limit, limit, (n_features, self.n_hidden)),
            'W2': np.random.uniform(-limit, limit, (self.n_hidden, n_classes))
        }
        self.bias = {
            'b1': np.zeros((1, self.n_hidden)),
            'b2': np.zeros((1, n_classes))
        }
    
    def fit(self, X, y):
        self.init_weights(X.shape[1], y.shape[1])
        
        for i in range(self.n_iters):
            # Forward pass
            hidden_input = np.dot(X, self.weights['W1']) + self.bias['b1']
            hidden_output = self.hidden_activation(hidden_input)
            output_layer_input = np.dot(hidden_output, self.weights['W2']) + self.bias['b2']
            y_pred = self.output_activation(output_layer_input)
            
            # Backward pass
            self.losses.append(self.loss(y_pred, y))
            grad_2 = self.loss.gradient(y_pred, y) * self.output_activation.gradient(output_layer_input)
            grad_w2 = np.dot(hidden_output.T, grad_2)
            grad_b2 = np.sum(grad_2, axis=0, keepdims=True)
            
            grad_1 = np.dot(grad_2, self.weights['W2'].T) * self.hidden_activation.gradient(hidden_input)
            grad_w1 = np.dot(X.T, grad_1)
            grad_b1 = np.sum(grad_1, axis=0, keepdims=True)
            
            # Update weights
            self.weights['W1'] -= self.learning_rate * grad_w1
            self.weights['W2'] -= self.learning_rate * grad_w2
            self.bias['b1'] -= self.learning_rate * grad_b1
            self.bias['b2'] -= self.learning_rate * grad_b2
    
    def predict(self, X):
        hidden_input = np.dot(X, self.weights['W1']) + self.bias['b1']
        hidden_output = self.hidden_activation(hidden_input)
        output_layer_input = np.dot(hidden_output, self.weights['W2']) + self.bias['b2']
        y_pred = self.output_activation(output_layer_input)
        return y_pred

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    X = normalize(digits.data)
    y = digits.target
    y = to_catrgorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = MLP(n_hidden=32, n_iters=1000, learning_rate=0.01)
    model.fit(X_train, y_train)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    acc = np.mean(y_pred == y_test)
    print("Accuracy:", acc)
    
    plt.plot(model.losses)
    plt.show()
    
        