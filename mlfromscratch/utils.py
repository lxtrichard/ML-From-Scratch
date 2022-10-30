import numpy as np

class l1_regularization():
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, ord=1)

    def grad(self, w):
        return self.alpha * np.sign(w)

class l2_regularization():
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * 0.5 * np.linalg.norm(w)**2

    def grad(self, w):
        return self.alpha * w

class CrossEntropy():
    def __call__(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)

def to_catrgorical(y, n_dims=None):
    y = y.astype(int)
    n_samples = y.shape[0]
    n_dims = np.max(y) + 1 if n_dims is None else n_dims
    one_hot = np.zeros((n_samples, n_dims))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def cal_entropy(y):
    """ Calculate the entropy of a list of examples. """
    entropy = 0
    labels = np.unique(y)
    for label in labels:
        p = y[y==label].shape[0] / y.shape[0]
        entropy += -p * np.log2(p)
    return entropy

def calculate_covariance_matrix(X, Y=None):
    """ Calculate the covariance matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)