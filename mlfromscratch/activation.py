import numpy as np

class Sigmoid():
    def __call__(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient(self, z):
        return self.__call__(z) * (1 - self.__call__(z))

class ReLu():
    def __call__(self, z):
        return np.maximum(0, z)

    def gradient(self, z):
        return np.where(z > 0, 1, 0)

class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)