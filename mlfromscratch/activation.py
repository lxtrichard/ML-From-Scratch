import numpy as np

class Sigmoid():
    def __call__(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient(self, z):
        return self.__call__(z) * (1 - self.__call__(z))