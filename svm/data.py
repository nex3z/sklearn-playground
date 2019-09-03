import numpy as np
from sklearn import datasets

np.random.seed(42)


def load_data():
    iris = datasets.load_iris()
    x = iris['data'][:, (2, 3)]
    y = iris['target']
    return x, y


def load_setosa_vs_versicolor():
    x, y = load_data()
    select = (y == 0) | (y == 1)
    return x[select], y[select]


def load_versicolor_vs_virginica():
    x, y = load_data()
    select = (y == 1) | (y == 2)
    return x[select], y[select]
