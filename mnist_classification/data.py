from sklearn.datasets import fetch_openml
import numpy as np
np.random.seed(42)


def load_data():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8)
    # sort_by_target(mnist)
    x, y = mnist['data'], mnist['target']
    x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
    # shuffle_index = np.random.permutation(60000)
    # x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
    return (x_train, y_train), (x_test, y_test)


# def sort_by_target(mnist):
#     reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
#     reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
#     mnist.data[:60000] = mnist.data[reorder_train]
#     mnist.target[:60000] = mnist.target[reorder_train]
#     mnist.data[60000:] = mnist.data[reorder_test + 60000]
#     mnist.target[60000:] = mnist.target[reorder_test + 60000]
