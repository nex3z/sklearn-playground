import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml


def plot_digit(digit, label=None):
    image = digit.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary)
    if label is not None:
        plt.title(label)
    plt.axis('off')


def plot_digits(digits, num_cols=2):
    num_rows = int(np.ceil(len(digits) / num_cols))
    rows = []
    for r in range(num_rows):
        row = np.concatenate([digits[r * num_cols + c] for c in range(num_cols)],  axis=1)
        rows.append(row)
    rows = np.concatenate(rows)
    plt.imshow(rows, cmap='binary')
    plt.axis('off')


def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    images, labels = mnist['data'], mnist['target']
    x_train, y_train = images[:60000], labels[:60000]
    x_test, y_test = images[60000:], labels[60000:]
    return (x_train, y_train), (x_test, y_test)


def load_mnist_5():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    images, labels = mnist['data'], mnist['target']
    x_train, x_test, y_train, y_test = images[:60000], images[60000:], labels[:60000], labels[60000:]
    y_train_5, y_test_5 = (y_train == '5'), (y_test == '5')
    return (x_train, y_train_5), (x_test, y_test_5)
