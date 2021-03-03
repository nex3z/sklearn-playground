import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def load_linear_data(seed=42):
    np.random.seed(seed)
    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)
    return x, y


def plot_xy(x, y):
    plt.scatter(x, y)
    plt.xlabel("$x_1$")
    plt.ylabel("$y$")


def plot_models(x, y, models, axis=None):
    if axis is None:
        axis = [-3, 3, 0, 10]
    plot_xy(x, y)
    x_new = np.linspace(axis[0], axis[1], 100).reshape(100, 1)
    for name, model in models:
        model.fit(x, y)
        y_new = model.predict(x_new)
        plt.plot(x_new, y_new, label=name)
    plt.axis(axis)
    plt.legend()


def plot_learning_curves(x, y, model):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(x_train)):
        model.fit(x_train[:m], y_train[:m])
        y_train_pred = model.predict(x_train[:m])
        y_val_pred = model.predict(x_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_pred))
        val_errors.append(mean_squared_error(y_val, y_val_pred))

    plt.plot(np.sqrt(train_errors), label="train")
    plt.plot(np.sqrt(val_errors), label="val")
    plt.legend()
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
