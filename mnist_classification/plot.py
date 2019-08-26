import matplotlib.pyplot as plt


def plot_digit(digit):
    image = digit.reshape(28, 28)
    plt.imshow(image, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.show()
