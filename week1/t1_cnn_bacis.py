import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    s = image.shape
    v = image.reshape(s[0] * s[1] * s[2], 1)
    return v


def run():
    x = np.array([1, 2, 3, 4])
    # print(sigmoid(x))
    # print(sigmoid_derivative(x))
    image = np.array([[[0.67826139, 0.29380381],
                       [0.90714982, 0.52835647],
                       [0.4215251, 0.45017551]],

                      [[0.92814219, 0.96677647],
                       [0.85304703, 0.52351845],
                       [0.19981397, 0.27417313]],

                      [[0.60659855, 0.00533165],
                       [0.10820313, 0.49978937],
                       [0.34144279, 0.94630077]]])

    # print("image2vector(image) = " + str(image2vector(image)))


run()
