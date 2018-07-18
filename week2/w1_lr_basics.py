import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

def sigmoid(x):
    return 1/(1 + np.exp(-x))
    # return np.tanh(x) + 0.0000000000000000001
    # return np.max(0.0, x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def load_dataset():
    train_dataset = h5py.File('/home/vishal/Downloads/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('/home/vishal/Downloads/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1));
    b = 0
    return w, b

def propagate(w, b, X, Y):
    m = Y.shape[1]
    A = sigmoid(np.dot(w.T,  X) + b)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis=1, keepdims=True)/m
    db = (1/m) * (np.sum(A-Y, axis=1, keepdims=True))
    dw = (1/m) * (np.dot(X, (A-Y).T))
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = [];
    for i in range (num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w, b, X):
    Y_prediction = sigmoid(np.dot(w.T, X) + b)
    Y_prediction = Y_prediction >= 0.5
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros((X_train.shape[0]))
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

def run():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    m_train = train_set_y.shape[1]
    m_test = test_set_y.shape[1]
    num_px = train_set_x_orig.shape[1]
    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_set_x shape: " + str(train_set_x_orig.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x shape: " + str(test_set_x_orig.shape))
    print("test_set_y shape: " + str(test_set_y.shape))

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
    print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
              print_cost=True)

    print("done...")

run()