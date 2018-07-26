import  math
import  numpy  as  np
import  h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from week4.planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y


def initialize_parameters_deep(layer_dims):
    L = len(layer_dims)
    parameters = {}

    for l in range(1, L):
        w_l = "W" + str(l)
        b_l = "b" + str(l)
        parameters[w_l] = tf.get_variable(w_l, [layer_dims[l], layer_dims[l-1]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters[b_l] = tf.get_variable(b_l, [layer_dims[l], 1], initializer=tf.zeros_initializer())

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    ZL -- the output of the last LINEAR unit
    """
    L = len(parameters.keys()) // 2
    A_prev = X

    for l in range(1, L):
        w = "W" + str(l)
        b = "b" + str(l)
        Z = tf.add(tf.matmul(parameters[w], A_prev), parameters[b])
        A = tf.tanh(Z)
        A_prev = A

    w = "W" + str(L)
    b = "b" + str(L)
    ZL = tf.add(tf.matmul(parameters[w], A_prev), parameters[b])
    AL = tf.sigmoid(ZL)

    return AL

def run():
    layer_dims = [2, 20, 7, 5, 1]

    X_train, Y_train = load_planar_dataset()

    print(X_train.shape)
    print(Y_train.shape)

    # return 0

    X, Y = create_placeholders(X_train.shape[0], Y_train.shape[0])

    parameters = initialize_parameters_deep(layer_dims)

    A = forward_propagation(X, parameters)

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        res = session.run(A, feed_dict={X: X_train, Y: Y_train})
        print(str(res))

    print(" I from TF DNN")