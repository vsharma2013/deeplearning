import  math
import  numpy  as  np
import  h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from week4.planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets
from week4.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict


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


def compute_cost(AL, Y):
    logits = AL
    labels = Y

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    return cost


def model(X_train, Y_train, layer_dims, learning_rate=0.0001, num_iterations=2500, print_cost=False):
    ops.reset_default_graph()

    X, Y = create_placeholders(X_train.shape[0], Y_train.shape[0])

    parameters = initialize_parameters_deep(layer_dims)

    A = forward_propagation(X, parameters)

    cost_tensor = compute_cost(A, Y_train)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_tensor)

    init = tf.global_variables_initializer()

    costs = []

    with tf.Session() as session:
        session.run(init)
        for i in range(num_iterations):
            results = session.run([optimizer, cost_tensor], feed_dict={X: X_train, Y: Y_train})
            cost = results[1]
            costs.append(cost)
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)

        predictions = tf.round(A)
        predictions = predictions.eval({X: X_train, Y: Y_train})
        print("Predictions = ", predictions)
        hit = 0
        for i in range(Y_train.shape[0]):
            for j in range(Y_train.shape[1]):
                if predictions[i, j] == Y_train[i, j]:
                    hit += 1

        print("Accuracy = ", (hit/Y_train.size)*100)



    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show(block=True)


def run_plannar():
    X_train, Y_train = load_planar_dataset()
    layer_dims = [2, 4, 1]
    # layer_dims = [2, 5, 1]

    model(X_train, Y_train, layer_dims, learning_rate=0.00001, num_iterations=10000, print_cost=True)


def run_cat():
    layer_dims = [12288, 4, 1]

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten / 255.
    X_test = X_test_flatten / 255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)

    print("number of training examples = " + str(X_train.shape[1]))
    print("number of test examples = " + str(X_test.shape[1]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    # model(X_train, Y_train,layer_dims)


def run():
    # run_plannar()
    run_cat()