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
        A = tf.nn.relu(Z)
        A_prev = A

    w = "W" + str(L)
    b = "b" + str(L)
    ZL = tf.add(tf.matmul(parameters[w], A_prev), parameters[b])
    return ZL


def compute_cost(ZL, Y):
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    return cost



def model(X_train, Y_train, X_test, Y_test, layer_dims, learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=False):
    ops.reset_default_graph()

    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters_deep(layer_dims)

    Z = forward_propagation(X, parameters)

    cost = compute_cost(Z, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

                # Print the cost every epoch
                if print_cost == True and epoch % 100 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("\n\n\nTrain Accuracy:", accuracy.eval({X: X_train, Y: Y_train}) * 100, "%")
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}) * 100, "%")

        plt.show(block=True)

        return parameters

def run():
    print('m workshop')

    layer_dims = [12288, 25, 12, 6]

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

    plt.imshow(X_train_orig[2])
    plt.show(block=True)
