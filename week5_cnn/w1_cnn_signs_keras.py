import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from week5_cnn.cnn_utils import *
import math
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops

def SignsModel(input_shape):
	X_input = Input(input_shape)

	# Zero-Padding: pads the border of X_input with zeroes
	X = ZeroPadding2D((3, 3))(X_input)

	# layer group1 32*32*32
	# CONV -> BN -> RELU Block applied to X
	X = Conv2D(32, (7, 7), strides=(1, 1), name='conv1')(X)
	X = BatchNormalization(axis=3, name='bn1')(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((2, 2), name='max_pool1')(X)

	# layer group2 16*16*64
	X = ZeroPadding2D((2, 2))(X)
	# CONV -> BN -> RELU Block applied to X
	X = Conv2D(64, (5, 5), strides=(1, 1), name='conv2')(X)
	X = BatchNormalization(axis=3, name='bn2')(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((2, 2), name='max_pool2')(X)

	# layer group3 8*8*128
	X = ZeroPadding2D((1, 1))(X)
	# CONV -> BN -> RELU Block applied to X
	X = Conv2D(128, (3, 3), strides=(1, 1), name='conv3')(X)
	X = BatchNormalization(axis=3, name='bn3')(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((2, 2), name='max_pool3')(X)

	# layer group4 8*8*64
	# CONV -> BN -> RELU Block applied to X
	X = Conv2D(64, (1, 1), strides=(1, 1), name='conv4')(X)
	X = BatchNormalization(axis=3, name='bn4')(X)
	X = Activation('relu')(X)

	# layer group5 4*4*32
	X = ZeroPadding2D((1, 1))(X)
	# CONV -> BN -> RELU Block applied to X
	X = Conv2D(32, (3, 3), strides=(1, 1), name='conv5')(X)
	X = BatchNormalization(axis=3, name='bn5')(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((2, 2), name='max_pool5')(X)

	# FLATTEN X (means convert it to a vector) + FULLYCONNECTED
	X = Flatten()(X)
	X = Dense(128, activation='sigmoid', name='fc1')(X)
	X = Dense(32, activation='sigmoid', name='fc2')(X)
	X = Dense(6, activation='sigmoid', name='fc3')(X)

	# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
	model = Model(inputs=X_input, outputs=X, name='SignsModel')

	### END CODE HERE ###

	return model



def run():
	X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
	X_train = X_train_orig / 255.
	X_test = X_test_orig / 255.
	Y_train = convert_to_one_hot(Y_train_orig, 6).T
	Y_test = convert_to_one_hot(Y_test_orig, 6).T
	print("number of training examples = " + str(X_train.shape[0]))
	print("number of test examples = " + str(X_test.shape[0]))
	print("X_train shape: " + str(X_train.shape))
	print("Y_train shape: " + str(Y_train.shape))
	print("X_test shape: " + str(X_test.shape))
	print("Y_test shape: " + str(Y_test.shape))

	happyModel = SignsModel((64, 64, 3))

	happyModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

	happyModel.fit(x=X_train, y=Y_train, epochs=20, batch_size=16)

	preds = happyModel.evaluate(x=X_test, y=Y_test)

	# ## END CODE HERE ###
	print()
	print("Loss = " + str(preds[0]))
	print("Test Accuracy = " + str(preds[1]))






