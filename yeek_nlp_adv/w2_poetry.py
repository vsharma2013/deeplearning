
# https://deeplearningcourses.com/c/deep-learning-advanced-nlp
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import os
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU

# some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 3000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10
LATENT_DIM = 25

# load in the data
input_texts = []
target_texts = []

def run():
# load in the data
	for line in open('yeek_nlp_adv/data/poetry/robert_frost.txt'):
		line = line.rstrip()
		if not line:
			continue

# convert the sentences (strings) into integers
		input_text = '<sos> ' + line
		target_text = line + ' <eos>'

		input_texts.append(input_text)
		target_texts.append(target_text)

	all_lines = input_texts + target_texts

	tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
	tokenizer.fit_on_texts(all_lines)
	input_sequences = tokenizer.texts_to_sequences(input_texts)
	target_sequences = tokenizer.texts_to_sequences(target_texts)

	max_sequence_length_from_data = max(len(s) for s in input_sequences)
	print('Max sequence length:', max_sequence_length_from_data)

	# get word -> integer mapping
	word2idx = tokenizer.word_index
	print('Found %s unique tokens.' % len(word2idx))
	assert('<sos>' in word2idx)
	assert('<eos>' in word2idx)

	max_sequence_length = min(max_sequence_length_from_data, MAX_SEQUENCE_LENGTH)
	input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
	target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')
	print('Shape of data tensor:', input_sequences.shape)

	print('Loading word vectors...')
	word2vec={}

	with open(os.path.join('yeek_nlp_adv/data/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
		for line in f:
			values = line.split()
			word = values[0]
			vec = np.asarray(values[1:], dtype='float32')
			word2vec[word] = vec

	print('Found %s word vectors.' % len(word2vec))

	num_words = min(len(word2vec) + 1, MAX_VOCAB_SIZE)
	embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

	for word, i in word2idx.items():
		if i < MAX_VOCAB_SIZE:
			embedding_vector = word2vec.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all zeros.
				embedding_matrix[i] = embedding_vector



	one_hot_targets = np.zeros((len(input_sequences), max_sequence_length, num_words))
	for i, target_sequence in enumerate(target_sequences):
		for t, word in enumerate(target_sequence):
			if word > 0:
				one_hot_targets[i, t, word] = 1

	embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix])

	input_ = Input(shape=(max_sequence_length,))
	input_h = Input(shape=(LATENT_DIM,))
	input_c = Input(shape=(LATENT_DIM,))

	x = embedding_layer(input_)
	lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
	x, _, _ = lstm(x, initial_state=[input_h, input_c])

	dense = Dense(num_words, activation="softmax")
	output = dense(x)

	model = Model([input_, input_h, input_c], output)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
	z = np.zeros((len(input_sequences), LATENT_DIM))
	r = model.fit([input_sequences, z, z], one_hot_targets, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)

	# plot some data
	plt.plot(r.history['loss'], label='loss')
	plt.plot(r.history['val_loss'], label='val_loss')
	plt.legend()
	plt.show(block=True)

	# accuracies
	plt.plot(r.history['acc'], label='acc')
	plt.plot(r.history['val_acc'], label='val_acc')
	plt.legend()
	plt.show(block=True)

	
	# make a sampling model
	input2 = Input(shape=(1,)) # we'll only input one word at a time
	x = embedding_layer(input2)
	x, h, c = lstm(x, initial_state=[initial_h, initial_c]) # now we need states to feed back in
	output2 = dense(x)
	sampling_model = Model([input2, initial_h, initial_c], [output2, h, c])


	# reverse word2idx dictionary to get back words
	# during prediction
	idx2word = {v:k for k, v in word2idx.items()}


	def sample_line():
		# initial inputs
		np_input = np.array([[ word2idx['<sos>'] ]])
		h = np.zeros((1, LATENT_DIM))
		c = np.zeros((1, LATENT_DIM))

		# so we know when to quit
		eos = word2idx['<eos>']

		# store the output here
		output_sentence = []

		for _ in range(max_sequence_length):
			o, h, c = sampling_model.predict([np_input, h, c])

			# print("o.shape:", o.shape, o[0,0,:10])
			# idx = np.argmax(o[0,0])
			probs = o[0,0]
			if np.argmax(probs) == 0:
			print("wtf")
			probs[0] = 0
			probs /= probs.sum()
			idx = np.random.choice(len(probs), p=probs)
			if idx == eos:
			break

		# accuulate output
		output_sentence.append(idx2word.get(idx, '<WTF %s>' % idx))

		# make the next input into model
		np_input[0,0] = idx

		return ' '.join(output_sentence)

	# generate a 4 line poem
	while True:
	for _ in range(4):
	print(sample_line())

	ans = input("---generate another? [Y/n]---")
	if ans and ans[0].lower().startswith('n'):
	break