import numpy as np
from week8_rnn.dino_utils import *
import random

def run():
    data = open('dinos.txt', 'r').read()
    data = data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))


run()