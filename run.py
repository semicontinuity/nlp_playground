#!/usr/bin/env python
from gensim.models import Word2Vec

import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

# Check Python Version
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

dimVectors = 1024
sparseness = 0.03

# Context size
C = 5
SPARSENESS = 0.03
LAMBDA = 0.05
GAMMA = 0.05

startTime=time.time()
wordVectors = np.random.rand(nWords, dimVectors) - (1 - SPARSENESS)


# print(tokens)
rejectProb = dataset.rejectProb()
print(len(rejectProb))
print(len(tokens))

print(rejectProb[:20])
print(dataset._revtokens)
print(dataset.tokens())
# print(dataset.allSentences()[11123])
# print(tokens[1])
# print(tokens[2])