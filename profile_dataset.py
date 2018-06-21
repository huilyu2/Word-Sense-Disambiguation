#!/usr/bin/python

import numpy as np
from collections import defaultdict

word_freq = defaultdict(int)
word_sense_freq_map = {}

with open('./word_list.txt') as f:
# with open('./small.txt') as f:
    for w in f:
        w = w.rstrip()
        dataset = './dataset/words/%s' % w

        matrix = np.loadtxt(dataset,delimiter=',')
        word_freq[w] = matrix.shape[0]

        labels = matrix[:,-1]

        sense_freq = defaultdict(int)
        for l in labels:
            sense_freq[l] += 1
        word_sense_freq_map[w] = sense_freq

for w in sorted(word_freq, key=word_freq.get, reverse=True):
    print w, word_freq[w]
    print '(sense id: frequncy)'
    sense_freq = word_sense_freq_map[w]
    for sid, count in sense_freq.items():
        print '%d: %d' %(sid, count)
    print ''
