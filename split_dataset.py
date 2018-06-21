#!/usr/bin/python

import os
import numpy as np
from collections import defaultdict
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def up_sample(X,y,min_threshold):
    X = np.array(X)
    y = np.array(y)

    class_freq = Counter(y)
    minority_classes = {x: class_freq[x] for x in class_freq if class_freq[x] <
            min_threshold}

    upsampled_X = np.array(X)
    upsampled_y = np.array(y)
    for l in minority_classes:
        indices = np.where(np.array(y)==l)
        minor_X = X[indices,:][0]
        minor_y = y[indices]
        shape_X = minor_X.shape
        assert minor_X.shape[0] == minor_y.shape[0]
        resampled_X, resampled_y = resample(minor_X, minor_y, n_samples =
                min_threshold, replace=True)
        print resampled_X.shape, resampled_y.shape
        upsampled_X = np.vstack((upsampled_X,resampled_X))
        upsampled_y = np.append(upsampled_y,resampled_y)
    assert upsampled_X.shape[0] == upsampled_y.shape[0]
    return upsampled_X,upsampled_y

out_dir = './dataset/word_pos_large'
sub_dirs = ['X_train', 'Y_train', 'X_test', 'Y_test']
for d in sub_dirs:
    _dir = '%s/%s' %(out_dir,d)
    if not os.path.exists(_dir):
        os.makedirs(_dir)

# with open('./word_list.txt') as f:
with open('./large.txt') as f:
    for w in f:
        w = w.rstrip()
        # dataset = './dataset/words/%s' % w
        dataset = './dataset/large_instances/%s' % w

        matrix = np.loadtxt(dataset,delimiter=',')
        X = matrix[:,:-1]
        y = matrix[:,-1]
        
        MIN_THRESHOLD = 20
        [upsampled_X,upsampled_y] = up_sample(X,y,MIN_THRESHOLD)
        # X_train, X_test, y_train, y_test = train_test_split(X, y,
        X_train, X_test, y_train, y_test = train_test_split(upsampled_X, upsampled_y,
                # test_size=0.2, stratify=y, random_state=42)
                test_size=0.2, stratify=upsampled_y, random_state=42)
    
        np.savetxt('%s/X_train/%s' %(out_dir,w), X_train,delimiter=',')
        np.savetxt('%s/Y_train/%s' %(out_dir, w), y_train,delimiter=',')
        np.savetxt('%s/X_test/%s' %(out_dir, w), X_test,delimiter=',')
        np.savetxt('%s/Y_test/%s' %(out_dir, w), y_test,delimiter=',')
