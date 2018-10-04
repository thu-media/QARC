from __future__ import division, print_function, absolute_import

import tflearn
import h5py
import numpy as np
import os

def load_h5(filename):
    h5f = h5py.File(filename, 'r')
    X = h5f['X']
    Y = h5f['Y']
    return X, Y

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
X, Y = load_h5('train.h5')
testX, testY = load_h5('test.h5')
X = np.reshape(X,(-1,25,36*64*3))
testX = np.reshape(testX,(-1,25,36*64*3))

input_layer = tflearn.input_data(shape=[None, 25, 36*64*3])
lstm1 = tflearn.simple_rnn(input_layer, 64)
out = tflearn.fully_connected(lstm1, 5, activation='sigmoid')

net = tflearn.regression(out, optimizer='adam',
                         loss='mean_square', learning_rate=1e-3)

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=50, validation_set=(testX, testY),
          show_metric=False, run_id="dense_model")
