from __future__ import division, print_function, absolute_import

import tflearn
import h5py
import os

def load_h5(filename):
    h5f = h5py.File(filename, 'r')
    X = h5f['X']
    Y = h5f['Y']
    return X, Y

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

X, Y = load_h5('train_720p_vmaf.h5')
testX, testY = load_h5('test_720p_vmaf.h5')
input_layer = tflearn.input_data(shape=[None, 25, 36, 64, 3])
dense1 = tflearn.fully_connected(input_layer, 64, activation='relu')
dense1 = tflearn.fully_connected(dense1, 64, activation='relu')
dense1 = tflearn.fully_connected(dense1, 64, activation='relu')
out = tflearn.fully_connected(dense1, 5, activation='sigmoid')

net = tflearn.regression(out, optimizer='adam',
                         loss='mean_square', learning_rate=1e-3)

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=50, validation_set=(testX, testY),
          show_metric=False, run_id="dense_model")
