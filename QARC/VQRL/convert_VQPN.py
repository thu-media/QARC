# add queuing delay into halo
import numpy as np
import time
import cv2
import os
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import max_pool_2d

VIDEO_BIT_RATE = [0.3, 0.5, 0.8, 1.1, 1.4]

RANDOM_SEED = 42
PACKET_PAYLOAD_PORTION = 0.95

NOISE_LOW = 0.95
NOISE_HIGH = 1.05

INPUT_W = 64
INPUT_H = 36
INPUT_D = 3
# long seq
INPUT_SEQ = 25
OUTPUT_DIM = 5
KERNEL = 32
DENSE_SIZE = 128


class Environment:

    def CNN_Core(self, x, reuse=False):
        with tf.variable_scope('cnn_core', reuse=reuse):
            network = tflearn.conv_2d(
                x, KERNEL, 5, activation='relu', regularizer="L2", weight_decay=0.0001)
            network = tflearn.avg_pool_2d(network, 3)
            network = tflearn.conv_2d(
                network, KERNEL, 3, activation='relu', regularizer="L2", weight_decay=0.0001)
            network = tflearn.avg_pool_2d(network, 2)
            network = tflearn.fully_connected(
                network, DENSE_SIZE, activation='relu')
            split_flat = tflearn.flatten(network)
            return split_flat

    def vqn_model(self, x):
        with tf.variable_scope('vqn'):
            inputs = tflearn.input_data(placeholder=x)
            _split_array = []

            for i in range(INPUT_SEQ):
                tmp_network = tf.reshape(
                    inputs[:, i:i+1, :, :, :], [-1, INPUT_H, INPUT_W, INPUT_D])
                if i == 0:
                    _split_array.append(self.CNN_Core(tmp_network))
                else:
                    _split_array.append(self.CNN_Core(tmp_network, True))

            merge_net = tflearn.merge(_split_array, 'concat')
            merge_net = tflearn.flatten(merge_net)
            _count = merge_net.get_shape().as_list()[1]

            with tf.variable_scope('full-cnn'):
                net = tf.reshape(
                    merge_net, [-1, INPUT_SEQ, _count / INPUT_SEQ, 1])
                network = tflearn.conv_2d(
                    net, KERNEL, 5, activation='relu', regularizer="L2", weight_decay=0.0001)
                network = tflearn.max_pool_2d(network, 3)
                network = tflearn.layers.normalization.batch_normalization(
                    network)
                network = tflearn.conv_2d(
                    network, KERNEL, 3, activation='relu', regularizer="L2", weight_decay=0.0001)
                network = tflearn.max_pool_2d(network, 2)
                network = tflearn.layers.normalization.batch_normalization(
                    network)
                cnn_result = tflearn.fully_connected(
                    network, DENSE_SIZE, activation='relu')

            out = tflearn.fully_connected(
                cnn_result, OUTPUT_DIM, activation='sigmoid')

            return out

    def __init__(self, random_seed=RANDOM_SEED, filename=None):
        np.random.seed(random_seed)
        self.simtime = 0.0
        self._video = None
        self._videocount = 0.0
        self._videoindex = 0.0
        self._video_vmaf = []
        self._video_len = []
        self.random_video()

        self.x = tf.placeholder(
            shape=[None,  INPUT_SEQ, INPUT_H, INPUT_W, INPUT_D], dtype=tf.float32)
        self.y_ = tf.placeholder(shape=[None, OUTPUT_DIM], dtype=tf.float32)
        self.core_net = self.vqn_model(self.x)

        self.core_net_loss = tflearn.objectives.mean_square(
            self.core_net, self.y_)
        # + lossL2
        # self.core_train_op = tf.train.AdamOptimizer(
        #    learning_rate=LR_RATE).minimize(self.core_net_loss)
        # self.core_net_acc = tf.reduce_mean(
        #    tf.abs(core_net - y_) / (tf.abs(core_net) + tf.abs(y_) / 2))
        # core_net_mape = tf.subtract(1.0, tf.reduce_mean(
        #    tf.abs(core_net - y_) / tf.abs(y_)))
        #train_len = X.shape[0]

        #g2 = tf.Graph()
        #gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "model/nn_model_ep_350.ckpt")
        self.x_buff = np.zeros([INPUT_SEQ, INPUT_H, INPUT_W, INPUT_D])

    def read_all_logs(self, filename):
        _file = open(filename, 'r')
        _array = []
        for _line in _file:
            _stride = _line.split(',')
            _tmp = []
            for p in _stride:
                if len(p) > 1:
                    _tmp.append(float(p))
            _array.append(np.array(_tmp))
        return np.array(_array)

    def random_video(self):
        self._video = '7.flv'
        print 'random video', self._video
        _video_file = os.listdir('img/' + self._video + '/')
        _count = int(len(_video_file) / 5.0)
        self._videoindex = -1
        self._video_vmaf = self.read_all_logs(self._video + '_vmaf.log')
        self._video_len = self.read_all_logs(self._video + '_len.log')
        self._videocount = min(
            self._video_vmaf.shape[0], self._video_len.shape[0])
        self._videocount = min(self._videocount - 1, _count)

    def get_image(self):
        self._videoindex += 1
        if self._videoindex >= self._videocount:
            return None
        _index = self._videoindex * 5
        for p in range(1, 6):
            self.x_buff = np.roll(self.x_buff, -1, axis=1)
            filename = 'img/' + self._video + '/' + str(_index + p) + '.png'
            img = cv2.imread(filename)
            if img is None:
                print filename
                filename = 'img/' + self._video + \
                    '/' + str(_index + p - 1) + '.png'
                img = cv2.imread(filename)
            self.x_buff[-1, :, :, :] = img
        return self.x_buff

    def predict_vmaf(self):
        _images = self.get_image()
        if _images is None:
            return None
        _images = np.reshape(
            _images, [1, INPUT_SEQ, INPUT_H, INPUT_W, INPUT_D])
        _test_y = self.sess.run(self.core_net, feed_dict={self.x: _images})
        return _test_y[0]

    def get_vmaf(self, video_quality):
        return self._video_vmaf[self._videoindex+1, video_quality]

_file  = open('7.flv_all.log','w')
env = Environment()
while True:
    _value = env.predict_vmaf()
    if _value is None:
        _file.close()
        exit()
    else:
        for p in _value:
            _file.write(str(p) + ',')
        _file.write('\n')