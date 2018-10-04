# add queuing delay into halo
import numpy as np
from DelayQueue import DelayQueue
from fixed_delay_queue import NoloopDelayQueue
#from SimTime import SimTime
import time
import cv2
import os
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import max_pool_2d

MTU_PACKET_SIZE = 1500
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

            
        with tf.variable_scope('full-lstm'):
            net = tf.reshape(merge_net, [-1, _count / DENSE_SIZE, DENSE_SIZE])
            net = tflearn.gru(net, DENSE_SIZE, return_seq=True)
            out_gru = tflearn.gru(net, DENSE_SIZE,dropout=0.8)
            gru_result = tflearn.fully_connected(out_gru, DENSE_SIZE, activation='relu')

        out = tflearn.fully_connected(
            gru_result, OUTPUT_DIM, activation='sigmoid')

        return out

    def __init__(self, random_seed=RANDOM_SEED, filename=None):
        np.random.seed(random_seed)
        self._filename = filename
        if self._filename is None:
            self.delay_queue = DelayQueue(0.01)
        else:
            self.delay_queue = NoloopDelayQueue(self._filename, 0.01)
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
        _dirs = os.listdir('img/')
        self._video = _dirs[np.random.randint(len(_dirs))]
        #self._video = '3.flv'
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
            self.random_video()
            return self.get_image()
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
        _images = np.reshape(
            _images, [1, INPUT_SEQ, INPUT_H, INPUT_W, INPUT_D])
        _test_y = self.sess.run(self.core_net, feed_dict={self.x: _images})
        return _test_y[0]

    def get_vmaf(self, video_quality):
        return self._video_vmaf[self._videoindex+1, video_quality]

    def get_video_len(self, video_quality):
        return self._video_len[self._videoindex+1, video_quality]

    def send_video_queue(self, video_quality, timeslot):
        # another fast algorithm with random walk - poisson process
        video_quality = int(video_quality * 1024.0 * 1024.0)
        _packet_count = int(video_quality / MTU_PACKET_SIZE)
        _last_packet_len = video_quality % MTU_PACKET_SIZE
        if _last_packet_len > 0:
            _packet_count += 1
        _temp = np.random.randint(0, int(timeslot), _packet_count)
        _d_ms_array = _temp[np.argsort(_temp)] + self.simtime

        for _t in range(len(_d_ms_array) - 1):
            self.delay_queue.write(MTU_PACKET_SIZE, _d_ms_array[_t])

        if _last_packet_len > 0:
            self.delay_queue.write(_last_packet_len, _d_ms_array[-1])
        else:
            self.delay_queue.write(MTU_PACKET_SIZE, _d_ms_array[-1])

        self.simtime += timeslot
        _total_delay, _total_bytes_len, _limbo_bytes_len = self.delay_queue.syncread(
            timeslot)
        #assert  _total_delay < 100 * 1000
        return _total_delay, _total_bytes_len, _limbo_bytes_len

    def get_video_chunk(self, quality, timeslot=1000):

        choose_quality = VIDEO_BIT_RATE[quality]  # self.video_size[quality]

        queuing_delay, _total_bytes_len, _limbo_bytes_len = self.send_video_queue(
            choose_quality, timeslot)

        if queuing_delay is None:
            return None, None, None, None, None, None

        _total_bytes_len = float(_total_bytes_len) / float(1024 * 1024)
        _limbo_bytes_len = float(_limbo_bytes_len) / float(1024 * 1024)

        #throughput * duration * PACKET_PAYLOAD_PORTION
        packet_payload = _total_bytes_len * \
            np.random.uniform(NOISE_LOW, NOISE_HIGH)
        # use the delivery opportunity in mahimahi
        loss = 0.0  # in ms
        if packet_payload > choose_quality:
            loss = 0
            # add a multiplicative noise to loss
            _real_packet_payload = choose_quality * \
                np.random.uniform(NOISE_LOW, NOISE_HIGH)
        else:
            loss = 1 - packet_payload / choose_quality
            _real_packet_payload = packet_payload

        return timeslot / 1000.0, loss, \
            packet_payload, \
            queuing_delay, _real_packet_payload, _limbo_bytes_len
