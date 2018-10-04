# add queuing delay into halo
import numpy as np
from scipy import interpolate
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
import gym
from gym import spaces
from gym.utils import seeding
import copy

LOG_FILE = './results/log'
VMAF_LOGS = './vmaf_logs/'

MTU_PACKET_SIZE = 1500
VIDEO_BIT_RATE = [0.01, 0.3, 0.5, 0.8, 1.1, 1.4]
A_DIM = len(VIDEO_BIT_RATE)
DELAY_GRADIENT_MAX = 1.0
BUFFER_NORM_FACTOR = 5.0


PACKET_PAYLOAD_PORTION = 0.95

NOISE_LOW = 0.95
NOISE_HIGH = 1.05


S_INFO = 6
S_LEN = 10  # take how many frames in the past

INPUT_W = 64
INPUT_H = 36
INPUT_D = 3
# long seq
INPUT_SEQ = 25
OUTPUT_DIM = 5
KERNEL = 64
DENSE_SIZE = 64


class QARCEnv(gym.Env):
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

    def __init__(self, random_seed=0, filename=None):
        #os.system('rm -rf results')
        os.system('mkdir results')
        np.random.seed(int(time.time()))
        _id = np.random.randint(1000)
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
        self.last_rtt = -1
        self.last_vmaf = -1
        self.time_stamp = 0.0
        self.state = np.zeros((S_INFO, S_LEN))

        self.x = tf.placeholder(
            shape=[None,  INPUT_SEQ, INPUT_H, INPUT_W, INPUT_D], dtype=tf.float32)
        self.y_ = tf.placeholder(shape=[None, OUTPUT_DIM], dtype=tf.float32)
        self.core_net = self.vqn_model(self.x)

        self.core_net_loss = tflearn.objectives.mean_square(
            self.core_net, self.y_)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "model/nn_model_ep_best.ckpt")
        self.x_buff = np.zeros([INPUT_SEQ, INPUT_H, INPUT_W, INPUT_D])

        self.action_space = spaces.Discrete(A_DIM)
        self.observation_space = spaces.Box(
            0, 5.0, [S_INFO, S_LEN], dtype=np.float32)

        self.last_vmaf_fake = -1
        _strtime = time.strftime('%b-%d-%H:%M:%S-%Y', time.localtime())
        self.log_file = open(LOG_FILE + '_' + str(_strtime), 'wb')

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
        #print 'random video', self._video
        _video_file = os.listdir('img/' + self._video + '/')
        _count = int(len(_video_file) / 5.0)
        self._videoindex = -1
        self._video_vmaf = self.read_all_logs(
            VMAF_LOGS + self._video + '_vmaf.log')
        self._video_len = self.read_all_logs(
            VMAF_LOGS + self._video + '_len.log')
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
                #print filename
                filename = 'img/' + self._video + \
                    '/' + str(_index + p - 1) + '.png'
                img = cv2.imread(filename)
            self.x_buff[-1, :, :, :] = img
        return self.x_buff

    def reset(self):
        if self._filename is None:
            self.delay_queue = DelayQueue(0.01)
        else:
            self.delay_queue = NoloopDelayQueue(self._filename, 0.01)
        self.random_video()
        self.state = np.zeros((S_INFO, S_LEN))
        return self.state
        #self.time_stamp = 0

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

    def send_video_queue(self, video_quality, timeslot, replay=False):
        # another fast algorithm with random walk - poisson process
        video_quality *= 1024 * 1024
        video_quality = int(video_quality)
        _packet_count = int(video_quality / MTU_PACKET_SIZE)
        _last_packet_len = video_quality % MTU_PACKET_SIZE
        if _last_packet_len > 0:
            _packet_count += 1
        _temp = np.random.randint(0, int(timeslot), _packet_count)
        _d_ms_array = _temp[np.argsort(_temp)] + self.simtime

        if replay == True:
            #_start = time.time()
            _tmp_queue = copy.copy(self.delay_queue)
            #_end = time.time()
            #print _end - _start, 's'

        for _t in range(len(_d_ms_array) - 1):
            self.delay_queue.write(MTU_PACKET_SIZE, _d_ms_array[_t])

        if _last_packet_len > 0:
            self.delay_queue.write(_last_packet_len, _d_ms_array[-1])
        else:
            self.delay_queue.write(MTU_PACKET_SIZE, _d_ms_array[-1])

        self.simtime += timeslot
        _total_delay, _total_bytes_len, _limbo_bytes_len = self.delay_queue.syncread(
            timeslot)
        #assert _limbo_bytes_len <= 10.0
        #assert  _total_delay < 100 * 1000
        if replay == True:
            del self.delay_queue
            self.delay_queue = _tmp_queue

        return _total_delay, _total_bytes_len, _limbo_bytes_len

    def step_without_change(self, action):
        bit_rate = action
        state = self.state
        delay, loss, recv_bitrate, rtt, throughput, limbo_bytes_len = \
            self.get_video_chunk(bit_rate, replay=True)

        _norm_bitrate = VIDEO_BIT_RATE[bit_rate]
        rtt = float(rtt) / float(1000)
        if self.last_rtt < 0:
            self.last_rtt = rtt
        _norm_send_bitrate = bit_rate / A_DIM
        _queuing_delay = abs(rtt - self.last_rtt)
        _norm_recv_bitrate = min(
            float(recv_bitrate) / delay / BUFFER_NORM_FACTOR, 1.0)

       # assert limbo_bytes_len <= 10.0

        self.time_stamp += delay  # in ms
        if bit_rate == 0:
            vmaf = self.last_vmaf
            _queuing_delay += 1.0
        else:
            vmaf = self.get_vmaf(bit_rate - 1)
        if self.last_vmaf < 0:
            self.last_vmaf = vmaf

            # min(_queuing_delay, DELAY_GRADIENT_MAX) / DELAY_GRADIENT_MAX - \
        reward = \
            1.0 * vmaf - \
            0.2 * _norm_bitrate - \
            1.0 / DELAY_GRADIENT_MAX * min(_queuing_delay, DELAY_GRADIENT_MAX) - \
            1.0 * abs(self.last_vmaf - vmaf)

        #observation, reward, done, info = env.step(action)
        return state, reward, False, {}

    def step(self, action):
        bit_rate = action
        state = self.state
        delay, loss, recv_bitrate, rtt, throughput, limbo_bytes_len = \
            self.get_video_chunk(bit_rate)

        _norm_bitrate = VIDEO_BIT_RATE[bit_rate]
        rtt = float(rtt) / float(1000)
        if self.last_rtt < 0:
            self.last_rtt = rtt
        _norm_send_bitrate = bit_rate / A_DIM
        _queuing_delay = abs(rtt - self.last_rtt)
        _norm_recv_bitrate = min(
            float(recv_bitrate) / delay / BUFFER_NORM_FACTOR, 1.0)

       # assert limbo_bytes_len <= 10.0

        self.time_stamp += delay  # in ms
        if bit_rate == 0:
            vmaf = self.last_vmaf
            _queuing_delay += 1.0
        else:
            vmaf = self.get_vmaf(bit_rate - 1)
        if self.last_vmaf < 0:
            self.last_vmaf = vmaf

        reward = \
            1.0 * vmaf - \
            0.2 * _norm_bitrate - \
            1.0 / DELAY_GRADIENT_MAX * min(_queuing_delay, DELAY_GRADIENT_MAX) - \
            1.0 * abs(self.last_vmaf - vmaf)

        self.last_vmaf = vmaf
        self.last_rtt = rtt
        self.log_file.write(str(self.time_stamp) + '\t' +
                            str(_norm_bitrate) + '\t' +
                            str(recv_bitrate) + '\t' +
                            str(limbo_bytes_len) + '\t' +
                            str(rtt) + '\t' +
                            str(vmaf) + '\t' +
                            str(reward) + '\n')
        self.log_file.flush()

        state = np.roll(state, -1, axis=1)
        state[0, -1] = _norm_send_bitrate  # last quality
        state[1, -1] = _norm_recv_bitrate  # kilo byte / ms
        state[2, -1] = _queuing_delay  # max:500ms
        state[3, -1] = float(loss)  # changed loss
        state[4, -1] = self.last_vmaf_fake
        # test:add fft feature
        #_fft = np.fft.fft(state[1])
        #state[5] = _fft.real
        #state[6] = _fft.imag
        state[5, 0:5] = self.predict_vmaf()
        self.last_vmaf_fake = state[5, bit_rate]
        self.state = state
        #observation, reward, done, info = env.step(action)
        return state, reward, False, {}

    def get_video_chunk(self, quality, timeslot=1000, replay=False):

        choose_quality = VIDEO_BIT_RATE[quality]
        # self.video_size[quality]
        queuing_delay, _total_bytes_len, _limbo_bytes_len = self.send_video_queue(
            choose_quality, timeslot, replay)
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
