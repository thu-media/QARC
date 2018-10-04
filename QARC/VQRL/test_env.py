# add queuing delay into halo
import numpy as np
from scipy import interpolate
from DelayQueue import DelayQueue
from fixed_delay_queue import NoloopDelayQueue
#from SimTime import SimTime
import time
import cv2
import os

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
TEST_MOVIE = '0.flv'

class Environment:
    
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
        self._videoindex = 0
        self._video_vmaf = []
        self._video_len = []
        self.inital_video()


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

    def inital_video(self):
        self._video = self.read_all_logs(TEST_MOVIE + '_all.log')
        self._video_vmaf = self.read_all_logs(TEST_MOVIE + '_vmaf.log')
        self._video_len = self.read_all_logs(TEST_MOVIE + '_len.log')
        print self._video_vmaf

    def predict_vmaf(self):
        self._videoindex += 1
        return self._video[self._videoindex]

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
