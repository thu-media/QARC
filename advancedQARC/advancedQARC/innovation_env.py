# add queuing delay into halo
import numpy as np
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
        self._videoindex = 0.0
        self._video_vmaf = []
        self._video_len = []
        self.random_video()

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
        self._video_vmaf = self.read_all_logs('logs/' + self._video + '_vmaf.log')
        self._video_len = self.read_all_logs('logs/' + self._video + '_len.log')
        self._videocount = min(
            self._video_vmaf.shape[0], self._video_len.shape[0])
        self._videocount = min(self._videocount - 1, _count)

    def get_single_image(self):
        self._videoindex += 1
        if self._videoindex >= self._videocount:
            self.random_video()
            return self.get_single_image()
        _index = self._videoindex * 5
        _img_array = []
        for p in range(1, 6):
            filename = 'img/' + self._video + '/' + str(_index + p) + '.png'
            img = cv2.imread(filename)
            if img is None:
                filename = 'img/' + self._video + \
                    '/' + str(_index + p - 1) + '.png'
                img = cv2.imread(filename)
            gray_image = np.zeros([img.shape[0], img.shape[1]])
            gray_image = img[:, :, 0] * 0.12 + \
                img[:, :, 1] * 0.29 + img[:, :, 2] * 0.59
            _mean = np.mean(gray_image)
            _img_array.append(_mean)
        _mean_all = np.mean(_img_array) / 255.0
        #print _mean_all
        return _mean_all
        
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
