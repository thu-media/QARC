# add queuing delay into halo
import numpy as np
from scipy import interpolate
from DelayQueue import DelayQueue
#from SimTime import SimTime
import time

MTU_PACKET_SIZE = 1500

RANDOM_SEED = 42
PACKET_PAYLOAD_PORTION = 0.95

NOISE_LOW = 0.95
NOISE_HIGH = 1.05

BITRATE_MIN = 0.3
BITRATE_MAX = 1.2
BITRATE_TIME = 1.0
BITRATE_LEVELS = 50

FPS = 25


class Environment:
    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        self.delay_queue = DelayQueue(0.01)
        self.video_size = []
        self.simtime = 0.0

        delta = (BITRATE_MAX - BITRATE_MIN) / BITRATE_LEVELS
        VIDEO_BIT_RATE = []
        for t in range(BITRATE_LEVELS):
            self.video_size.append(delta * t + BITRATE_MIN)

    def send_video_queue(self, video_quality, timeslot):
        # another fast algorithm with random walk - poisson process
        video_quality *= 1024 * 1024
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
        if _total_delay > 100 * 1000:
            assert 'error delay,delay bigger than 100!'
        return _total_delay, _total_bytes_len, _limbo_bytes_len
        
    def get_images(self):
        return None
     
    def get_vmaf(self):
        return None

    def get_video_chunk(self, quality, timeslot=1000):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        # in our env,'video_chunk_counter' is useless
        choose_quality = self.video_size[quality]
        _timestart = time.time()
        queuing_delay, _total_bytes_len, _limbo_bytes_len = self.send_video_queue(
            choose_quality, timeslot)
        if queuing_delay is None:
            return None
            
        _total_bytes_len = float(_total_bytes_len) / float(1024 * 1024)
        _limbo_bytes_len = float(_limbo_bytes_len) / float(1024 * 1024)
        # print choose_quality, _total_bytes_len, _limbo_bytes_len
        #assert (choose_quality >= _total_bytes_len)
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
        #assert choose_quality < packet_payload
        #print 'estimate', round(choose_quality * 1024, 2), 'kb', ',recv', round(_total_bytes_len * 1024, 2), 'kb', ',delay', queuing_delay, ',loss', loss
        return BITRATE_TIME, loss, \
            packet_payload, \
            queuing_delay, _real_packet_payload, _limbo_bytes_len
