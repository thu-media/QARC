import numpy as np
from TinyQueue import TinyQueue
from Packet import DelayedPacket, PartialPacket
import random
import os
import time
from matplotlib import pyplot as plt
TRAIN_SIM_FLODER = "./train_sim_traces/"
BITRATE_TIME = 1.0


class DelayQueue:
    def reset_trace_core(self, cooked_file):
        _schedule = []
        f = open(TRAIN_SIM_FLODER + cooked_file, "r")
        if not f:
            print "error:open file"
        cooked_time, cooked_bw = [], []
        _base_time = 0.0
        for line in f:
            parse = line.split()
            if len(parse) > 1:
                cooked_time.append(float(parse[0]) - _base_time)
                cooked_bw.append(float(parse[1]))
                _base_time = float(parse[0])
        f.close()
        # random trace
        # half_cook_len = len(cooked_bw) / 2
        # random_start = np.random.randint(half_cook_len)
        # random_end = np.random.randint(half_cook_len)
        # cooked_time = cooked_time[random_start:half_cook_len + random_end]
        # cooked_bw = cooked_bw[random_start:half_cook_len + random_end]

        for (_time, _bw) in zip(cooked_time, cooked_bw):
            if _time < 0.05:
                continue
            _bw *= _time
            _time *= 1000  # random_seed
            _bw *= 1024 * 1024
            _packets_count = int(_bw / self.SERVICE_PACKET_SIZE)
            if _packets_count == 0:
                _packets_count = 1
            _temp = np.random.randint(0, int(_time), _packets_count)
            _d_ms_array = _temp[np.argsort(_temp)] + self._basic_ms
            for _d_ms in _d_ms_array:
                self._limbo.push(_d_ms)
            self._basic_ms += _time

    def reset_trace(self):
        #_start = time.time()
        trace_files = os.listdir(TRAIN_SIM_FLODER)
        _test_num = len(trace_files)
        _random_index = np.random.randint(_test_num)
        self.reset_trace_core(trace_files[_random_index])
        #print 'reset trace ', trace_files[_random_index], ' completed:', time.time() - _start, 's'

    def __init__(self, loss_rate=0.0):

        self.SERVICE_PACKET_SIZE = 1500
        self._sender = TinyQueue()
        self._limbo = TinyQueue()

        self._loss_rate = loss_rate
        self._queued_bytes = 0
        self._base_timestamp = 0
        self._packets_added = 0
        self._packets_dropped = 0
        #self._time = TinyTime()
        self._basic_ms = 0

        self.last_queuing_delay = 0

        # self.init_trace()
        self.reset_trace()

    def write(self, packet, now):
        r = random.random()
        self._packets_added += 1
        if (r < self._loss_rate):
            self._packets_dropped += 1
            # print("%s,Stochastic drop of packet,packets_added so far %d,packets_dropped %d,drop rate %f" %
            #      (self._name, self._packets_added, self._packets_dropped,
            #       float(self._packets_dropped) / float(self._packets_added)))
        else:
            p = DelayedPacket(now, now, packet)
            self._sender.push(p)
            self._queued_bytes += packet

    def getfront(self):
        if self._limbo.size() <= 0:
            self.reset_trace()
            # self._sender.clear()
            return None
        else:
            return self._limbo.front()

    def getdelay(self, delaytime):
        while True:
            _front = self.getfront()
            if _front is None:
                return None
            if _front >= delaytime:
                break
            self._limbo.pop()
        _delta = self.getfront() - delaytime
        return _delta

    def syncread(self, duration):
        #_start = time.time()
        _delay_len = self._sender.size()
        _rtt = 0
        _bytes_send = 0
        self._limbo.start()
        self._sender.start()

        while True:
            _timestart = self.getfront()
            if _timestart is None:
                break
                # return None, None, None
            while self._sender.size() > 0:
                _front = self.getfront()
                if _front is None:
                    break
                    # return None, None, None
                if _front - _timestart >= duration:
                    break
                _delay_front = self._sender.front()
                # print 'read:_delay_front',_delay_front
                _d_delay = self.getdelay(_delay_front.entry_time)
                if _d_delay is None:
                    break
                    # return None, None, None
                _rtt += _d_delay
                _bytes_send += _delay_front.contents
                self._sender.pop()
                self._limbo.pop()
            break
        # print 'sync time:', time.time() - _start, 's'
        _senderend = self._sender.stop()
        _av_bytes_send = self._limbo.stop() * self.SERVICE_PACKET_SIZE
        
        if _senderend > 0:
            return _rtt / _senderend, _bytes_send, _av_bytes_send
        else:
            return 0.0, _bytes_send, _av_bytes_send