import numpy as np
#import time
from TinyQueue import TinyQueue
from Packet import DelayedPacket, PartialPacket
import random
import os
import time
TRACE_FLODER = "./test_traces/"
TRAIN_SIM_FLODER = "./test_sim_traces/"
INTERVAL = 50
BITRATE_TIME = 1.0
MTU_PACKET_SIZE = 1500


class NoloopDelayQueue:
    def reset_trace_core(self, cooked_file):
        _schedule = []
        f = open(TRAIN_SIM_FLODER + cooked_file, "r")
        print 'opening file ', TRAIN_SIM_FLODER + cooked_file
        if not f:
            print "error:open file"
        # temp = 0
        cooked_time, cooked_bw = [], []
        _base_time = 0.0
        for line in f:
            parse = line.split()
            if len(parse) > 1:
                cooked_time.append(float(parse[0]) - _base_time)
                cooked_bw.append(float(parse[1]))
                _base_time = float(parse[0])
        f.close()
        _basic_ms = 0.0
        for (_time, _bw) in zip(cooked_time, cooked_bw):
            if _time < 0.05:
                continue
            _bw *= _time
            _time *= 1000  # random_seed
            _bw *= 1024 * 1024
            _packets_count = int(_bw / MTU_PACKET_SIZE)
            if _packets_count == 0:
                _packets_count = 1
            _temp = np.random.randint(0, int(_time), _packets_count)
            _d_ms_array = _temp[np.argsort(_temp)] + _basic_ms
            for _d_ms in _d_ms_array:
                self._schedule.push(_d_ms)
            _basic_ms += _time
        return _base_time

    def reset_trace(self):
        #_start = time.time()
        _trace_file = self.s_name
        self.reset_trace_core(self.s_name)

    def __init__(self, s_name=None, loss_rate=0.0):
        # np.random.seed(222)
        self.SERVICE_PACKET_SIZE = 1500
        self.s_name = s_name
        self._sender = TinyQueue()
        self._schedule = TinyQueue()

        #self._ms_delay = s_ms_delay
        self._loss_rate = loss_rate
        self._queued_bytes = 0
        #self._base_timestamp = base_timestamp
        self._packets_added = 0
        self._packets_dropped = 0
        #self._time = TinyTime()
        self._basic_ms = 0

        self.last_queuing_delay = 0

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
        if self._schedule.size() <= 0:
            return None
        else:
            return self._schedule.front()

    def getdelay(self, delaytime):
        while True:
            _front = self.getfront()
            if _front is None:
                return None
            if _front >= delaytime:
                break
            self._schedule.pop()
        _delta = self.getfront() - delaytime
        return _delta

    def syncread(self, duration):
        #_start = time.time()
        _delay_len = self._sender.size()
        _rtt = 0
        _bytes_send = 0
        self._schedule.start()
        self._sender.start()
        _timestart = self.getfront()
        if _timestart is None:
            return None, None, None
        while self._sender.size() > 0:
            _front = self.getfront()
            if _front is None:
                return None, None, None
            if _front - _timestart >= duration:
                break
            _delay_front = self._sender.front()
            # print 'read:_delay_front',_delay_front
            _d_delay = self.getdelay(_delay_front.entry_time)
            if _d_delay is None:
                return None, None, None
            _rtt += _d_delay
            _bytes_send += _delay_front.contents
            self._sender.pop()
            self._schedule.pop()
        # print 'sync time:', time.time() - _start, 's'
        _senderend = self._sender.stop()
        _av_bytes_send = self._schedule.stop() * self.SERVICE_PACKET_SIZE
        if _senderend > 0:
            return _rtt / _senderend, _bytes_send, _av_bytes_send
        else:
            return 0, _bytes_send, _av_bytes_send
