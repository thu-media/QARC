import numpy as np
  
class DelayedPacket:
    entry_time = 0
    release_time = 0
    contents = 0
    def __init__(self,s_e,s_r,s_c):
        self.entry_time = s_e
        self.release_time = s_r
        self.contents = s_c
    
    def record_send_time(self,s_t):
        self.send_time = s_t

class PartialPacket:
    bytes_earned = 0
    packet = None
    def __init__(self,s_b_e,s_packet):
        self.bytes_earned = s_b_e
        self.packet = s_packet