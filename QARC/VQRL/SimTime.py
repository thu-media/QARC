#this code is not related to 'tiny times' which is producted by Jingming Guo:)
class SimTime:
    def __init__(self):
        self._tick = 0

    def gettime(self):
        return self._tick

    def add(self,delay):
        self._tick += delay