
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot, savefig
from pykalman import KalmanFilter
import sys

LW = 4
SCALE = 300



def read_csv(filename):
    _file = open(filename, 'r')
    _tmp = []
    for _line in _file:
        _tmp.append(float(_line))
    return np.array(_tmp)


def moving_average(data, alpha=0.9):
    _tmp = []
    _val = data[0]
    for p in data:
        _val = p * (1.0 - alpha) + _val * alpha
        _tmp.append(_val)
    return np.array(_tmp)


plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.labelweight'] = 'bold'
font = {'size': 28}
matplotlib.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(15, 6), dpi=100)
y_ = read_csv(sys.argv[1])[101:]
ax1.grid(True)
ax1.set_title(sys.argv[1])
l4 = ax1.plot(y_, color='red', lw=LW, alpha=0.3, label='original')
# for p in range(3):
l4 = ax1.plot(moving_average(y_), color='red', lw=LW, label='ma')
ax1.legend()
savefig(sys.argv[1] + '.png')
print 'done'
