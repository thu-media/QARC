import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.pyplot import plot, savefig
import sys

FILENAME = sys.argv[1]


def tocdf(x, y):
    dx = .01

    # Normalize the data to a proper PDF
    y_ = y / (dx*y).sum()

    # Compute the CDF
    return x, np.cumsum(y_*dx)


def filter(x, y, z=500):
    x = np.array(x)
    y = np.array(y)
    _x, _y = [], []
    for p in range(z, len(y) - z):
        _x.append(x[p])
        _y.append(np.mean(y[p - z:p + z]))
    return np.array(_x), np.array(_y)


def exponential_moving_average(x, data, alpha=0.993):
    _tmp = []
    data *= 5.0 / 4.0
    _val = data[0]
    for p in data:
        _val = p * (1.0 - alpha) + _val * alpha
        _tmp.append(_val)
    return x, np.array(_tmp)


def readarray(_filename):
    _handle = open(_filename, 'r')
    x = []
    for lines in _handle:
        p = lines.split(' ')
        tmp = []
        for _p in p:
            tmp.append(float(_p))
        x.append(tmp)
    return np.array(x)


LW = 3
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
font = {'size': 23}
matplotlib.rc('font', **font)
plt.figure()

fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
from matplotlib import ticker
#formatter = ticker.ScalarFormatter(useMathText=True)
#formatter.set_scientific(True)
#formatter.set_powerlimits((-1, 1))
#ax.xaxis.set_major_formatter(formatter)
ax.set_ylabel('VMAF')
ax.set_xlabel('bitrate(Kbps)')
ax.grid(True)

_x = readarray(FILENAME)

plt.ylim((0.0, 100.0))
plt.xlim(150, 1450)
_x_range = [200, 300, 500, 800, 1100, 1400]
# ax.set_xscale('log')
A, = ax.plot(_x_range, _x[0], '-.', color='darkred', label='libx264', lw=LW)
ax.scatter(_x_range, _x[0], marker = 'x', s = 100, color='darkred')
B, = ax.plot(_x_range, _x[1], color='darkblue',
             alpha=0.8, label='libx265', lw=LW)
ax.scatter(_x_range, _x[1], marker = 'o', s = 100, color='darkblue')
C, = ax.plot(_x_range, _x[2], '--', color='salmon', label='avone', lw=LW)
ax.scatter(_x_range, _x[2], marker = '+', s = 100, color='salmon')

ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
          ncol=3, mode="expand")
savefig(FILENAME + '.png')
