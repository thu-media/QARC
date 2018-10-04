import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.pyplot import plot, savefig
import sys

DIR = sys.argv[1]
def tocdf(y):
    dx = .01

    # Normalize the data to a proper PDF
    y_ = y / (dx*y).sum()

    # Compute the CDF
    return np.cumsum(y_*dx)


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
    _handle = open(DIR + '/' + _filename, 'r')
    x = []
    for lines in _handle:
        x.append(float(lines))
    return np.array(x)


LW = 4
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
font = {'size': 20}
matplotlib.rc('font', **font)
plt.figure()

fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
from matplotlib import ticker
#formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
#formatter.set_powerlimits((-1, 1))
# ax.xaxis.set_major_formatter(formatter)
ax.set_ylabel('Weights')
ax.set_xlabel('Sequence')
ax.grid(True)
#plt.ylim((0.0, 0.05))
#plt.xlim(0.0, 25.0)

_range = []
index = 0
_plot_array = ['-.', '', '--', '-', '+']
_plot_color = ['darkred', 'darkblue', 'salmon', 'gray', 'darkgreen']
for p in os.listdir(DIR):
    _attention = p.split('.')
    _x = tocdf(readarray(p))
    ax.plot(_x, _plot_array[index], color=_plot_color[index],
            label=_attention[0], lw=LW)
    index += 1

ax.legend()
savefig(DIR + '_cdf.png')
