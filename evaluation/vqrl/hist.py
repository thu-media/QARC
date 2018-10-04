#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import sys
FILENAME = sys.argv[1]

_f = open(FILENAME + '.csv', 'r')

u_dp = [[] for i in range(4)]
u_l = [[] for i in range(4)]
u_c = [[] for i in range(4)]
u_r = [[] for i in range(4)]
i = 0
for line in _f:
    print 'line', line
    ls = [float(x) for x in line.split(',')]
    u_dp[i].append(ls[0])
    u_l[i].append(ls[1])
    u_c[i].append(ls[2])
    u_r[i].append(ls[3])
    i += 1

width = 0.8
space = 0.08
gap = 0.8
bt = 0
#print matplotlib.colors.ListedColormap()
#c = ('gray', 'darkred', 'salmon',  'lightblue')
c = ('darkblue', '#8fc2f6', 'lightblue',  'silver')

st1 = 0.6
left1 = (st1, st1+width+space, st1+2*(width+space), st1+3*(width+space))

st2 = left1[3]+width+gap
left2 = (st2, st2+width+space, st2+2*(width+space), st2+3*(width+space))

st3 = left2[3]+width+gap
left3 = (st3, st3+width+space, st3+2*(width+space), st3+3*(width+space))

#st4 = left3[3]+width+gap
# left3 = (0,0,0,0)#(st4, st4+width+space, st4+2*(width+space), st4+3*(width+space))

y1 = [np.mean(u_dp[j])-bt for j in range(4)]
y2 = [np.mean(u_l[j])-bt for j in range(4)]
y3 = [np.mean(u_c[j])-bt for j in range(4)]
y4 = [np.mean(u_r[j])-bt for j in range(4)]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.labelsize'] = 50
plt.rcParams['axes.labelweight'] = 'bold'
font = {'size': 50}
matplotlib.rc('font', **font)
WIDTH = 30
fig, ax = plt.subplots(figsize=(WIDTH,int(WIDTH*0.50)), dpi=30)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
for p in ax.spines:
    try:
        p.set_linewidth(50)
    except:
        pass

ax.set_xlim([0, left3[3]+width])
ax.set_ylim([0, 1.0])
ax.set_yticks(np.round(np.linspace(0, 1.0, 4), 1))
# ax.grid(True)
objects = (r'\textbf{Queuing Delay}(s)', r'\textbf{Sending Bitrate}(Mbps)', r'\textbf{Video Quality}[0,1]')
width1 = 6.5
plt.grid(True)
plt.xticks((left1[2]-width1*space, left2[2]-width1*space, left3[2]-width1 *
            space, left3[2] - width1*space), objects)
patterns = ["/", ".", "|", "-", "+", "x", "o", "O", ".", "*"]
p1 = ax.bar((left1[0], left2[0], left3[0], left3[0]), y1,
            bottom=bt, width=width, facecolor=c[0], alpha=0.8, ec='black', lw=8, label=r'Baseline')
#ax.fill_between([left1[0], left2[0]], [left3[0], left3[0]], color="white", hatch="X", edgecolor="b", linewidth=0.0)
p2 = ax.bar((left1[1], left2[1], left3[1], left3[1]), y2,
            bottom=bt, width=width, facecolor=c[1], ec='black', lw=8, label=r'$\alpha$:0.2 $\beta$:10.0 $\gamma$:1.0')
p3 = ax.bar((left1[2], left2[2], left3[2], left3[2]), y3,
            bottom=bt, width=width, facecolor=c[2], ec='black', lw=8, label=r'$\alpha$:0.1 $\beta$:10.0 $\gamma$:0.0')
p4 = ax.bar((left1[3], left2[3], left3[3], left3[3]), y4,
            bottom=bt, width=width, facecolor=c[3], ec='black', lw=8, label=r'$\alpha$:2.0 $\beta$:0.1 $\gamma$:0.0')
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, prop={'size': 60}, 
#           ncol=4, mode="expand", borderaxespad=0.)
ax.legend(loc="upper left", frameon=True, fontsize=45,
          labelspacing=0.3, handlelength=2.8, borderpad=0.4)
legend = plt.gca().get_legend()
frame = legend.get_frame()
# frame.set_linewidth(0.7)
ax.set_ylabel(r'\textbf{Average Value}')
ax.set_xlabel(r'\textbf{Broadband Network Uplink}')
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
# plt.tight_layout(pad=0.5)
# fig.savefig("cost_band.eps", format="eps")
fig.savefig(FILENAME.replace('.','_') + '.pdf')
# plt.show()
