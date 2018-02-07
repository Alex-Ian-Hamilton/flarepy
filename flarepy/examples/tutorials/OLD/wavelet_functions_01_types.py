# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:31:33 2017

@author: alex_
"""

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


# Parameters for generating graphs
points = 101
arr_x = np.arange(0,points) # x axis values


# Generate plot for ricker (Mexican hat) wavelet
figure_ricker = plt.figure()
axes_ricker = figure_ricker.add_subplot(111)
for a in range(1,10,3):
    arr_wavelet = signal.ricker(points, a) # y axis values for this wavelet
    axes_ricker.plot(arr_x, arr_wavelet, label='ricker('+str(points)+', '+str(a)+')')#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])
#plt.plot(arr_wave)
legend_ricker = axes_ricker.legend()
figure_ricker.show()

# Generate plot for morlet wavelet
figure_morlet = plt.figure()
axes_morlet = figure_morlet.add_subplot(111)
for a in range(1,10,3):
    arr_wavelet = signal.morlet(points, a) # y axis values for this wavelet
    axes_morlet.plot(arr_x, arr_wavelet, label='morlet('+str(points)+', '+str(a)+')')#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])
#plt.plot(arr_wave)
legend_morlet = axes_morlet.legend()
figure_morlet.show()

# Generate plot for exponential wavelet
figure_exp = plt.figure()
axes_exp = figure_exp.add_subplot(111)
for a in range(1,3,1):
    arr_wavelet = signal.exponential(points) # y axis values for this wavelet
    axes_exp.plot(arr_x, arr_wavelet, label='exponential('+str(points)+', '+str(a)+')')#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])
#plt.plot(arr_wave)
legend_exp = axes_exp.legend()
figure_exp.show()
