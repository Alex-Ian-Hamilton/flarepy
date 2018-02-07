# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:57:57 2017

@author: alex_
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.font_manager import FontProperties

int_points = 1000 # The width of the total dataset (#values)
int_ricker_a = 50
int_gauss_std = 50

# Create the figure/axes for holding the plot
figure = plt.figure()          # Can contain multiple sub-plots
axes = figure.add_subplot(111) # In this case only make one plot (set of axes)

# Data arrays
arr_y_gaussian = signal.gaussian(int_points, std=int_gauss_std)
arr_y_ricker = signal.ricker(int_points, int_ricker_a)
arr_x = np.arange(0,int_points) # Makes an array: [0, 1, ..., len(points)-1]

# Add the lines to the plot
axes.plot(arr_x, arr_y_gaussian, label='Gaussian (STD='+str(int_gauss_std)+')')
axes.plot(arr_x, arr_y_ricker, linestyle='--', color='gray', label='Ricker (a='+str(int_ricker_a)+')')

# Add the detected peak/s
arr_x_gau_cwt_peaks = signal.find_peaks_cwt(arr_y_gaussian, [int_gauss_std])
arr_y_gau_cwt_peaks = arr_y_gaussian[arr_x_gau_cwt_peaks] # Get the height from the Gaussian)
axes.plot(arr_x_gau_cwt_peaks, arr_y_gau_cwt_peaks, 'x', label='#'+str(len(arr_x_gau_cwt_peaks))+' peaks for width ['+str(int_gauss_std)+']', markersize=4, lw=3)

# Add plot axes labels and title
title = axes.set_title('Plot of Ricker CWT used on a Gaussian function')
axes.set_xlabel('points')
axes.set_ylabel('value')

# Add plot legend (use smaller font to make it smaller)
fontP = FontProperties()
fontP.set_size('xx-small')
legend = axes.legend(prop = fontP)

# Save the plot (high dpi/resolution)
figure.savefig('gaussian.png', dpi=900)