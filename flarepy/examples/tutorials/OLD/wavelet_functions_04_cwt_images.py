# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:54:05 2017

@author: alex_
"""

#import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.font_manager import FontProperties

int_points = 1000 # The width of the total dataset (#values)
int_ricker_a = 50
int_gauss_std = 50
lis_lis_cwt_widths = [[int_gauss_std-40], [int_gauss_std-20], [int_gauss_std], [int_gauss_std+20], [int_gauss_std+40]]
lis_lis_cwt_widths = [np.arange(1,20), np.arange(1,40), np.arange(1,60), np.arange(1,80), np.arange(1,100)]

# Create the figure/axes for holding the plot
figure = plt.figure()          # Can contain multiple sub-plots
plt.subplots_adjust(hspace=0.001)
plt.subplots_adjust(bottom=0.08, right=0.84, top=0.92, left=0.05)

# Data arrays
arr_y_gaussian = signal.gaussian(int_points, std=int_gauss_std)
#arr_y_ricker = signal.ricker(int_points, int_ricker_a)
arr_x = np.arange(0,int_points) # Makes an array: [0, 1, ..., len(points)-1]

# Add the initial plot with a line for the Gaussian
axes = figure.add_subplot(len(lis_lis_cwt_widths)+1, 1, 1)
axes.plot(arr_x, arr_y_gaussian, linewidth=1, label='Gaussian (STD='+str(int_gauss_std)+')')
#axes.plot(arr_x, arr_y_ricker, linewidth=1, linestyle='--', color='gray', label='Ricker (a='+str(int_ricker_a)+')')
axes.set_xticks([])

# Add plot legend (use smaller font to make it smaller)
fontP = FontProperties()
fontP.set_size('xx-small')
legend = axes.legend(prop = fontP)

# Change the y-axis font size and tick locations
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(4)

# Create lists of all the axes and datasets generated
lis_axes = []
lis_axes.append(axes)
lis_data_ctw = []
lis_img_plots = []
lis_clim = [0,0]

for i, lis_ctw_widths in enumerate(lis_lis_cwt_widths):
    # Create unique axes for this plot and add to the list
    axes = figure.add_subplot(len(lis_lis_cwt_widths)+1, 1, i+2)
    lis_axes.append(axes)

    # Calculate the CWT image
    arr_cwt = signal.cwt(arr_y_gaussian, signal.ricker, lis_ctw_widths)
    lis_data_ctw.append(arr_cwt)
    imgplot = plt.imshow(arr_cwt)
    lis_img_plots.append(imgplot)

    # Remove the x-ticks for each sub-plot
    axes.set_xticks([])

    # Change the y-axis font size and tick locations
    axes.set_yticks([lis_ctw_widths[0], lis_ctw_widths[-1]])
    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(4)

    # Remove axis lines
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')

    # Get the extreme clim(it) values
    tup_clim = imgplot.get_clim()
    if tup_clim[0] < lis_clim[0]: lis_clim[0] = tup_clim[0]
    if tup_clim[1] > lis_clim[1]: lis_clim[1] = tup_clim[1]

    """
    # Add the lines to the plot
    axes.plot(arr_x, arr_y_gaussian, linewidth=1, label='Gaussian (STD='+str(int_gauss_std)+')')
    axes.plot(arr_x, arr_y_ricker, linewidth=1, linestyle='--', color='gray', label='Ricker (a='+str(int_ricker_a)+')')

    # Add the detected peak/s
    arr_x_gau_cwt_peaks = signal.find_peaks_cwt(arr_y_gaussian, [int_gauss_std])
    arr_y_gau_cwt_peaks = arr_y_gaussian[arr_x_gau_cwt_peaks] # Get the height from the Gaussian)
    axes.plot(arr_x_gau_cwt_peaks, arr_y_gau_cwt_peaks, 'x', label='#'+str(len(arr_x_gau_cwt_peaks))+' peaks for width '+str(lis_ctw_widths)+'', markersize=4, lw=3)

    # Add plot axes labels and title
    axes.set_ylabel('value')
    """

# Normalise the colours
for img_plot in lis_img_plots:
    img_plot.set_clim(lis_clim)

"""
# Plot the colourbar
cax = plt.axes([0.87, 0.05, 0.05, 0.90])
cbar = plt.colorbar(cax=cax)
"""

# Add plot title and x-axes labels
title = lis_axes[0].set_title('Plot of Ricker CWT used on a Gaussian function')
axes.set_xticks([0,200,400,600,800,1000])
axes.set_xlabel('points')

# Change the x-axis font size
for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(4)


# Save the plot (high dpi/resolution)
figure.savefig('gaussian.png', dpi=900)