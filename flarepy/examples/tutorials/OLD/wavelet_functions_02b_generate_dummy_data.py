# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:23:15 2017

@author: alex_
"""

import numpy as np
import flarepy.synthetic_data_generation as gen
import matplotlib.pyplot as plt
from scipy import signal
import os
#import sunpy.timeseries as ts
#import pandas as pd
lis_markers = ['x','+','1','2','3','4','o','v', '^', '<', '>']


###############################################################################
#       Detecting single ricker wavelet using single-value CWT widths         #
###############################################################################
points = 1000 # The width of the total dataset (#values)
tup_ylim = (-0.1,0.2)
for i, in_width in enumerate([30, 50, 70, 90, 110]):
    # The parameters
    parameters = [[1000, in_width]] # points and a-value for each wavelet
    positions = [int(points / 2)] # Central starting position for each wavelet

    # Add the folder if needed
    str_fig_folder = 'single-wavelet__single-width_CWT'
    if not os.path.exists(str_fig_folder):
        try:
            os.stat(str_fig_folder)
        except:
            os.mkdir(str_fig_folder)

    # Make a plot
    str_title_1 = 'Detecting single ricker wavelet using single-value CWT with width '+str(in_width)
    figure_1 = plt.figure()
    axes_1 = figure_1.add_subplot(111)

    # Generate the wavelet function
    arr_data = gen.gen_synthetic_1D_dataset(points, parameters, positions)
    #arr_data = np.absolute(arr_data)
    axes_1.plot(np.arange(0,points), arr_data, label='ricker('+str(parameters[0][0])+', '+str(parameters[0][1])+') at '+str(positions[0]), lw=1)#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])

    # Try detecting the wavelet
    # The (individual) widths to try
    single_widths = [in_width - 20, in_width - 10, in_width, in_width + 10, in_width + 20]

    for j, widths in enumerate(single_widths):
        arr_cwt_det = signal.find_peaks_cwt(arr_data, [widths])
        axes_1.plot(arr_cwt_det, arr_data[arr_cwt_det], lis_markers[j], label='#'+str(len(arr_cwt_det))+' peaks for width ['+str(widths)+']', markersize=5, lw=3)#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])

    # Tweak the display of this plot
    title_1 = axes_1.set_title(str_title_1)
    legend_1 = axes_1.legend()
    if tup_ylim: axes_1.set_ylim(tup_ylim)

    # Save the figure
    #figure_a.show()
    str_save = os.path.join(str_fig_folder,'single-wavelet__single-width_CWT___ricker('+str(parameters[0][0])+', '+str(parameters[0][1])+')___widths_['+str(single_widths[0])+']_['+str(single_widths[1])+']_['+str(single_widths[2])+']_['+str(single_widths[3])+']_['+str(single_widths[4])+']')
    figure_1.savefig(str_save[0:200]+'.png', dpi=900)



###############################################################################
#        Detecting single ricker wavelet using 3-value CWT widths          #
###############################################################################
points = 1000 # The width of the total dataset (#values)
tup_ylim = (-0.1,0.2)
for i, in_width in enumerate([30, 50, 70, 90, 110]):
    # The parameters
    parameters = [[1000, in_width]] # points and a-value for each wavelet
    positions = [int(points / 2)] # Central starting position for each wavelet

    # Add the folder if needed
    str_fig_folder = 'single-wavelet__3-value-widths_CWT'
    if not os.path.exists(str_fig_folder):
        try:
            os.stat(str_fig_folder)
        except:
            os.mkdir(str_fig_folder)

    # Try detecting the wavelet
    # The (individual) widths to try. Using fixed number differences in widths
    lis_widths_1 = [
            [ in_width],
            [in_width - 10, in_width, in_width + 10],
            [in_width - 20, in_width - 10, in_width, in_width + 10, in_width + 20],
            ]
    str_file_prefix_1 = 'single-wavelet__3-value-widths___int-10-increments'
    # The (individual) widths to try. Using proportional differences in widths
    # Thirds
    lis_widths_2 = [
            [ in_width],
            [int(in_width * 0.6666), in_width, int(in_width * 1.3333)],
            [int(in_width * 0.3333), int(in_width * 0.6666), in_width, int(in_width * 1.3333), int(in_width * 1.6666)],
            ]
    str_file_prefix_2 = 'single-wavelet__3-value-widths___0.3333-increments'
    # Quarters
    lis_widths_3 = [
            [ in_width],
            [int(in_width * 0.75), in_width, int(in_width * 1.25)],
            [int(in_width * 0.50), in_width, int(in_width * 1.50)],
            [int(in_width * 0.25), in_width, int(in_width * 1.25)],
            #[int(in_width * 0.50), int(in_width * 0.75), in_width, int(in_width * 1.25), int(in_width * 1.50)],
            #[int(in_width * 0.3333), int(in_width * 0.6666), in_width, int(in_width * 1.3333), int(in_width * 1.6666)],
            ]
    str_file_prefix_3 = 'single-wavelet__3-value-widths___0.2500-increments'
    # Tenths
    lis_widths_4 = [
            [ in_width],
            [int(in_width * 0.9), in_width, int(in_width * 1.1)],
            [int(in_width * 0.8), in_width, int(in_width * 1.2)],
            [int(in_width * 0.7), in_width, int(in_width * 1.3)],
            [int(in_width * 0.6), in_width, int(in_width * 1.4)],
            [int(in_width * 0.5), in_width, int(in_width * 1.5)],
            [int(in_width * 0.4), in_width, int(in_width * 1.6)],
            [int(in_width * 0.3), in_width, int(in_width * 1.7)],
            [int(in_width * 0.2), in_width, int(in_width * 1.8)],
            #[int(in_width * 0.50), int(in_width * 0.75), in_width, int(in_width * 1.25), int(in_width * 1.50)],
            #[int(in_width * 0.3333), int(in_width * 0.6666), in_width, int(in_width * 1.3333), int(in_width * 1.6666)],
            ]
    str_file_prefix_4 = 'single-wavelet__3-value-widths___0.1000-increments'
    # Single value
    lis_widths_5 = [
            [ in_width],
            [in_width - 2, in_width - 1, in_width, in_width + 1, in_width + 2],
            ]
    str_file_prefix_5 = 'single-wavelet__3-value-widths___int-1-increments'
    lis_str_file_prefix = [str_file_prefix_1, str_file_prefix_2, str_file_prefix_3, str_file_prefix_4, str_file_prefix_5 ]
    lis_widths_widths = [lis_widths_1, lis_widths_2, lis_widths_3, lis_widths_4, lis_widths_5]

    for j, lis_widths in enumerate(lis_widths_widths):
        # Get filename prefix
        str_file_prefix = lis_str_file_prefix[j]

        # Make a plot
        str_title_2 = 'Detecting single ricker wavelet using CWT with multiple widths'
        figure_2 = plt.figure()
        axes_2 = figure_2.add_subplot(111)

        # Generate the wavelet function
        arr_data = gen.gen_synthetic_1D_dataset(points, parameters, positions)
        axes_2.plot(np.arange(0,points), arr_data, label='ricker('+str(parameters[0][0])+', '+str(parameters[0][1])+') at '+str(positions[0]), lw=1)#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])

        # For each width
        str_widths = '' # For creating the filename.
        for m, widths in enumerate(lis_widths):
            # Append new width to the filename string
            str_widths = str_widths + '_' + str(widths).replace(' ','')
            arr_cwt_det = signal.find_peaks_cwt(arr_data, widths)
            axes_2.plot(arr_cwt_det, arr_data[arr_cwt_det], lis_markers[m], label='#'+str(len(arr_cwt_det))+' peaks for width '+str(widths)+'', markersize=5, lw=3)#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])

        # Tweak the display of this plot
        title_2 = axes_2.set_title(str_title_2)
        legend_2 = axes_2.legend()
        if tup_ylim: axes_2.set_ylim(tup_ylim)

        # Save the figure
        #figure_a.show()
        str_save = os.path.join(str_fig_folder,str_file_prefix+'___ricker('+str(parameters[0][0])+', '+str(parameters[0][1])+')_CWT_multi-widths_'+str_widths)
        figure_2.savefig(str_save[0:200]+'.png', dpi=900)


###############################################################################
#        Detecting single ricker wavelet using 5-value CWT widths          #
###############################################################################
points = 1000 # The width of the total dataset (#values)
tup_ylim = (-0.1,0.2)
for i, in_width in enumerate([30, 50, 70, 90, 110]):
    # The parameters
    parameters = [[1000, in_width]] # points and a-value for each wavelet
    positions = [int(points / 2)] # Central starting position for each wavelet

    # Add the folder if needed
    str_fig_folder = 'single-wavelet__5-value-widths_CWT'
    if not os.path.exists(str_fig_folder):
        try:
            os.stat(str_fig_folder)
        except:
            os.mkdir(str_fig_folder)

    # Try detecting the wavelet
    # The (individual) widths to try. Using fixed number differences in widths
    lis_widths_1 = [
            [ in_width],
            [in_width - 10, in_width, in_width + 10],
            [in_width - 20, in_width - 10, in_width, in_width + 10, in_width + 20],
            ]
    str_file_prefix_1 = 'single-wavelet__5-value-widths___int-10-increments'
    # The (individual) widths to try. Using proportional differences in widths
    # Thirds
    lis_widths_2 = [
            [ in_width],
            [int(in_width * 0.3333), int(in_width * 0.6666), in_width, int(in_width * 1.3333), int(in_width * 1.6666)],
            ]
    str_file_prefix_2 = 'single-wavelet__5-value-widths___0.3333-increments'
    # Quarters
    lis_widths_3 = [
            [ in_width],
            [int(in_width * 0.50), int(in_width * 0.75), in_width, int(in_width * 1.25), int(in_width * 1.50)],
            #[int(in_width * 0.50), in_width, int(in_width * 1.50)],
            #[int(in_width * 0.25), in_width, int(in_width * 1.25)],
            #[int(in_width * 0.50), int(in_width * 0.75), in_width, int(in_width * 1.25), int(in_width * 1.50)],
            #[int(in_width * 0.3333), int(in_width * 0.6666), in_width, int(in_width * 1.3333), int(in_width * 1.6666)],
            ]
    str_file_prefix_3 = 'single-wavelet__5-value-widths___0.2500-increments'
    # Tenths
    lis_widths_4 = [
            [ in_width],
            [int(in_width * 0.8),int(in_width * 0.9), in_width, int(in_width * 1.1), int(in_width * 1.2)],
            #[int(in_width * 0.8), in_width, int(in_width * 1.2)],
            #[int(in_width * 0.7), in_width, int(in_width * 1.3)],
            #[int(in_width * 0.6), in_width, int(in_width * 1.4)],
            #[int(in_width * 0.5), in_width, int(in_width * 1.5)],
            #[int(in_width * 0.4), in_width, int(in_width * 1.6)],
            #[int(in_width * 0.3), in_width, int(in_width * 1.7)],
            #[int(in_width * 0.2), in_width, int(in_width * 1.8)],
            #[int(in_width * 0.50), int(in_width * 0.75), in_width, int(in_width * 1.25), int(in_width * 1.50)],
            #[int(in_width * 0.3333), int(in_width * 0.6666), in_width, int(in_width * 1.3333), int(in_width * 1.6666)],
            ]
    str_file_prefix_4 = 'single-wavelet__5-value-widths___0.1000-increments'
    # Single value
    lis_widths_5 = [
            [ in_width],
            [in_width - 2, in_width - 1, in_width, in_width + 1, in_width + 2],
            ]
    str_file_prefix_5 = 'single-wavelet__5-value-widths___int-1-increments'
    lis_str_file_prefix = [str_file_prefix_1, str_file_prefix_2, str_file_prefix_3, str_file_prefix_4, str_file_prefix_5 ]
    lis_widths_widths = [lis_widths_1, lis_widths_2, lis_widths_3, lis_widths_4, lis_widths_5]

    for j, lis_widths in enumerate(lis_widths_widths):
        # Get filename prefix
        str_file_prefix = lis_str_file_prefix[j]

        # Make a plot
        str_title_2 = 'Detecting single ricker wavelet using CWT with multiple widths'
        figure_2 = plt.figure()
        axes_2 = figure_2.add_subplot(111)

        # Generate the wavelet function
        arr_data = gen.gen_synthetic_1D_dataset(points, parameters, positions)
        axes_2.plot(np.arange(0,points), arr_data, label='ricker('+str(parameters[0][0])+', '+str(parameters[0][1])+') at '+str(positions[0]), lw=1)#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])

        # For each width
        str_widths = '' # For creating the filename.
        for m, widths in enumerate(lis_widths):
            # Append new width to the filename string
            str_widths = str_widths + '_' + str(widths).replace(' ','')
            arr_cwt_det = signal.find_peaks_cwt(arr_data, widths)
            axes_2.plot(arr_cwt_det, arr_data[arr_cwt_det], lis_markers[m], label='#'+str(len(arr_cwt_det))+' peaks for width '+str(widths)+'', markersize=5, lw=3)#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])

        # Tweak the display of this plot
        title_2 = axes_2.set_title(str_title_2)
        legend_2 = axes_2.legend()
        if tup_ylim: axes_2.set_ylim(tup_ylim)

        # Save the figure
        #figure_a.show()
        str_save = os.path.join(str_fig_folder,str_file_prefix+'___ricker('+str(parameters[0][0])+', '+str(parameters[0][1])+')_CWT_multi-widths_'+str_widths)
        figure_2.savefig(str_save[0:200]+'.png', dpi=900)



"""
###############################################################################
#       Detecting single ricker wavelet using many-value CWT widths           #
###############################################################################
points = 1000
for i, in_width in enumerate([30, 50, 70, 90, 110]):
    # The parameters
    parameters = [[1000, in_width]] # points and a-value for each wavelet
    positions = [int(points / 2)] # Central starting position for each wavelet

    # Make a plot
    figure_a = plt.figure()
    axes_a = figure_a.add_subplot(111)

    # Generate the wavelet function
    arr_data = gen.gen_synthetic_1D_dataset(points, parameters, positions)
    axes_a.plot(np.arange(0,points), arr_data, label='ricker('+str(parameters[0][0])+', '+str(parameters[0][1])+') at '+str(positions[0]), lw=1)#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])

    # Try detecting the wavelet
    # The (individual) widths to try. Using proportional differences in widths
    lis_widths = [
            [ in_width ],
            [int(in_width * 0.6666), in_width, int(in_width * 1.3333)],
            [int(in_width * 0.3333), int(in_width * 0.6666), in_width, int(in_width * 1.3333), int(in_width * 1.6666)],
            ]

    for j, widths in enumerate(lis_widths):
        arr_cwt_det = signal.find_peaks_cwt(arr_data, widths)
        axes_a.plot(arr_cwt_det, arr_data[arr_cwt_det], lis_markers[j], label='#'+str(len(arr_cwt_det))+' peaks for width '+str(widths)+'', markersize=5, lw=3)#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])

    # Display this plot
    legend_a = axes_a.legend()
    #figure_a.show()
    str_save = 'single-wavelet__many-widths___ricker('+str(parameters[0][0])+', '+str(parameters[0][1])+')_CWT_multi-widths_'+str(lis_widths[0])+'_'+str(lis_widths[1])+'_'+str(lis_widths[2])+'.png'
    #figure_a.savefig(str_save, dpi=900)
"""

"""
# To make an array which is the sum of a set of 3 scipy.signal.ricker() functions with given parameters.
length = 1000
parameters = [[500, 20],[800, 30],[1000, 40]] # points and a-value for each wavelet
positions = [50, 300, 650 ] # Central starting position for each wavelet
arr_data = gen.gen_synthetic_1D_dataset(length, parameters, positions)

To plot this:
"""




#axes_a.plot(arr_cwt_det, arr_data[arr_cwt_det], 'x', label='peaks for widths ['+str(widths[0])+', '+str(widths[1])+', ..., '+str(widths[-1])+']')#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])


"""
for a in range(1,3,1):
    arr_wavelet = signal.exponential(points) # y axis values for this wavelet
    axes_a.plot(np.arange(0,points), arr_wavelet, label='exponential('+str(points)+', '+str(a)+')')#), color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])
#plt.plot(arr_wave)
"""
