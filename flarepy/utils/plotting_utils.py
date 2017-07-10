# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:02:52 2017

@author: Alex
"""
import pandas as pd
#import sunpy.timeseries as ts
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sunpy.time import parse_time
import datetime


###############################################################################
#
# Universal plotting functions for convinince
#
###############################################################################

def plot_goes_sat_timelines(h5_data):
    """
    A basic plotting function designed to show the times each GOES satelite were
    in operation during the given dataframe timerange.

    #### Currently doesn't work. ####

    Parameters
    ----------
    data: arr or series or dataframe
        ####

    Returns
    -------
    fig : `~matplotlib.Figure`
        A plot figure.
    """
    # Holding some parameters
    lis_yticks = []
    lis_yticklabels = []
    lis_tup_x_positions = []
    int_ylim_max = 0

    int_count = 0
    for key in h5_data.keys():
        if key != '/meta':
            int_count = int_count + 1

            # Add key
            lis_yticklabels.append(key)
            lis_yticks.append(5 + (10 * int_count))
            lis_tup_x_positions.append((1 + (10 * (int_count-1))))
            int_ylim_max = int_ylim_max + 10


            # Detect long gaps
            arr_td = (h5_data[key].index.values - np.roll(h5_data[key].index.values,1))
            int_time = 3 * (2 * 1000000000)
            arr_boo_long = arr_td > np.timedelta64(int_time,'ns')
            arr_long = np.array(np.where(arr_boo_long)[0])
            #arr_dt_gap_starts = h5_data[key].index[(arr_long - 1)]
            #arr_dt_gap_ends = h5_data[key].index[arr_long]

            arr_dt_ends = h5_data[key].index.values[(arr_long - 1)]
            #arr_dt_ends.
            arr_dt_starts = h5_data[key].index.values[arr_long]



            # Detect 9.99999972e-10 values in xrsa
            arr_boo_low = h5_data[key]['xrsa'].values == 9.99999972e-10
            arr_boo_good = h5_data[key]['xrsa'].values != 9.99999972e-10
            arr_boo_good_plus = np.roll(arr_boo_good,1)
            arr_boo_good_plus[0] = True
            arr_boo_good_minus = np.roll(arr_boo_good,-1)
            arr_boo_good_minus[-1] = True
            arr_boo_low_end = np.logical_and(arr_boo_low, arr_boo_good_minus)
            arr_int_low_ends = np.where(arr_boo_low_end)
            arr_dt_low_ends = h5_data[key].index[arr_int_low_ends]
            arr_boo_start_low = np.logical_and(arr_boo_low, arr_boo_good_plus)
            arr_int_low_starts = np.where(arr_boo_start_low)
            arr_dt_low_starts = h5_data[key].index[arr_int_low_starts]




    fig, ax = plt.subplots()
    ax.broken_barh([(110, 30), (150, 10)], (1, 4), facecolors='blue')
    ax.broken_barh([(110, 30), (150, 10)], (5, 4), facecolors='green')
    ax.broken_barh([(10, 50), (100, 20), (130, 10)], (11, 8),
                   facecolors=('red', 'yellow', 'green'))
    ax.set_ylim(0, 35)
    ax.set_xlim(0, 200)
    ax.set_xlabel('seconds since start')
    ax.set_yticks([5, 15, 25])
    ax.set_yticklabels(['Bill 1', 'Bill 2', 'Jim'])
    ax.grid(True)

    #ax.annotate('race interrupted', (61, 25),
    #            xytext=(0.8, 0.9), textcoords='axes fraction',
    #            arrowprops=dict(facecolor='black', shrink=0.05),
    #            fontsize=16,
    #            horizontalalignment='right', verticalalignment='top')

    plt.show()

    return fig



def plot_basic(data, points=None, savepath=None):
    """
    A basic plotting function designed to take a good variety of datatypes and
    make a simple plot.

    ATM allows the adding of an arbitrary list of datasets and

    Parameters
    ----------
    data: numpy.array or pandas.series or pandas.dataframe
        The data to be plotted.

    Returns
    -------
    fig : `~matplotlib.Figure`
        A plot figure.
    """
    # Data holders
    lis_x = []
    lis_y = []
    lis_col = []
    meta = {}
    lis_str_point_col = ['k','b','m','g','r','c']
    lis_str_point_lin = ['dashed', 'dashdot', 'dotted','dashed', 'dashdot', 'dotted']

    lis_x_points = []

    # Sanatize the input data
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = [ data ]
    if isinstance(data, list):
        for i in range(0,len(data)):
            # Get the data elements
            element = data[i]
            if isinstance(element, pd.DataFrame):
                # strip out the y-axes
                lis_col = element.columns
                for str_col in lis_col:
                    lis_x.append(np.array(element.index))
                    lis_y.append(np.array(element[str_col].values))
            elif isinstance(element, pd.Series):
                lis_x.append(element.index)
                lis_y.append(element.values)
                lis_col.append(str(i) + ' (series)')
    #elif isinstance(data, np.ndarray):

    # Sanatize the points input
    if isinstance(points, pd.DataFrame) or isinstance(points, pd.Series) or (isinstance(points, list) and not isinstance(points[0], list)) or (isinstance(points, np.ndarray) and not isinstance(points[0], np.ndarray)):
        points = [ points ]
    if isinstance(points, list):
        for i in range(0,len(points)):
            # Get the data elements
            element = points[i]
            if isinstance(element, pd.DataFrame):
                # strip out the y-axes
                lis_col = element.columns
                for str_col in lis_col:
                    lis_x_points.append(np.array(element.index))
            elif isinstance(element, pd.Series):
                lis_x_points.append(element.index)
            elif isinstance(element, list):
                lis_x_points.append(np.array(element))
            elif isinstance(element, np.ndarray):
                lis_x_points.append(element)


    # Now make the plots
    fig, ax = plt.subplots()
    lis_lines = []
    for i in range(0,len(lis_x)):
        line = ax.plot(lis_x[i], lis_y[i], '-', linewidth=2,
                       label=lis_col[i])
        lis_lines.append(line)
    # Adding vertical lines for points
    for i in range(0,len(lis_x_points)):
        #plt.vlines(lis_x_points[i],ax.get_ylim()[0],ax.get_ylim()[1],color='k',linestyles='dotted')
        plt.vlines(lis_x_points[i],ax.get_ylim()[0],ax.get_ylim()[1],color=lis_str_point_col[i],linestyles=lis_str_point_lin[i])

    # Add a legend and show the plot
    ax.legend(loc='lower right')
    plt.show()

    return fig

def plot_goes_old(df_data, lis_peaks=None, title="GOES Xray Flux"):
    """Plots GOES XRS light curve is the usual manner. An example is shown
    below.

    .. plot::

        import sunpy.timeseries
        import sunpy.data.sample
        ts_goes = sunpy.timeseries.TimeSeries(sunpy.data.sample.GOES_LIGHTCURVE, source='XRS')
        ts_goes.peek()

    Parameters
    ----------
    title : `str`
        The title of the plot.

    **kwargs : `dict`
        Any additional plot arguments that should be used when plotting.

    Returns
    -------
    fig : `~matplotlib.Figure`
        A plot figure.
    """
    figure = plt.figure()
    axes = plt.gca()

    dates = matplotlib.dates.date2num(parse_time(df_data.index))

    # Adding all the original data
    if isinstance(df_data, pd.DataFrame):
        for str_col in df_data.columns:
            if str_col is 'xrsa':
                axes.plot_date(dates, df_data['xrsa'], '-',
                               label='0.5--4.0 $\AA$', color='blue', lw=1)
            elif str_col is 'xrsb':
                axes.plot_date(dates, df_data['xrsb'], '-',
                             label='1.0--8.0 $\AA$', color='red', lw=1)
            else:
                axes.plot_date(dates, df_data[str_col], '-',
                             label=str_col, color='black', lw=1)
    else:
        axes.plot_date(dates, df_data, '-',
                      label='series', color='blue', lw=2)


    # Adding all peaks
    markers = [ 'x', '+', 's', '*', 'D', '1', '8']
    colours = [ 'k', 'b', 'g', 'r', 'c', 'm', 'y']
    if lis_peaks:
        for i in range(0,len(lis_peaks)):
            df_peaks = lis_peaks[i]
            peak_dates = matplotlib.dates.date2num(parse_time(df_peaks.index))
            axes.plot_date(peak_dates, df_peaks.values, markers[i],
            #              label='series', color=colours[i], markersize=3, lw=2)
            #axes.plot_date(peak_dates, df_peaks.values, '-',
                          label='series', color=colours[i], markersize=5, lw=2)

    # Setup the axes
    axes.set_yscale("log")
    axes.set_ylim(1e-9, 1e-2)
    axes.set_title(title)
    axes.set_ylabel('Watts m$^{-2}$')
    axes.set_xlabel(datetime.datetime.isoformat(df_data.index[0])[0:10])

    ax2 = axes.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(1e-9, 1e-2)
    ax2.set_yticks((1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2))
    ax2.set_yticklabels((' ', 'A', 'B', 'C', 'M', 'X', ' '))

    axes.yaxis.grid(True, 'major')
    axes.xaxis.grid(False, 'major')
    axes.legend()

    # @todo: display better tick labels for date range (e.g. 06/01 - 06/05)
    formatter = matplotlib.dates.DateFormatter('%H:%M')
    axes.xaxis.set_major_formatter(formatter)

    axes.fmt_xdata = matplotlib.dates.DateFormatter('%H:%M')
    figure.autofmt_xdate()
    figure.show()

    return figure




def plot_goes(dic_lines, dic_peaks=None, title="GOES Xray Flux", ylim=(1e-10, 1e-2), xlabel=None, textlabels=None, legncol=3):
    """Plots GOES XRS light curve in the usual manner.
    The basic template was taken from the sunpy.timeseries.XRSTimeSeries class.
    It now additionally adds the ability to add an arbitrary number of lines and sets of marks.

    Parameters
    ----------
    title : `str` ("GOES Xray Flux")
        The title of the plot.

    dic_lines: `dict`
        Contains lines to plot with keys for the names.
        Generally designed to take panda.Series for each line to plot.
        If the name matches a pre-defined one in the function, then the matching
        line visual specifications will be applied.
        Working on allowing dataframes and other objects.


    dic_peaks :
        Contains pandas.Series for each set of marks, with key for the name.
        If the name matches a pre-defined one in the function, then the matching
        mark visual specifications will be applied.

    ylim : `tuple` ((1e-10, 1e-2))
        Manually chance the y-axis limits.

    xlabel : `str` (None)
        Manually chance the x-axis label.

    textlabels :  (None)

    legncol : `int` (3)
        Manually change the number of columns used for the legend.

    **kwargs : `dict`
        Any additional plot arguments that should be used when plotting.

    Returns
    -------
    fig : `~matplotlib.Figure`
        A plot figure.
    """
    figure = plt.figure()
    #plt.subplots_adjust(hspace=0.5)
    #axes = figure.add_subplot(111, adjustable='box', frame_on=True, position=position)#plt.gca()
    #axes = figure.add_subplot(111, adjustable='box')
    axes = figure.add_subplot(111)
    plt.subplots_adjust(wspace=0.4)
    #axes.SubplotParams(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    #print(dir(axes))
    figure.tight_layout()

    # Given a single value (not in dic)
    if not isinstance(dic_lines, dict):
        dic_lines = { '': dic_lines}
    # A dict for the final list of lines
    dic_seperated_lines = {}

    # Adding all the lined data
    for key, value in dic_lines.items():
        # If we have a dataframe then split up into series
        if isinstance(value, pd.DataFrame):
            for str_col in value.columns.values.tolist():
                str_prefix = key
                if len(str_prefix) > 0:
                    str_prefix = str_prefix + ' - '
                dic_seperated_lines[str_prefix + str_col] = value[str_col]
        # Just add a series to the list
        if isinstance(value, pd.Series):
            dic_seperated_lines[key] = value

    # Now plot each line
    # Some parameters for specifc lines
    dic_line_settings = {#                    lw,  col,   line,     alpha,   label,                     markevery, zorder
                         'xrsa':             (0.6 , 'red',  '-', 0.9, 'XRSA: 0.5 — 4.0 $\AA$',           'None', 5),
                         'xrsb':             (0.6 , 'blue', '-', 1.0, 'XRSB: 1.0 — 8.0 $\AA$',           'None', 10),
                         'xrsb - filtered':  (0.6 , 'blue', '-', 1.0, 'XRSB: 1.0 — 8.0 $\AA$ Filtered',  'None', 10),
                         'xrsa - raw':       (1 , 'grey', '-', 0.5, 'XRSA: 1.0 — 8.0 $\AA$ Raw',         'None', 4),
                         'xrsb - raw':       (1 , 'grey', '-', 0.5, 'XRSB: 1.0 — 8.0 $\AA$ Raw',         'None', 9)
                    }

    # Plot each pd.Serial line
    for key, value in dic_seperated_lines.items():
        dates = matplotlib.dates.date2num(parse_time(value.index))
        if key in dic_line_settings:
            conf = dic_line_settings[key]
            axes.plot_date(dates, value.values, conf[2],
                                   label=conf[4], color=conf[1], lw=conf[0], zorder=conf[6])
        else:
            axes.plot_date(dates, value.values, '-',
                           label='series', color='blue', lw=1)


    # Adding all peaks
    # Random parameters
    markers = [ 'x', '+', 's', '*', 'D', '1', '8']
    colours = [ 'k', 'b', 'g', 'r', 'c', 'm', 'y'] # c0, c1, c2, ... MPL colour sequence??
    # Pre-set parameters
    dic_peak_mark_settings = {#                      y_fixed, mark, size, col,  alpha,  label,             zorder
                             'CWT':                (False,   'x' , 6,    'green', 1.0, 'CTW Peaks',        25),
                             'CWT-wlines':          (False,   'x' , 6,    'green', 1.0, 'CTW Peaks',        25),
                             'CWT [1, ..., 10]':   (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 10]', 25),
                             'CWT [1, ..., 20]':   (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 20]', 25),
                             'CWT [1, ..., 30]':  (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 30]', 25),
                             'CWT [1, ..., 40]':  (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 40]', 25),
                             'CWT [1, ..., 50]':   (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 50]', 25),
                             'CWT [1, ..., 60]':   (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 60]', 25),
                             'CWT [1, ..., 70]':   (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 70]', 25),
                             'CWT [1, ..., 80]':  (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 80]', 25),
                             'CWT [1, ..., 90]':  (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 90]', 25),
                             'CWT [1, ..., 100]':   (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 100]', 25),
                             'CWT [1, ..., 150]':   (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 150]', 25),
                             'CWT [1, ..., 200]':  (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 200]', 25),
                             'CWT [1, ..., 300]':  (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 300]', 25),
                             'CWT [1, ..., 400]':  (False,   'x' , 6,    'green', 1.0, 'CWT [1, ..., 400]', 25),
                             'CWT [1, ..., 10]nolab':   (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 20]nolab':   (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 30]nolab':  (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 40]nolab':  (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 50]nolab':   (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 60]nolab':   (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 70]nolab':   (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 80]nolab':  (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 90]nolab':  (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 100]nolab':   (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 150]nolab':   (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 200]nolab':  (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 300]nolab':  (False,   'x' , 6,    'green', 1.0, None, 25),
                             'CWT [1, ..., 400]nolab':  (False,   'x' , 6,    'green', 1.0, None, 25),
                             'HEK':                (False,   '+' , 10,    'red',  1.0, 'HEK Reference Peaks',         21),
                             'HEK-wlines':          (False,   '+' , 10,    'red',  1.0, 'HEK Reference Peaks',         21),
                             'local-max':          (False,   'x' , 8,    'blue',  1.0, 'Local Maxima',         21)
                    }
    dic_peak_line_settings = {#                      ymin,  ymax, colors,  alpha, ls,  lw,  label,        zorder
                             'CWT-wlines':          (None,  None, 'green', 0.5,   '--', 0.5,       '',           25),
                             'HEK-wlines':          (None,  None, 'red',   0.5,   'dotted', 0.5,       '',           25),
                    }

    if dic_peaks:
        # Given a single value (not in dic)
        if not isinstance(dic_peaks, dict) and dic_peaks:
            dic_peaks = { '': dic_peaks}
        # A dict for the final list of peaks
        dic_seperated_peaks = {}

        # Adding all the peak data
        for key, value in dic_peaks.items():
            # If we have a dataframe then split up into series
            if isinstance(value, pd.DataFrame):
                for str_col in value.columns.values.tolist():
                    str_prefix = key
                    if len(str_prefix) > 0:
                        str_prefix = str_prefix + ' - '
                    dic_seperated_peaks[str_prefix + str_col] = value[str_col]
            # Just add a series to the list
            if isinstance(value, pd.Series):
                dic_seperated_peaks[key] = value

        # Plot each pd.Serial peaks
        int_count = 0
        for key, value in dic_seperated_peaks.items():
            # Marks
            if key in dic_peak_mark_settings:
                conf = dic_peak_mark_settings[key]
            else:
                #
                conf = (False, markers[int_count], 5, colours[int_count], 1.0, key, 20)
                int_count = int_count + 1

            # And now plot it
            peak_dates = matplotlib.dates.date2num(parse_time(value.index))
            axes.plot_date(peak_dates, value.values, conf[1],
                          label=conf[5], color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])

            # Lines
            if key in dic_peak_line_settings:
                conf = dic_peak_line_settings[key]
                plt.vlines(peak_dates,-1,1,color=conf[2],linestyles=conf[4],alpha=conf[3],lw=conf[5])

    # Add text bables
    if textlabels:
        for label in textlabels:
            axes.text(label[0], label[1], label[2], fontsize=label[3])

    # Setup the axes
    #axes.set_position([0.1,0.3,0.5,0.5])
    axes.set_yscale("log")
    axes.set_ylim(ylim[0], ylim[1])
    axes.set_title(title)
    axes.set_ylabel('Intensity - Watts m$^{-2}$')
    if xlabel:
        axes.set_xlabel(xlabel)
    #axes.set_xlabel(datetime.datetime.isoformat(df_data.index[0])[0:10])

    # Make ticks (pos and label) for given limits
    dic_flare_class = { 'A':10**-8, 'B':10**-7, 'C':10**-6, 'M':10**-5, 'X':10**-4 }
    lis_ticks = []
    lis_tick_labels = []
    flo_tick = ylim[0]
    while flo_tick <= ylim[1]:
        # Add this tick
        lis_ticks.append(flo_tick)

        # Now look for labels
        boo_found = False
        for key, value in dic_flare_class.items():
            if flo_tick > value * 0.9 and flo_tick < value * 1.1:
                lis_tick_labels.append(key)
                boo_found = True
        if not boo_found:
            lis_tick_labels.append(' ')

        # Next tick
        flo_tick = flo_tick * 10

    # Make second y-axis for flare-class
    ax2 = axes.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(ylim[0], ylim[1])
    #ax2.set_yticks((1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2))
    ax2.set_yticks(lis_ticks)
    ax2.set_yticklabels(lis_tick_labels)#[' ', ' ', 'A', 'B', 'C', 'M', 'X', ' '])
    ax2.set_ylabel('Flare Classification')

    axes.yaxis.grid(True, 'major')
    axes.xaxis.grid(False, 'major')

    # Make legend and position diff if x-axis label given
    tup_leg_bbox = (0.5, -0.15)
    if xlabel:
        tup_leg_bbox = (0.5, -0.2)
    legend = axes.legend(loc=9, bbox_to_anchor=tup_leg_bbox, ncol=legncol, framealpha=1.0)
    legend.set_zorder(20)

    # @todo: display better tick labels for date range (e.g. 06/01 - 06/05)
    formatter = matplotlib.dates.DateFormatter('%H:%M')
    #formatter = matplotlib.dates.DateFormatter('%H:%M')
    axes.xaxis.set_major_formatter(formatter)

    axes.fmt_xdata = matplotlib.dates.DateFormatter('%H:%M')
    figure.autofmt_xdate(rotation=30)

    """
    # set date ticks to something sensible:
    xax = axes.get_xaxis()
    xax.set_major_locator(dates.DayLocator())
    xax.set_major_formatter(dates.DateFormatter('%d/%b'))

    xax.set_minor_locator(dates.HourLocator(byhour=range(0,24,3)))
    xax.set_minor_formatter(dates.DateFormatter('%H'))
    xax.set_tick_params(which='major', pad=15)
    """

    figure.show()

    return figure

"""
Function to make a quick histogram plot for freq vs classification.
"""
def plot_histogram_v01(data, bins=50, savepath=None, title=None, log=False, xlabel=None, ylabel=None, hist_range=None):
    if isinstance(data, np.ndarray):
        arr_data = data
    if isinstance(data, pd.Series):
        arr_data = data.values

    # Create the figure
    plt.figure()

    # Create the histogram of the data
    if not log:
        n, bins, patches = plt.hist(arr_data, bins, normed=1, facecolor='green', alpha=0.75)
    else:
        # Note: negitive values are turned positive before logging
        #n, bins, patches = plt.hist(np.log2(np.absolute(arr_data)), log=True, bins=bins, normed=1, facecolor='red', alpha=0.75)
        n, bins, patches = plt.hist(np.log2(np.absolute(arr_data))[np.log2(np.absolute(arr_data)) > -1E308], log=True, bins=bins, normed=1, facecolor='red', alpha=0.75, range=hist_range)

    # Add x and y axis labels
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('Value')
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Probability')

    # Add title if specified
    if title:
        plt.title(title)

    # Final style tweaks
    plt.grid(True)

    # Save to file if necessary
    if savepath:
        plt.savefig(savepath, dpi=600)

    # Show the plot
    figure.show()

    return figure


