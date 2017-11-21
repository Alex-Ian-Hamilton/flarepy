# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:02:52 2017

@author: Alex
"""
import pandas as pd
#import sunpy.timeseries as ts
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
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
    fig : `~mpl.Figure`
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
    ax.broken_barh([(110, 30), (150, 10)], (1, 4), facecolors='C0')
    ax.broken_barh([(110, 30), (150, 10)], (5, 4), facecolors='C2')
    ax.broken_barh([(10, 50), (100, 20), (130, 10)], (11, 8),
                   facecolors=('C3', 'yellow', 'C2'))
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

    dates = mpl.dates.date2num(parse_time(df_data.index))

    # Adding all the original data
    if isinstance(df_data, pd.DataFrame):
        for str_col in df_data.columns:
            if str_col is 'xrsa':
                axes.plot_date(dates, df_data['xrsa'], '-',
                               label='0.5--4.0 $\AA$', color='C0', lw=1)
            elif str_col is 'xrsb':
                axes.plot_date(dates, df_data['xrsb'], '-',
                             label='1.0--8.0 $\AA$', color='C3', lw=1)
            else:
                axes.plot_date(dates, df_data[str_col], '-',
                             label=str_col, color='black', lw=1)
    else:
        axes.plot_date(dates, df_data, '-',
                      label='series', color='C0', lw=2)


    # Adding all peaks
    markers = [ 'x', '+', 's', '*', 'D', '1', '8']
    colours = [ 'k', 'b', 'g', 'r', 'c', 'm', 'y']
    if lis_peaks:
        for i in range(0,len(lis_peaks)):
            df_peaks = lis_peaks[i]
            peak_dates = mpl.dates.date2num(parse_time(df_peaks.index))
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
    formatter = mpl.dates.DateFormatter('%H:%M')
    axes.xaxis.set_major_formatter(formatter)

    axes.fmt_xdata = mpl.dates.DateFormatter('%H:%M')
    figure.autofmt_xdate()
    figure.show()

    return figure



def plot_goes_dayplots(dic_lines, dic_peaks=None, dic_fills=None, title="GOES Xray Flux", ylim=(1e-10, 1e-2), xlabel=None, textlabels=None, legncol=3, miniplot_peaks=None, miniplot_windows=datetime.timedelta(hours=1), miniplot_save_loc=None):
    """
    A method for plotting each days data from a multi-day dataset.
    Mostly follows the interface of plot_goes(), which is what it uses to make the plots.
    Almost exactly the same as plot_goes_miniplots().

    Parameters
    ----------
    title : `str` ("GOES Xray Flux")
        Manually change the title of the plot.

    dic_lines: `dict`
        Contains lines to plot with keys for the names.
        Generally designed to take panda.Series for each line to plot.
        If the name matches a pre-defined one in the function, then the matching
        line visual specifications will be applied.
        Working on allowing dataframes and other objects.


    dic_peaks: `dict`
        Contains pandas.Series for each set of marks, with key for the name.
        If the name matches a pre-defined one in the function, then the matching
        mark visual specifications will be applied.

    dic_fills: `dict`
        Contains pandas.Series or pandas.DataFrame for each region you want to
        fill, with key for the name.
        If the name matches a pre-defined one in the function, then the matching
        fill visual specifications will be applied.
        If a series is given this is assumed to be the upper bound and lower
        bound is assumed to be constant 0.0.
        If a dataframe is given the headings should be "upper" and "lower" or
        the first will be assumed to be the lower.

    ylim : `tuple` ((1e-10, 1e-2))
        Manually change the y-axis limits.

    xlabel : `str` (None)
        Manually chance the x-axis label.

    textlabels : `list` (None)
        A list of text labels, giving the x pos [i][0], y pos [i][1], text [i][2]
        and fontsize [i][3].

    legncol : `int` (3)
        Manually change the number of columns used for the legend.
        The legend is below the x-axis and a width of 3 tends to work well.

    miniplot_peaks : ~`pandas.Series` or ~`pandas.DataFrame` or 'str' (None)
        The x-axis points to define the centre of each miniplot.
        This will likely be a string for the key of one of the entries in
        dic_peaks, bot it can be a completely different pandas DataFrame for
        versalility.

    miniplot_windows : `~datetime.timedelta` (datetime.timedelta(hours=1))
        A timedelta for the half-width of the window you want for each miniplot.
        I may implment a variable with option, but the issue is it make directly
        comparing miniplots harder.

    miniplot_save_loc : `str` (None)
        If you want the miniplots saved (which is the easiest way to view/output)
        then this is the folder to save them into.

    **kwargs : `dict`
        Any additional plot arguments that should be used when plotting.

    Returns
    -------
    fig : `list` of `~matplotlib.Figure`
        A list of the miniplot figures.
    """
    # Sanitize inputs
    if miniplot_peaks is None:
        raise ValueError('Need to specify peak locations for miniplots.')
    if isinstance(miniplot_peaks, str):
        miniplot_peaks = dic_peaks[miniplot_peaks]
    if isinstance(miniplot_peaks, pd.Series):
        # Make contrived start/end times
        arr_window_start = miniplot_peaks.index - 0.5 * miniplot_windows
        arr_window_end = miniplot_peaks.index + 0.5 * miniplot_windows
        dic_data = { 'fl_peakflux': miniplot_peaks.values,
                     'window_starttime': arr_window_start,
                     'window_endtime': arr_window_end}
        miniplot_peaks = pd.DataFrame(data=dic_data, index=miniplot_peaks.index)

    # Get a list of the days (assumes the first line covers the dataset)
    dt_end = parse_time(dic_lines[list(dic_lines.keys())[0]].index[-1])
    dt_start = parse_time(dic_lines[list(dic_lines.keys())[0]].index[0])

    # Start at the beginning of the day
    dt_day = datetime.datetime(dt_start.year, dt_start.month, dt_start.day, tzinfo=dt_end.tzinfo)

    # Add each day to the list until we get to the end day
    lis_dt_days = []
    while dt_day < dt_end:
        lis_dt_days.append(dt_day)
        dt_day = dt_day + datetime.timedelta(days=1)
    lis_dt_days.append(dt_end)

    # Now make each of the plots
    lis_figs = []
    for i in range(1,len(lis_dt_days)):
        dt_start = lis_dt_days[i-1]
        dt_end = lis_dt_days[i]
        str_day = dt_start.strftime("%d/%m/%Y")#str(dt_start.day) + '/' + str(dt_start.month) + '/' + str(dt_start.year)

        #print('dt_start : '+str(dt_start))
        #print('dt_end : '+str(dt_end))

        # Truncate all the data
        dic_lines_trunc = {}
        if not dic_lines is None:
            for key, pandas in dic_lines.items():
                # Truncate the data
                pandas_trunc = pandas.truncate(dt_start, dt_end)

                # Only add to the output if we haven't truncated out all entries
                if len(pandas_trunc) > 0:
                    dic_lines_trunc[key] = pandas_trunc
        dic_peaks_trunc = {}
        if not dic_peaks is None:
            for key, pandas in dic_peaks.items():
                print('key: '+str(key))
                # Truncate the data
                pandas_trunc = pandas.truncate(dt_start, dt_end)

                # Only add to the output if we haven't truncated out all entries
                if len(pandas_trunc) > 0:
                    dic_peaks_trunc[key] = pandas_trunc
        dic_fills_trunc = {}
        if not dic_fills is None:
            for key, pandas in dic_fills.items():
                # Truncate the data
                pandas_trunc = pandas.truncate(dt_start, dt_end)

                # Only add to the output if we haven't truncated out all entries
                if len(pandas_trunc) > 0:
                    dic_fills_trunc[key] = pandas_trunc

        #### Note: not working with text labels

        # Now plot
        #print('dic_lines_trunc:'+str(dic_lines_trunc))
        fig = plot_goes(dic_lines_trunc, dic_peaks=dic_peaks_trunc, dic_fills=dic_fills_trunc, title=title+' - '+str(dt_peak), ylim=(1e-10, 1e-2), xlabel=xlabel, textlabels=None, legncol=legncol)
        lis_figs.append(fig)

        # Save if specified
        if not miniplot_save_loc is None:
            #print('save the minifig')
            fig.savefig(miniplot_save_loc+str(dt_peak).replace(':','-')+'.png', dpi=900, bbox_inches='tight')
    """
        # Truncate all the data
        dic_lines_trunc = {}
        if not dic_lines is None:
            for key, pandas in dic_lines.items():
                # Truncate the data
                pandas_trunc = pandas.truncate(dt_start, dt_end)

                # Only add to the output if we haven't truncated out all entries
                if len(pandas_trunc) > 0:
                    dic_lines_trunc[key] = pandas_trunc
        dic_peaks_trunc = {}
        if not dic_peaks is None:
            for key, pandas in dic_peaks.items():
                # Truncate the data
                pandas_trunc = pandas.truncate(dt_start, dt_end)

                # Only add to the output if we haven't truncated out all entries
                if len(pandas_trunc) > 0:
                    dic_peaks_trunc[key] = pandas_trunc
        dic_fills_trunc = {}
        if not dic_fills is None:
            for key, pandas in dic_fills.items():
                # Truncate the data
                pandas_trunc = pandas.truncate(dt_start, dt_end)

                # Only add to the output if we haven't truncated out all entries
                if len(pandas_trunc) > 0:
                    dic_fills_trunc[key] = pandas_trunc

        #### Note: not working with text labels

        # Now plot
        #print('dic_lines_trunc:'+str(dic_lines_trunc))
        fig = plot_goes(dic_lines_trunc, dic_peaks=dic_peaks_trunc, dic_fills=dic_fills_trunc, title=title+' - '+str_day, ylim=(1e-10, 1e-2), xlabel=xlabel, textlabels=None, legncol=legncol)
        lis_figs.append(fig)

        # Save if specified
        if not miniplot_save_loc is None:
            #print('save the minifig')
            fig.savefig(miniplot_save_loc+str_day.replace(':','-').replace('/','-')+'.png', dpi=900, bbox_inches='tight')

    return lis_figs
    """

def plot_goes_miniplots(dic_lines, dic_peaks=None, dic_fills=None, title="GOES Xray Flux", ylim=(1e-10, 1e-2), xlabel=None, textlabels=None, legncol=3, miniplot_peaks=None, miniplot_windows=datetime.timedelta(hours=1), miniplot_save_loc=None, rtn_figs=False):
    """
    A method for plotting small plots showing the zoom in of each flare.
    Mostly follows the interface of plot_goes(), which is what it uses to make the plots.

    Parameters
    ----------
    title : `str` ("GOES Xray Flux")
        Manually change the title of the plot.

    dic_lines: `dict`
        Contains lines to plot with keys for the names.
        Generally designed to take panda.Series for each line to plot.
        If the name matches a pre-defined one in the function, then the matching
        line visual specifications will be applied.
        Working on allowing dataframes and other objects.


    dic_peaks: `dict`
        Contains pandas.Series for each set of marks, with key for the name.
        If the name matches a pre-defined one in the function, then the matching
        mark visual specifications will be applied.

    dic_fills: `dict`
        Contains pandas.Series or pandas.DataFrame for each region you want to
        fill, with key for the name.
        If the name matches a pre-defined one in the function, then the matching
        fill visual specifications will be applied.
        If a series is given this is assumed to be the upper bound and lower
        bound is assumed to be constant 0.0.
        If a dataframe is given the headings should be "upper" and "lower" or
        the first will be assumed to be the lower.

    ylim : `tuple` ((1e-10, 1e-2))
        Manually change the y-axis limits.

    xlabel : `str` (None)
        Manually chance the x-axis label.

    textlabels : `list` (None)
        A list of text labels, giving the x pos [i][0], y pos [i][1], text [i][2]
        and fontsize [i][3].

    legncol : `int` (3)
        Manually change the number of columns used for the legend.
        The legend is below the x-axis and a width of 3 tends to work well.

    miniplot_peaks : ~`pandas.Series` or ~`pandas.DataFrame` or 'str' (None)
        The x-axis points to define the centre of each miniplot.
        This will likely be a string for the key of one of the entries in
        dic_peaks, bot it can be a completely different pandas DataFrame for
        versalility.

    miniplot_windows : `~datetime.timedelta` (datetime.timedelta(hours=1))
        A timedelta for the half-width of the window you want for each miniplot.
        I may implment a variable with option, but the issue is it make directly
        comparing miniplots harder.

    miniplot_save_loc : `str` (None)
        If you want the miniplots saved (which is the easiest way to view/output)
        then this is the folder to save them into.

    **kwargs : `dict`
        Any additional plot arguments that should be used when plotting.

    rtn_figs : `bool` (False)
        Should this return a list of the figures, defaults of False because this
        can be very RAM hungry if you have more then 100 plots.

    Returns
    -------
    fig : `list` of `~matplotlib.Figure`
        A list of the miniplot figures.
    """
    print('plot_goes_miniplots 00')
    # Sanitize inputs
    if miniplot_peaks is None:
        raise ValueError('Need to specify peak locations for miniplots.')
    if isinstance(miniplot_peaks, str):
        print('isinstance(miniplot_peaks, str)')
        miniplot_peaks = dic_peaks[miniplot_peaks]
    if isinstance(miniplot_peaks, pd.Series):
        print('isinstance(miniplot_peaks, pd.Series)')
        # Make contrived start/end times
        arr_window_start = miniplot_peaks.index - 0.5 * miniplot_windows
        arr_window_end = miniplot_peaks.index + 0.5 * miniplot_windows
        dic_data = { 'fl_peakflux': miniplot_peaks.values,
                     'window_starttime': arr_window_start,
                     'window_endtime': arr_window_end}
        miniplot_peaks = pd.DataFrame(data=dic_data, index=miniplot_peaks.index)

    # Now make each of the plots
    lis_figs = []
    for index, row in miniplot_peaks.iterrows():
        #print('row:\n'+str(row))
        # Get the start/end datetimes
        dt_start = row['window_starttime']
        dt_end = row['window_endtime']
        print('dt_start: '+str(dt_start))
        print('dt_end: '+str(dt_end))
        dt_peak = index
        #print('dt_start : '+str(dt_start))
        #print('dt_end : '+str(dt_end))

        # Truncate all the data
        ### Lines
        dic_lines_trunc = {}
        if not dic_lines is None:
            for key, pandas in dic_lines.items():
                # Truncate the data
                pandas_trunc = pandas.truncate(dt_start, dt_end)

                # Only add to the output if we haven't truncated out all entries
                if len(pandas_trunc) > 0:
                    dic_lines_trunc[key] = pandas_trunc
        ### Peaks
        dic_peaks_trunc = {}
        if not dic_peaks is None:
            print('dic_peaks:\n'+str(dic_peaks)+'\n\n\n')
            for key, pandas in dic_peaks.items():
                print('\nkey:\n'+str(key)+'\n')
                print('\npandas:\n'+str(pandas)+'\n\n\n')
                # Truncate the data
                pandas_trunc = pandas.truncate(dt_start, dt_end)

                # Only add to the output if we haven't truncated out all entries
                if len(pandas_trunc) > 0:
                    dic_peaks_trunc[key] = pandas_trunc
        ### Fills
        dic_fills_trunc = {}
        if not dic_fills is None:
            for key, pandas in dic_fills.items():
                # Truncate the data
                pandas_trunc = pandas.truncate(dt_start, dt_end)

                # Only add to the output if we haven't truncated out all entries
                if len(pandas_trunc) > 0:
                    dic_fills_trunc[key] = pandas_trunc

        #### Note: not working with text labels

        # Now plot
        #print('dic_lines_trunc:'+str(dic_lines_trunc))
        fig = plot_goes(dic_lines_trunc, dic_peaks=dic_peaks_trunc, dic_fills=dic_fills_trunc, title=title+' - '+str(dt_peak), ylim=(1e-10, 1e-2), xlabel=xlabel, textlabels=None, legncol=legncol)

        # Save if specified
        if not miniplot_save_loc is None:
            #print('save the minifig')
            fig.savefig(miniplot_save_loc+str(dt_peak).replace(':','-')+'.png', dpi=900, bbox_inches='tight')

        if rtn_figs:
            lis_figs.append(fig)
        else:
            plt.close()

    return lis_figs


def plot_goes(dic_lines, dic_peaks=None, dic_fills=None, title="GOES Xray Flux", ylim=(1e-10, 1e-2), xlabel=None, textlabels=None, showleg=True, legncol=3):
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


    dic_peaks: `dict`
        Contains pandas.Series for each set of marks, with key for the name.
        If the name matches a pre-defined one in the function, then the matching
        mark visual specifications will be applied.

    dic_fills: `dict`
        Contains pandas.Series or pandas.DataFrame for each region you want to
        fill, with key for the name.
        If the name matches a pre-defined one in the function, then the matching
        fill visual specifications will be applied.
        If a series is given this is assumed to be the upper bound and lower
        bound is assumed to be constant 0.0.
        If a dataframe is given the headings should be "upper" and "lower" or
        the first will be assumed to be the lower.

    ylim : `tuple` ((1e-10, 1e-2))
        Manually chance the y-axis limits.

    xlabel : `str` (None)
        Manually chance the x-axis label.

    textlabels : `list` (None)
        A list of text labels, giving the x pos [i][0], y pos [i][1], text [i][2]
        and fontsize [i][3].

    showleg : `bool` (True)
        Manually control if the legend is shown.

    legncol : `int` (3)
        Manually change the number of columns used for the legend.
        The legend is below the x-axis and a width of 3 tends to work well.

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


    ####################
    # Adding Lines
    ####################
    # Some parameters for specifc lines
    dic_line_settings = {#                    lw,  col,   line,     alpha,   label,                     markevery, zorder
                         'xrsa':             (0.6 , 'C3',  '-', 0.9, 'XRSA: 0.5 — 4.0 $\AA$',           'None', 5),
                         'xrsb':             (0.6 , 'C0', '-', 1.0, 'XRSB: 1.0 — 8.0 $\AA$',           'None', 10),
                         'xrsb - filtered':  (0.6 , 'C0', '-', 1.0, 'XRSB: 1.0 — 8.0 $\AA$ Filtered',  'None', 10),
                         'xrsa - raw':       (1 , 'grey', '-', 0.5, 'XRSA: 1.0 — 8.0 $\AA$ Raw',         'None', 4),
                         'xrsb - raw':       (1 , 'grey', '-', 0.5, 'XRSB: 1.0 — 8.0 $\AA$ Raw',         'None', 9)
                    }

    # Plot each pd.Serial line
    for key, value in dic_seperated_lines.items():
        dates = mpl.dates.date2num(parse_time(value.index))
        if key in dic_line_settings:
            conf = dic_line_settings[key]
            axes.plot_date(dates, value.values, conf[2],
                                   label=conf[4], color=conf[1], lw=conf[0], zorder=conf[6])
        else:
            axes.plot_date(dates, value.values, '-',
                           label='series', color='C0', lw=1)

    ####################
    # Adding Peaks
    ####################

    # Random parameters
    markers = [ 'x', '+', 's', '*', 'D', '1', '8']
    colours = [ 'k', 'b', 'g', 'r', 'c', 'm', 'y'] # c0, c1, c2, ... MPL colour sequence??
    # Pre-set parameters
    dic_peak_mark_settings = {#                      y_fixed, mark, size, col,  alpha,  label,             zorder
                             'CWT':                (False,   'x' , 6,    'C2', 1.0, 'CTW Peaks',        25),
                             'CWT-wlines':          (False,   'x' , 6,    'C2', 1.0, 'CTW Peaks',        25),
                             'CWT [1, ..., 10]':   (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 10]', 25),
                             'CWT [1, ..., 20]':   (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 20]', 25),
                             'CWT [1, ..., 30]':  (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 30]', 25),
                             'CWT [1, ..., 40]':  (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 40]', 25),
                             'CWT [1, ..., 50]':   (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 50]', 25),
                             'CWT [1, ..., 60]':   (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 60]', 25),
                             'CWT [1, ..., 70]':   (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 70]', 25),
                             'CWT [1, ..., 80]':  (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 80]', 25),
                             'CWT [1, ..., 90]':  (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 90]', 25),
                             'CWT [1, ..., 100]':   (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 100]', 25),
                             'CWT [1, ..., 150]':   (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 150]', 25),
                             'CWT [1, ..., 200]':  (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 200]', 25),
                             'CWT [1, ..., 300]':  (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 300]', 25),
                             'CWT [1, ..., 400]':  (False,   'x' , 6,    'C2', 1.0, 'CWT [1, ..., 400]', 25),
                             'CWT [1, ..., 10]nolab':   (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 20]nolab':   (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 30]nolab':  (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 40]nolab':  (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 50]nolab':   (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 60]nolab':   (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 70]nolab':   (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 80]nolab':  (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 90]nolab':  (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 100]nolab':   (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 150]nolab':   (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 200]nolab':  (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 300]nolab':  (False,   'x' , 6,    'C2', 1.0, None, 25),
                             'CWT [1, ..., 400]nolab':  (False,   'x' , 6,    'C2', 1.0, None, 25),
                             '4-min Rise':              (False,   's' , 4,    'C0',  1.0, '4-min Rise',         22),
                             'HEK':                (False,   '+' , 10,    'C3',  1.0, 'HEK Reference Peaks',         21),
                             'HEK-wlines':          (False,   '+' , 10,    'C3',  1.0, 'HEK Reference Peaks',         21),
                             'local-max':          (False,   'x' , 8,    'C0',  1.0, 'Local Maxima',         21)
                    }
    dic_peak_line_settings = {#                      ymin,  ymax, colors,  alpha, ls,  lw,  label,        zorder
                             'CWT-wlines':          (None,  None, 'C2', 0.5,   '--', 0.5,       '',           25),
                             'HEK-wlines':          (None,  None, 'C3',   0.5,   'dotted', 0.5,       '',           25),
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
            peak_dates = mpl.dates.date2num(parse_time(value.index))
            axes.plot_date(peak_dates, value.values, conf[1],
                          label=conf[5], color=conf[3], markersize=conf[2], lw=2, zorder=conf[6])

            # Lines
            if key in dic_peak_line_settings:
                conf = dic_peak_line_settings[key]
                plt.vlines(peak_dates,-1,1,color=conf[2],linestyles=conf[4],alpha=conf[3],lw=conf[5])

    ####################
    # Adding Fills
    ####################

    # Add fills
    if dic_fills:
        for fill in dic_fills:
            arr_upper = fill['upper']
            arr_lower = fill['lower']
            axes.fill_between(fill.index, arr_lower, arr_upper, facecolor='yellow', alpha=0.5, label='a fill region')

    ####################
    # Adding Text Labels - User input text to place on the plot
    ####################
    if textlabels:
        for label in textlabels:
            axes.text(label[0], label[1], label[2], fontsize=label[3])

    ####################
    # Setup the Y axes
    ####################
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

    # Make second y-axis for flare-class (right)
    ax2 = axes.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(ylim[0], ylim[1])
    #ax2.set_yticks((1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2))
    ax2.set_yticks(lis_ticks)
    ax2.set_yticklabels(lis_tick_labels)#[' ', ' ', 'A', 'B', 'C', 'M', 'X', ' '])
    ax2.set_ylabel('Flare Classification')

    axes.yaxis.grid(True, 'major')
    axes.xaxis.grid(False, 'major')

    ####################
    # Setup Legend
    ####################
    if showleg:
        # Make legend and position diff if x-axis label given
        tup_leg_bbox = (0.5, -0.15)
        if xlabel:
            tup_leg_bbox = (0.5, -0.2)
        legend = axes.legend(loc=9, bbox_to_anchor=tup_leg_bbox, ncol=legncol, framealpha=1.0)
        legend.set_zorder(20)


    ####################
    # X-axes
    ####################

    # @todo: display better tick labels for date range (e.g. 06/01 - 06/05)
    formatter = mpl.dates.DateFormatter('%H:%M')
    #formatter = mpl.dates.DateFormatter('%H:%M')
    axes.xaxis.set_major_formatter(formatter)

    axes.fmt_xdata = mpl.dates.DateFormatter('%H:%M')
    figure.autofmt_xdate(rotation=30)

    """
    # Date x-axis (not working right now)
    print('\n\naxes.get_xlim: '+str(axes.get_xlim())+'\n\n')
    dt_start = mpl.dates.num2date(axes.get_xlim()[0])
    dt_end = mpl.dates.num2date(axes.get_xlim()[1])
    dt_day = dt_start
    if (dt_day.hour != 0) or (dt_day.minute != 0) or (dt_day.second != 0) or (dt_day.microsecond != 0):
        dt_day = datetime.datetime(dt_start.year, dt_start.month, dt_start.day, tzinfo=dt_end.tzinfo)
        dt_day = dt_day + datetime.timedelta(days=1)
    lis_dt_days = []
    lis_str_days = []
    while dt_day < dt_end:
        lis_dt_days.append(dt_day)
        lis_str_days.append(str(dt_day.day)+'/'+str(dt_day.month)+'/'+str(dt_day.year))
        dt_day = dt_day + datetime.timedelta(days=1)
    print('\n\n')
    print(lis_dt_days)
    print('\n\n')
    print(lis_str_days)

    #arr_days = mpl.dates.date2num(parse_time(df_peaks_cwt.index))
    ax2.set_xlim(axes.get_xlim())
    ax2.set_xticks(mpl.dates.date2num(lis_dt_days))
    #ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xticklabels(lis_str_days)
    ax2.set_xlabel("Dates")
    """
    """
    offset=0
    labeloffset=20
    second_bottom = mpl.spines.Spine(axes, 'bottom', axes.spines['bottom']._path)
    second_bottom.set_position(('outward', offset))
    axes.spines['second_bottom'] = second_bottom

    axes.annotate('label',
            xy=(0.5, 0), xycoords='axes fraction',
            xytext=(0, -labeloffset), textcoords='offset points',
            verticalalignment='top', horizontalalignment='center')
    """

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


def plot_static_window_stats(df_data, percentage=False):
    """
    A basic plot to show the matched flares, false rejection and acceptaions relative to a reference.
    The reference data is "usually" HEK flares.
    """
    # Plot parameters
    N = len(df_data.columns)
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35
    dic_colours = {'false acceptance':'C0', 'matched':'C2', 'false rejection': 'C3'}
    dic_bottoms = {'false acceptance':0.0, 'matched':df_data['false acceptance'].values, 'false rejection': df_data['matched'].values}


    # Get the bars
    lis_bars = []
    for index, row in df_data.iterrows():
        bottom = dic_bottoms[index]
        bar = plt.bar(ind, row.values, width, color=dic_colours.get(index, '#a0a0a0'), label=index, bottom=bottom)
        lis_bars.append(bar)


    plt.ylabel('Scores')
    plt.title('Scores by group and gender')
    plt.xticks(ind, list(df_data.columns))
    #plt.yticks(np.arange(0, 5, 10,15,20))
    plt.legend()#(p1[0], p2[0]), ('Men', 'Women'))

    plt.show()

def remove_list_duplicates(input):
    """
    """
    lis_out = []
    for element in input:
        if element not in lis_out:
            lis_out.append(element)

    return lis_out


def plot_varied_window_stats(df_data, percentage=False, title='Statistical Correlation of Results to HEK Reference', legncol=3):
    """
    A basic plot to show the matched flares, false rejection and acceptaions relative to a reference.
    The reference data is "usually" HEK flares.

    Parameters
    ----------
    df_data : ~`pandas.DataFrame`
        The data to be ploted, this is a pandas DataFrame with multi-index's.
        It incluse columns for each comparison window and sub-columns for false
        acceptance, matches and false rejection.
        The primay vertical index contains sub-columns for the pre-rocessing,
        detection method and detection parameters.
        This hopefully allows for any reasonable set of results for different
        methods/parameters to be easily plotted.

    percentage : `bool` (False)
        Should the data be plotted as percentages or as absolute numbers of
        flares matched/rejected.

    title : `str` ('Statistical Correlation of Results to HEK Reference')
        Title for the plot

    legncol : `int` (3)
        The number of columns in the legend at the bottom of the plot.

    Returns
    -------
    result : ~`numpy.ndarray`
        The array of indices of the data spikes.
    """
    # Plot parameters
    # lis_str_windows = df_data.columns.levels[0] # Doesn't preserve order
    lis_str_windows = remove_list_duplicates(df_data.columns.get_level_values(0))
    N_rows = len(df_data)
    N_windows = len(lis_str_windows)
    arr_xpos = np.arange(N_rows)    # the x locations for the groups
    all_width = 0.6
    win_width = all_width / N_windows

    # Decide label font-size
    int_label_size = 11 - (N_rows - 5)
    if int_label_size < 1:
        int_label_size = 1

    #dic_colours = {'false acceptance':'C0', 'matched':'C2', 'false rejection': 'C3'}
    #dic_bottoms = {'false acceptance':0.0, 'matched':df_data['false acceptance'].values, 'false rejection': df_data['matched'].values}

    figure = plt.figure()
    #plt.subplots_adjust(hspace=0.5)
    #axes = figure.add_subplot(111, adjustable='box', frame_on=True, position=position)#plt.gca()
    #axes = figure.add_subplot(111, adjustable='box')
    axes = figure.add_subplot(111)
    plt.subplots_adjust(wspace=0.4)
    figure.tight_layout()

    # Get the bars
    # For each window
    lis_flo_pos_window_labels = []
    lis_str_labels_windows = []
    for i in range(0,len(lis_str_windows)):
        # The reduced DataFrame with just that window
        str_win_index = lis_str_windows[i]
        #print('str_win_index: ' + str(str_win_index))
        df_window = df_data[str_win_index]
        #print('df_window:\n' + str(df_window))

        # Get the bar values for each metric (try to make it tolerant of heading variations)
        arr_FA = df_window.get('FA', df_window.get('False Acceptance', df_window.get('false acceptance', df_window.get('False_Acceptance', df_window.get('false_acceptance', df_window.get('fa', 0.0)))))).values
        arr_M = df_window.get('M', df_window.get('m', df_window.get('matched', df_window.get('Matched', df_window.get('Matches', df_window.get('matches', 0.0)))))).values
        arr_FR = df_window.get('FR', df_window.get('False Rejection', df_window.get('false rejection', df_window.get('False_Rejection', df_window.get('false_rejection', df_window.get('fr', 0.0)))))).values

        # Add the bars, include labels if they're the
        if i is 0:
            bars_FA = axes.bar(arr_xpos+i*win_width, arr_FA, win_width, color='C0', bottom=0.0, label='False Acceptance')
            bars_M = axes.bar(arr_xpos+i*win_width, arr_M, win_width, color='C2', bottom=arr_FA, label='Matched')
            bars_FR = axes.bar(arr_xpos+i*win_width, arr_FR, win_width, color='C3', bottom=arr_FA+arr_M, label='False Rejection')
        else:
            bars_FA = axes.bar(arr_xpos+i*win_width, arr_FA, win_width, color='C0', bottom=0.0)
            bars_M = axes.bar(arr_xpos+i*win_width, arr_M, win_width, color='C2', bottom=arr_FA)
            bars_FR = axes.bar(arr_xpos+i*win_width, arr_FR, win_width, color='C3', bottom=arr_FA+arr_M)

        # Add window label details
        lis_flo_pos_window_labels = lis_flo_pos_window_labels + list(arr_xpos+i*win_width)
        lis_str_labels_windows = lis_str_labels_windows + [str_win_index] * (len(arr_xpos) + 1)


    # Now go through each row for labels (which has pre-processing, method and parameters)
    lis_str_labels_windows = []
    for i in range(0,len(df_data['0s'])):
        # Get the row
        row = df_window.iloc[i]

        # Get the text and position for the label
        str_label = str(row.name).replace('\', \'','; ').replace('\'','').replace('(','').replace(')','')
        flo_xpos = arr_xpos[i]

        # Add the label
        plt.text(flo_xpos-0.9*win_width, 0.5, str_label, size=int_label_size, rotation=90.,va="bottom", ha="center")

        # Add
        lis_str_labels_windows = lis_str_labels_windows + lis_str_windows

    axes.set_ylabel('# Of Flares')
    axes.set_title(title)
    #plt.ylim = [0,100]
    #axes.set_ylim([ymin,ymax])
    ####axes.set_ylim([0,100])
    ####print('lis_flo_pos_window_labels: ' + str(lis_flo_pos_window_labels))
    ####print('lis_str_labels_windows: ' + str(lis_str_labels_windows))
    plt.xticks(sorted(lis_flo_pos_window_labels), lis_str_labels_windows, rotation=90, fontsize=int_label_size)
    #plt.yticks(np.arange(0, 5, 10,15,20))
    ######legend = axes.legend()#(p1[0], p2[0]), ('Men', 'Women'))
    tup_leg_bbox = (0.5, -0.15)
    legend = axes.legend(loc=9, bbox_to_anchor=tup_leg_bbox, ncol=legncol, framealpha=1.0)
    legend.set_zorder(20)

    figure.show()

    return figure




def plot_histogram_v01(data, bins=50, savepath=None, title=None, log=False, xlabel=None, ylabel=None, hist_range=None):
    """
    Function to make a quick histogram plot for freq vs classification.

    Parameters
    ----------
    data : ~`pandas.DataFrame`
        #

    bins : `int` (50)
        #

    title : `str` (None)
        Title for the plot

    xlabel : `str` (None)
        X-axis label.

    ylabel : `str` (None)
        Y-axis label.

    savepath : `str` (None)
        The file name and path to save the resulting plot.

    log : `bool` (False)
        Option to plot the log values.

    hist_range : `` (None)
        #

    Returns
    -------
    result : ~`matplotlib.figure.Figure`
        The array of indices of the data spikes.
    """
    if isinstance(data, np.ndarray):
        arr_data = data
    if isinstance(data, pd.Series):
        arr_data = data.values

    # Create the figure and axes
    fig = plt.figure()
    axes = fig.add_subplot(111)

    # Create the histogram of the data
    if not log:
        # For non-log plots
        n, bins, patches = plt.hist(arr_data, bins, normed=1, facecolor='C2', alpha=0.75)
    else:
        # Note: negative values are turned positive before logging
        #n, bins, patches = plt.hist(np.log2(np.absolute(arr_data)), log=True, bins=bins, normed=1, facecolor='C3', alpha=0.75)
        n, bins, patches = plt.hist(np.log2(np.absolute(arr_data))[np.log2(np.absolute(arr_data)) > -1E308], log=True, bins=bins, normed=1, facecolor='C3', alpha=0.75, range=hist_range)

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
    fig.show()

    return fig


def plot_histogram_v02(data, bins=50, savepath=None, title=None, log=False, xlabel=None, ylabel=None, hist_range=None, zero_start=True):
    """
    Function to make a quick histogram plot for freq vs classification.

    Parameters
    ----------
    data : ~`pandas.DataFrame`
        #

    bins : `int` (50)
        #

    title : `str` (None)
        Title for the plot

    xlabel : `str` (None)
        X-axis label.

    ylabel : `str` (None)
        Y-axis label.

    savepath : `str` (None)
        The file name and path to save the resulting plot.

    log : `bool` (False)
        Option to plot the log values.

    hist_range : `` (None)
        #

    Returns
    -------
    result : ~`matplotlib.figure.Figure`
        The array of indices of the data spikes.
    """
    # Sanatise input
    if isinstance(data, np.ndarray):
        arr_data = data
    if isinstance(data, pd.Series):
        arr_data = data.values

    flo_start = arr_data.min()
    if zero_start:
        flo_start = 0.0

    # Make the figure and axes
    fig = plt.figure()
    #plt.subplots_adjust(hspace=0.5)
    #axes = figure.add_subplot(111, adjustable='box', frame_on=True, position=position)#plt.gca()
    #axes = figure.add_subplot(111, adjustable='box')
    axes = fig.add_subplot(111)
    #plt.subplots_adjust(wspace=0.4)
    fig.tight_layout()


    # Decide label font-size
    int_label_size = 11 - (bins - 5)
    if int_label_size < 1:
        int_label_size = 1

    #
    flo_bin_width = (arr_data.max() - flo_start)/bins
    arr_bins = np.arange(flo_start, arr_data.max(), flo_bin_width)
    arr_bins = np.append(arr_bins, arr_data.max())

    tup_hist = np.histogram(arr_data, bins=arr_bins)
    print('tup_hist[0]'+str(len(tup_hist[0]))+': ' + str(tup_hist[0]))
    print('tup_hist[1]'+str(len(tup_hist[1]))+': ' + str(tup_hist[1]))
    """
    arr_distribution = []
    for i in range(1,arr_bins):
        arr_distribution.append()
    """
    # Make bars
    flo_bar_width = flo_bin_width
    print('flo_bar_width: ' + str(flo_bar_width))
    axes.bar(tup_hist[1][:-1]+(flo_bar_width/2.0), tup_hist[0], flo_bar_width, color='C0', bottom=0.0, label='False Acceptance')

    #
    #print('axes.get_xlim(): '+str(axes.get_xlim()))
    #axes.set_xlim((flo_start, tup_hist[1][-1]))
    axes.set_xlim((flo_start, arr_data.max()))

    #dic_colours = {'false acceptance':'C0', 'matched':'C2', 'false rejection': 'C3'}
    #dic_bottoms = {'false acceptance':0.0, 'matched':df_data['false acceptance'].values, 'false rejection': df_data['matched'].values}

    """
    # Create the figure and axes
    fig = plt.figure()
    axes = fig.add_subplot(111)

    # Create the histogram of the data
    if not log:
        # For non-log plots
        n, bins, patches = plt.hist(arr_data, bins, normed=1, facecolor='C2', alpha=0.75)
    else:
        # Note: negative values are turned positive before logging
        #n, bins, patches = plt.hist(np.log2(np.absolute(arr_data)), log=True, bins=bins, normed=1, facecolor='C3', alpha=0.75)
        n, bins, patches = plt.hist(np.log2(np.absolute(arr_data))[np.log2(np.absolute(arr_data)) > -1E308], log=True, bins=bins, normed=1, facecolor='C3', alpha=0.75, range=hist_range)

    """
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
    """
    # Final style tweaks
    plt.grid(True)
    """
    # Save to file if necessary
    if savepath:
        plt.savefig(savepath, dpi=600)

    # Show the plot
    fig.show()

    return fig