import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from sunpy.lightcurve import GOESLightCurve
from sunpy.time import TimeRange

from cwt_modified_methods_02 import *
from cwt_modified_methods_02 import _filter_ridge_lines

from scipy import signal

from matplotlib.transforms import Bbox

# Configure the global font size
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 25}

import matplotlib
matplotlib.rc('font', **font)

def make_fig():
    pass

def make_signal_axes(ax, signal, signal_x=None, peaks=None, log=True):
    """
    Make a simple line plot with peak crosses for the input signal.
    """
    global str_title
    global int_width_upper
    global int_width_lower
    global int_width_steps

    # Clear the axes first
    ax.clear()

    # If we arn't given signal x-coordinates assume they're just ascending integers
    if isinstance(signal_x, type(None)):
        signal_x = np.arange(len(signal))

    # Plot the signal/data line:
    tempb = ax.plot(signal_x, signal, color='blue', marker='None', linestyle='-')

    print('type(tempb): '+str(type(tempb)))
    print('tempb: '+str(tempb))

    print('type(tempb[0]): '+str(type(tempb[0])))
    print('tempb[0]: '+str(tempb[0]))

    # Remove the x-axis tick labels
    #ax.xaxis.set_ticks_position('none')
    ax.axes.set_xticklabels('')

    # Plot the peaks:
    if not isinstance(peaks, type(None)):
        x_peaks = peaks
        y_peaks = signal[peaks]
        ax.plot(x_peaks, y_peaks, color='green', marker='*', linestyle='None', markersize=15)

    # Log the plot if asked
    if log:
        ax.set_yscale("log")

    # Add a title
    str_title = str_data_key + ' GOES data CWT ricker wavelet widths: ['+str(int_width_lower)+':'+str(int_width_upper)+':'+str(int_width_steps)+']'
    #axarr[0].set_title(title)
    axarr[0].text(0.5, 1.25, str_title, transform=axarr[0].transAxes, fontsize=24, verticalalignment='top', horizontalalignment ='center')

    #
    ax.set_xlim(0, len(signal))


def make_ridge_image_axes(ax, cwt_image=None, ridge_lines=None, filtered_ridge_lines=None, cwt_widths=None):
    """
    Makes an image with the ridge lines plotted on it.
    """
    # Clear the axes first
    ax.clear()

    # Add the CWT image
    if not isinstance(cwt_image, type(None)):
        cwt_image_copy = cwt_image.copy()

        # If the widths are wider then 1 unit then duplicate the rows to stretch the image
        int_cwt_width_gap = cwt_widths[1] - cwt_widths[0]
        if int_cwt_width_gap > 1:
            # Duplicating the rows
            cwt_image_copy = np.repeat(cwt_image_copy, int_cwt_width_gap, axis=0)

            # Aligning the first row correctly
            #cwt_image_copy = cwt_image_copy[int(int_cwt_width_gap * 0.5) - cwt_widths[0]::]

        # Add the image
        global im
        global has_colorbar
        im = ax.imshow(cwt_image_copy, origin='lower', cmap='PRGn', extent=[0,cwt_image.shape[1],-0.5* int_cwt_width_gap+cwt_widths[0], cwt_widths[-1]+0.5* int_cwt_width_gap+cwt_widths[0]])
        if not has_colorbar:
            has_colorbar = True
            plt.colorbar(im, ax=ax, format='$%.5f$', orientation='horizontal', fraction=.1, aspect=55)#, panchor=(0.5,0.4))#shrink=0.5)#, pad=0.15)
        #fig.colorbar(im, cax=ax, format='$%.2f$')#, ticks=t
        #print('cwt_image.shape: '+str(cwt_image.shape))

    # Getting the points for the ridge lines
    x_all = []
    y_all = []
    if not isinstance(ridge_lines, type(None)):
        # Adding all ridge points
        for i, ridge_line in enumerate(ridge_lines):
            #print('i: '+str(i))
            for j in range(len(ridge_line[0])):
                #print('    j: '+str(j))
                y_all.append(ridge_lines[i][0][j])
                x_all.append(ridge_lines[i][1][j])

        # Now translate from width index to actual width number.
        # This is required if the widths are >1 apart.
        y_all = cwt_widths[y_all]

        # Add these to the plot
        ax.plot(x_all, y_all, color='k', marker='.', linestyle='None', markersize=5)

    # The filtered ridge lines
    x_filtered = []
    y_filtered = []
    if not isinstance(filtered_ridge_lines, type(None)):
        # Adding the filtered ridge points, those associated with a peak detection
        for i, ridge_line in enumerate(filtered_ridge_lines):
            #print('i: '+str(i))
            for j in range(len(ridge_line[0])):
                #print('    j: '+str(j))
                y_filtered.append(filtered_ridge_lines[i][0][j])
                x_filtered.append(filtered_ridge_lines[i][1][j])

        # Now translate from width index to actual width number.
        # This is required if the widths are >1 apart.
        y_filtered = cwt_widths[y_filtered]

        # Add these to the plot
        ax.plot(x_filtered, y_filtered, color='blue', marker='.', linestyle='None', markersize=5)

    # Set the x and y limits for this plot
    ax.set_xlim(0, cwt_image.shape[1])
    ax.set_ylim(0, 101)

    # Try and get y-axis ticks corrisponding with widths
    #int_max_ticks = 8
    int_max_ticks = int(np.ceil((8.0/100)*(cwt_widths[-1] - cwt_widths[0])))
    if len(cwt_widths) // int_max_ticks > 0:
        ticks = cwt_widths[::len(cwt_widths) // int_max_ticks]
    else:
        ticks = cwt_widths
    ax.set_yticks(ticks)



def updateData(val):
    """
    """
    pass

def update(val, str_changed):
    """
    """
    # Pull in all the variables
    global dic_data
    global ser_data
    global str_data_key
    global tup_cwt_parameters
    global arr_cwt_widths
    global wavelet
    global max_distances
    global gap_thresh
    global window_size
    global min_length
    global min_snr
    global noise_perc
    global df_peaks_cwt
    global cwt_image
    global ridge_lines
    global filtered
    global int_width_upper
    global int_width_lower
    global int_width_steps


    if str_changed == 'data':
        # Change the data used
        str_data_key = rad_data.value_selected
        ser_data = dic_data[str_data_key]

        # Reset all slider values
        reset(None)

    elif str_changed == 'wavelet':
        # Read in the wavelet radio box value
        str_wavelet = rad_wavelet.value_selected
        if str_wavelet == 'morlet':
            wavelet = signal.morlet
        else:
            wavelet = signal.ricker

    elif str_changed == 'width':
        # Read in the widths slider values
        int_width_upper = int(sld_width_upper.val)
        int_width_lower = int(sld_width_lower.val)
        int_width_steps = int(sld_width_steps.val)

        # Check the lower widths val is smaller then the upper val
        if int_width_upper < int_width_lower:
            int_width_upper = int_width_lower
            sld_width_upper.set_val(int_width_upper)

        # The resulting widths array is
        arr_cwt_widths = np.arange(101)[int_width_lower :int_width_upper:int_width_steps]
        #print('arr_cwt_widths: '+str(arr_cwt_widths))

        # Change the filter min length to the default value
        min_length = np.ceil(len(arr_cwt_widths) / 4)
        sld_filter_min_len.set_val(min_length)

    elif str_changed == 'ridge':
        #max_distances = sld_ridge_max_dis.val
        gap_thresh = sld_ridge_gap_thr.val

    elif str_changed == 'filter':
        window_size = int(sld_filter_window.val)
        min_length = int(sld_filter_min_len.val)
        min_snr = sld_filter_min_snr.val
        noise_perc = sld_filter_noise.val


    # Get the updated CWT peaks and components
    df_peaks_cwt, cwt_image, ridge_lines, filtered = modified_get_flare_peaks_cwt(ser_data.interpolate(),
                                                                                widths=arr_cwt_widths,
                                                                                get_energies=False,
                                                                                wavelet=wavelet,
                                                                                max_distances=max_distances,
                                                                                gap_thresh=gap_thresh,
                                                                                window_size=window_size,
                                                                                min_length=min_length,
                                                                                min_snr=min_snr,
                                                                                noise_perc=noise_perc)

    # Redraw each of the graphs
    make_signal_axes(axarr[0], ser_data, peaks=df_peaks_cwt['i_index'].values, log=True)
    make_ridge_image_axes(axarr[1], cwt_image=cwt_image, ridge_lines=ridge_lines, filtered_ridge_lines=filtered, cwt_widths=arr_cwt_widths)

    #
    fig.canvas.draw_idle()


def update_ridge_filter(val):
    """
    Meant to be the update function when changinf the ridge line filter parameters,
    but it never seemed to work on it's own and it ended up being easier to just
    have a single (slow) update function.


    amp = sld_width_upper.val
    freq = sld_width_lower.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()

    window_size : int, optional
        Size of window to use to calculate noise floor.
        Default is ``cwt.shape[1] / 20``.

    min_length : int, optional
        Minimum length a ridge line needs to be acceptable.
        Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.

    min_snr : float, optional
        Minimum SNR ratio. Default 1. The signal is the value of
        the cwt matrix at the shortest length scale (``cwt[0, loc]``), the
        noise is the `noise_perc`th percentile of datapoints contained within a
        window of `window_size` around ``cwt[0, loc]``.

    noise_perc : float, optional
        When calculating the noise floor, percentile of data points
        examined below which to consider noise. Calculated using
        scipy.stats.scoreatpercentile.
    """
    # Read in the slider values
    window_size = int(sld_filter_window.val)
    min_length = int(sld_filter_min_len.val)
    min_snr = sld_filter_min_snr.val
    noise_perc = sld_filter_noise.val

    # Get the updated CWT peaks and components
    df_peaks_cwt, cwt_image, ridge_lines, filtered = modified_get_flare_peaks_cwt(ser_data.interpolate(),
                                                                                widths=arr_cwt_widths,
                                                                                get_energies=False,
                                                                                wavelet=wavelet,
                                                                                max_distances=max_distances,
                                                                                gap_thresh=gap_thresh,
                                                                                window_size=window_size,
                                                                                min_length=min_length,
                                                                                min_snr=min_snr,
                                                                                noise_perc=noise_perc)

    # Redraw each of the graphs
    make_signal_axes(axarr[0], ser_data, peaks=df_peaks_cwt['i_index'].values, log=True)
    make_ridge_image_axes(axarr[1], cwt_image=cwt_image, ridge_lines=ridge_lines, filtered_ridge_lines=filtered, cwt_widths=arr_cwt_widths)

    """
    # Re-filter the ridges
    filtered = _filter_ridge_lines(cwt_image, ridge_lines, window_size=window_size, min_length=min_length,
                                   min_snr=min_snr, noise_perc=noise_perc)

    # And re-plot
    make_signal_axes(axarr[0], ser_data, peaks=df_peaks_cwt['i_index'].values, log=True)
    make_ridge_image_axes(axarr[1], cwt_image=cwt_image, ridge_lines=ridge_lines, filtered_ridge_lines=filtered, cwt_widths=arr_cwt_widths)
    """

def update_widths(val):
    """
    Meant to be the update function when changing the CWT widths parameters,
    but it never seemed to work on it's own and it ended up being easier to just
    have a single (slow) update function.
    """

    # Read in the slider values
    int_width_upper = int(sld_width_upper.val)
    int_width_lower = int(sld_width_lower.val)
    int_width_steps = int(sld_width_steps.val)

    # Check the lower val is smaller then the upper val
    if int_width_upper < int_width_lower:
        int_width_upper = int_width_lower
        sld_width_upper.set_val(int_width_upper)

    # The resulting widths array is
    arr_cwt_widths = np.arange(101)[int_width_lower :int_width_upper:int_width_steps]
    print('arr_cwt_widths: '+str(arr_cwt_widths))

    # Get the updated CWT peaks and components
    df_peaks_cwt, cwt_image, ridge_lines, filtered = modified_get_flare_peaks_cwt(ser_data.interpolate(),
                                                                                widths=arr_cwt_widths,
                                                                                get_energies=False,
                                                                                wavelet=wavelet,
                                                                                max_distances=max_distances,
                                                                                gap_thresh=gap_thresh,
                                                                                window_size=window_size,
                                                                                min_length=min_length,
                                                                                min_snr=min_snr,
                                                                                noise_perc=noise_perc)

    # Redraw each of the graphs
    make_signal_axes(axarr[0], ser_data, peaks=df_peaks_cwt['i_index'].values, log=True)
    make_ridge_image_axes(axarr[1], cwt_image=cwt_image, ridge_lines=ridge_lines, filtered_ridge_lines=filtered, cwt_widths=arr_cwt_widths)

    #
    #fig.canvas.draw_idle()

def reset(event):
    global ser_data
    global window_size
    global min_length
    global arr_cwt_widths

    # Reset all the widths sliders
    sld_width_lower.reset()
    sld_width_upper.reset()
    sld_width_steps.reset()

    # Reset all the ridge sliders
    sld_ridge_max_dis.reset()
    sld_ridge_gap_thr.reset()

    # Reset all the filter sliders
    sld_filter_min_snr.reset()
    sld_filter_noise.reset()
    #sld_filter_window.reset()
    #sld_filter_min_len.reset()
    window_size=np.ceil(len(ser_data) / 20) #cwt.shape[1] / 20 # np.ceil(num_points / 20) # None
    min_length=np.ceil(len(arr_cwt_widths) / 4)  #cwt.shape[0] / 4 # np.ceil(cwt.shape[0] / 4) # None
    sld_filter_window.set_val(window_size)
    sld_filter_min_len.set_val(min_length)



def get_the_data():
    """
    Function to hold data getting stuff.
    """
    dic_dates = {'2014-03-29': ('2014-03-29 00:00:00', '2014-03-30 00:00:00'),
             '2012-07-06': ('2012-07-06 00:00:00', '2012-07-07 00:00:00'),
             '2017-09-06': ('2017-09-06 00:00:00', '2017-09-07 00:00:00'),
             '2017-09-10': ('2017-09-10 00:00:00', '2017-09-11 00:00:00'),
             '2011-08-09': ('2011-08-09 00:00:00', '2011-08-10 00:00:00')}

    dic_all_data = {}

    # Get the GOES data for each given date range
    for str_date, tup_dates in dic_dates.items():
        # Get the GOES data
        # Specify the start/end times - 2 Options Give
        str_start = tup_dates[0]
        str_end = tup_dates[1]

        lc_goes = GOESLightCurve.create(TimeRange(str_start, str_end))
        df_goes_XRS = lc_goes.data #pd.concat([lc_goes_5th.data, lc_goes_6th.data])


        ############
        #
        #   XRSB Data Pre-Processing
        #
        ############

        # Get raw dataset as a series and make a mask
        ser_xrsb_raw = df_goes_XRS['xrsb']
        ser_xrsb_raw_mask = pd.Series(data=np.logical_or(np.isnan(ser_xrsb_raw.values), ser_xrsb_raw.values == 0.0), index=ser_xrsb_raw.index)
        ser_xrsb_raw_int = ser_xrsb_raw.replace({0.0:np.nan}).interpolate()

        # Resample
        str_bins = '60S'
        ser_xrsb_raw_int_60S = ser_xrsb_raw_int.resample(str_bins).median()
        ser_xrsb_raw_int_60S_mask = ser_xrsb_raw_mask.resample(str_bins).max()

        # Rebin
        int_cart = 5
        ser_xrsb_raw_int_60S_box5 = ser_xrsb_raw_int_60S.rolling(int_cart).mean()
        ser_xrsb_raw_int_60S_box5 = ser_xrsb_raw_int_60S_box5[int_cart - 1: 1- int_cart] # remove NaNs
        ser_xrsb_raw_int_60S_mask = ser_xrsb_raw_int_60S_mask[int_cart - 1: 1- int_cart]
        ser_xrsb_raw_int_60S_box5_int = ser_xrsb_raw_int_60S_box5.interpolate()

        # Make series for plots (basically making nan holes where data gaps were)
        ser_xrsb_plt_fil = pd.Series(ser_xrsb_raw_int_60S_box5)
        ser_xrsb_plt_fil.iloc[np.where(ser_xrsb_raw_int_60S_mask != 0.0)] = np.nan
        ser_xrsb_plt_raw = pd.Series(ser_xrsb_raw)
        ser_xrsb_plt_raw.iloc[np.where(ser_xrsb_raw_mask != 0.0)] = np.nan

        # Add this data into the dictionary
        dic_all_data[str_date] = ser_xrsb_raw_int_60S_box5

    return dic_all_data

global dic_data
global ser_data
global str_data_key
global tup_cwt_parameters
global arr_cwt_widths
global wavelet
global max_distances
global gap_thresh
global window_size
global min_length
global min_snr
global noise_perc
global df_peaks_cwt
global cwt_image
global ridge_lines
global filtered
global has_colorbar
global str_title
global int_width_upper
global int_width_lower
global int_width_steps
has_colorbar = False
# For the plots



if __name__ == '__main__':
    # Get the GOES data and select the first option
    dic_data = get_the_data()
    str_data_key = list(dic_data.keys())[0]
    ser_data = dic_data[str_data_key]

    # Set the CWT default parameters
    # CWT
    tup_cwt_parameters = (1,100,1)
    arr_cwt_widths = np.arange(tup_cwt_parameters[1])[tup_cwt_parameters[0]:tup_cwt_parameters[1]:tup_cwt_parameters[2]]
    int_width_lower, int_width_upper, int_width_steps = tup_cwt_parameters
    wavelet=None
    # Ridge Line Finding
    max_distances=None
    gap_thresh = None
    # Ridge Filter
    window_size=np.ceil(len(ser_data) / 20) #cwt.shape[1] / 20 # np.ceil(num_points / 20) # None
    min_length=np.ceil(len(arr_cwt_widths) / 4)  #cwt.shape[0] / 4 # np.ceil(cwt.shape[0] / 4) # None
    min_snr=1
    noise_perc=10

    # Get the CWT peaks and components
    df_peaks_cwt, cwt_image, ridge_lines, filtered = modified_get_flare_peaks_cwt(ser_data.interpolate(),
                                                                                widths=arr_cwt_widths,
                                                                                get_energies=False,
                                                                                wavelet=wavelet,
                                                                                max_distances=max_distances,
                                                                                gap_thresh=gap_thresh,
                                                                                window_size=window_size,
                                                                                min_length=min_length,
                                                                                min_snr=min_snr,
                                                                                noise_perc=noise_perc)

    fig = plt.figure()
    #fig, ax = plt.subplots()
    #fig, axarr = plt.subplots(2, sharex=True)
    #fig.subplots_adjust(hspace=0)
    ax_signal = plt.axes([0.05, 0.65, 0.9, 0.25])
    ax_cwt = plt.axes([0.05, 0.465, 0.9, 0.25])
    axarr = [ ax_signal, ax_cwt ]

    # Add the initial plots
    make_signal_axes(axarr[0], ser_data, peaks=df_peaks_cwt['i_index'].values, log=True)
    make_ridge_image_axes(axarr[1], cwt_image=cwt_image, ridge_lines=ridge_lines, filtered_ridge_lines=filtered, cwt_widths=arr_cwt_widths)
    #make_signal_axes(ax_signal, ser_data, peaks=peaks, log=True)
    #make_ridge_image_axes(ax_cwt, cwt_image=cwt_image, ridge_lines=ridge_lines, filtered_ridge_lines=filtered_ridge_lines)

    """
    # Position the data plot
    plt.subplots_adjust(left=0.1, bottom=0.30)
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    s = a0*np.sin(2*np.pi*f0*t)
    l, = plt.plot(t, s, lw=2, color='red')
    plt.axis([0, 1, -10, 10])
    """

    axcolor = 'lightgoldenrodyellow'
    #axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    #axamp = plt.axes([0.25, 0.15, 0.65, 0.03])


    # Axes (positions) for the sliders
    int_ax_width_height = 0.3
    ax_width_lower = plt.axes([0.15, int_ax_width_height, 0.3, 0.03])
    ax_width_upper = plt.axes([0.50, int_ax_width_height, 0.3, 0.03])
    ax_width_steps = plt.axes([0.85, int_ax_width_height, 0.10, 0.03])

    ax_ridge_max_dis = plt.axes([0.05, int_ax_width_height - 0.05, 0.15, 0.03])
    ax_ridge_gap_thr = plt.axes([0.30, int_ax_width_height - 0.05, 0.15, 0.03])

    ax_filter_min_len = plt.axes([0.05, int_ax_width_height - 0.10, 0.15, 0.03])
    ax_filter_window = plt.axes([0.30, int_ax_width_height - 0.10, 0.15, 0.03])
    ax_filter_min_snr = plt.axes([0.55, int_ax_width_height - 0.10, 0.15, 0.03])
    ax_filter_noise = plt.axes([0.80, int_ax_width_height - 0.10, 0.15, 0.03])

    # Make parameter sliders
    sld_width_lower = Slider(ax_width_lower, 'widths [a:b:c]', 1, 100, valinit=tup_cwt_parameters[0], valfmt='%1.0f')
    sld_width_upper = Slider(ax_width_upper, '', 1, 101, valinit=tup_cwt_parameters[1], valfmt='%1.0f')
    sld_width_steps = Slider(ax_width_steps, '', 1, 25, valinit=tup_cwt_parameters[2], valfmt='%1.0f')

    sld_ridge_max_dis = Slider(ax_ridge_max_dis, 'max distance', 0, 10, valinit=1, valfmt='%1.0f')
    sld_ridge_gap_thr = Slider(ax_ridge_gap_thr, 'gap threshold', 0, 10, valinit=1, valfmt='%1.0f')

    sld_filter_min_len = Slider(ax_filter_min_len, 'min_len', 1, 100, valinit=min_length, valfmt='%1.0f')
    sld_filter_window = Slider(ax_filter_window, 'window', 1, 500, valinit=window_size, valfmt='%1.0f')
    sld_filter_min_snr = Slider(ax_filter_min_snr, 'min_SNR', 0, 10, valinit=min_snr)
    sld_filter_noise = Slider(ax_filter_noise, 'noise', 0, 100, valinit=noise_perc)

    # Detect slider changes
    sld_width_lower.on_changed(lambda x: update(x, 'width'))
    sld_width_upper.on_changed(lambda x: update(x, 'width'))
    sld_width_steps.on_changed(lambda x: update(x, 'width'))

    sld_ridge_max_dis.on_changed(lambda x: update(x, 'ridge'))
    sld_ridge_gap_thr.on_changed(lambda x: update(x, 'ridge'))

    sld_filter_min_len.on_changed(lambda x: update(x, 'filter'))
    sld_filter_window.on_changed(lambda x: update(x, 'filter'))
    sld_filter_min_snr.on_changed(lambda x: update(x, 'filter'))
    sld_filter_noise.on_changed(lambda x: update(x, 'filter'))

    # Add the reset button
    #resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    resetax = fig.add_axes([0.8, 0.05, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    #print(dir(resetax.get_children()[1]))
    print(resetax.get_children()[1].get_extents())
    button.on_clicked(reset)

    # Add the radio buttons for the wavelet
    ax_rad_wavelet = plt.axes([0.01, 0.01, 0.15, 0.18])
    rad_wavelet = RadioButtons(ax_rad_wavelet, ('ricker', 'morlet'), active=0)
    rad_wavelet.on_clicked(lambda x: update(x, 'wavelet'))

    # Add the radio buttons for the input data selection
    ax_rad_data = plt.axes([0.3, 0.01, 0.15, 0.18])
    rad_data = RadioButtons(ax_rad_data, list(dic_data.keys()), active=0)
    rad_data.on_clicked(lambda x: update(x, 'data'))

    plt.show()

    # Making a plot for the population
    fig_pop = plt.figure()
    ax_pop = plt.axes([0.05, 0.05, 0.9, 0.9])

