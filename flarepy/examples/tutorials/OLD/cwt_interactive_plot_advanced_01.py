from scipy import signal
from astropy.io import fits
import numpy as np

from cwt_modified_methods_03 import _identify_ridge_lines, _filter_ridge_lines, modified_get_flare_peaks_cwt

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib as mpl

import flarepy.utils as utils

import pandas as pd
import datetime

def getGoesData(dic_dates):
    """
    Function to hold data getting stuff.
    """
    # A variable to store all the given data in
    dic_all_goes_data = {}

    # Get the data for each given date ranges
    for str_date, tup_dates in dic_dates.items():
        # Get the GOES data
        # Specify the start/end times - 2 Options Give
        str_start = tup_dates[0]
        str_end = tup_dates[1]

        # Get data for given days
        df_goes_XRS = utils.get_goes_xrs_data_as_df(str_start, str_end)


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

        # Get the HEK data


        # Add this data into the dictionary
        dic_all_goes_data[str_date] = [ ser_xrsb_raw_int_60S_box5, ser_xrsb_raw ]

    return dic_all_goes_data








def ridge_lines_to_image(lis_ridge_lines):
    """
    Generate a 2-channel image from the ridge lines, designed for saving/loading
    ridge lines to/from FITS files.
    To account for the varied length of each ridge line, we add (0,0) value
    vectors once each ridge line is finished.
    """
    # Get the max length of the ridge lines
    int_length = 1
    for lis_arrays in lis_ridge_lines:
        if len(lis_arrays[0]) > int_length:
            int_length = len(lis_arrays[0])

    # Make an empty image array for the ridge lines
    arr_out = np.zeros((len(lis_ridge_lines), int_length, 2), dtype=np.int64) # [ridge line][point][x or y]
    #arr_out[:] = np.nan

    # Iterate through ridge lines, adding each to the array
    for i, lis_arrays in enumerate(lis_ridge_lines):
        # Getting the coordinate arrays
        arr_x = lis_arrays[1]
        arr_y = lis_arrays[0]

        # Place this ridge line into the output array
        arr_out[i, 0:len(arr_x), 1] = arr_x
        arr_out[i, 0:len(arr_x), 0] = arr_y

    return arr_out

def image_to_ridge_lines(arr_data):
    """
    Get the list of Ridge-Lines from a 2-channel image, designed for saving/loading
    ridge lines to/from FITS files.
    To account for the varied length of each ridge line, we add (0,0) value
    vectors once each ridge line is finished.
    """
    # List to store the ridge lines
    lis_ridge_lines = []

    # Iterate through ridge lines image, adding each to the array
    for lis_arrays in arr_data:
        #
        arr_ridge_line = lis_arrays[np.all(lis_arrays != 0, axis=1)]

        #
        arr_x = arr_ridge_line[:,1]
        arr_y = arr_ridge_line[:,0]

        # Add this ridge line
        lis_ridge_lines.append([arr_y, arr_x])


    return lis_ridge_lines


def generate_datasets(data, wavelet=signal.ricker, widths=np.arange(100)+1.0):
    """

    """
    """
    data = np.arange(100.0) + 1.0 # a simple sequence of floats from 0.0 to 99.9

    # Run CWT on data
    arr_cwt_img = signal.cwt(data, wavelet, widths=widths)

    # Find ridge lines from CWT (currently using defaults)
    gap_thresh = np.ceil(widths[0])
    max_distances = widths / 4.0
    lis_ridge_lines = _identify_ridge_lines(arr_cwt_img, max_distances, gap_thresh)

    # Download HEK entries
    utils.get_hek_goes_flares(str_start, str_end, minimise=True, add_peaks=True, fail_safe=True)

    # Build the CWT convolution image header
    str_wavelet = wavelet.__module__ + '.' + wavelet.__name__
    hdr_cwt = fits.Header({'content':'cwt_image', 'wavelet':str_wavelet})

    # Build the CWT convolution image header
    hdr_ridge_lines = fits.Header({'content':'ridge_lines', 'max_dist':'default', 'gap_thr':'default'})

    # Build the HDUL
    hdu_cwt = fits.PrimaryHDU(arr_cwt_img, hdr_cwt)
    hdu_ridge_lines = fits.ImageHDU(lis_ridge_lines, hdr_ridge_lines)
    hdul = fits.HDUList([hdu_cwt, hdu_ridge_lines])

    # Save the file
    hdul.writeto('new1.fits')
    """
    pass







def makePlot(settings):
    """
    Function to generate the first implementation of the plot.
    """
    # Make the figure
    fig = plt.figure()
    settings['figure'] = fig

    # Make axes for each of the plots
    ax_signal = plt.axes([0.05, 0.65, 0.9, 0.25])
    ax_cwt = plt.axes([0.05, 0.465, 0.9, 0.25])
    ax_signal_thumbnail = plt.axes([0.05, 0.375, 0.9, 0.05])
    settings['axes-signal'] = ax_signal
    settings['axes-cwt'] = ax_cwt
    settings['axes-thumbnail'] = ax_signal_thumbnail

    # Make axes for flare class distribution (i.e. histogram)
    ax_dist = plt.axes([0.6, 0.05, 0.35, 0.3])
    settings['axes-distribution'] = ax_dist

    # Axes (positions) for the sliders
    # For widths
    int_ax_width_height = 0.3
    ax_width_lower = plt.axes([0.15, int_ax_width_height, 0.3, 0.03])
    ax_width_upper = plt.axes([0.50, int_ax_width_height, 0.3, 0.03])
    ax_width_steps = plt.axes([0.85, int_ax_width_height, 0.10, 0.03])
    settings['axes-control-width-lower'] = ax_width_lower
    settings['axes-control-width-upper'] = ax_width_upper
    settings['axes-control-width-steps'] = ax_width_steps

    # For ridge line detection
    ax_ridge_max_dis = plt.axes([0.05, int_ax_width_height - 0.05, 0.15, 0.03])
    ax_ridge_gap_thr = plt.axes([0.30, int_ax_width_height - 0.05, 0.15, 0.03])
    settings['axes-control-ridge-max-dis'] = ax_ridge_max_dis
    settings['axes-control-ridge-gap-thr'] = ax_ridge_gap_thr

    # For ridge line filtering
    ax_filter_min_len = plt.axes([0.05, int_ax_width_height - 0.10, 0.15, 0.03])
    ax_filter_window = plt.axes([0.30, int_ax_width_height - 0.10, 0.15, 0.03])
    ax_filter_min_snr = plt.axes([0.55, int_ax_width_height - 0.10, 0.15, 0.03])
    ax_filter_noise = plt.axes([0.80, int_ax_width_height - 0.10, 0.15, 0.03])
    settings['axes-control-filter-min-len'] = ax_filter_min_len
    settings['axes-control-filter-window'] = ax_filter_window
    settings['axes-control-filter-min-snr'] = ax_filter_min_snr
    settings['axes-control-filter-noise'] = ax_filter_noise

    # Make parameter sliders
    # For widths
    sld_width_lower = Slider(ax_width_lower, 'widths [a:b:c]', 1, 100, valinit=settings.get('cwt-widths',(0,100,1))[0], valfmt='%1.0f')
    sld_width_upper = Slider(ax_width_upper, '', 1, 101, valinit=settings.get('cwt-widths',(0,100,1))[1], valfmt='%1.0f')
    sld_width_steps = Slider(ax_width_steps, '', 1, 25, valinit=settings.get('cwt-widths',(0,100,1))[2], valfmt='%1.0f')
    settings['slider-width-lower'] = sld_width_lower
    settings['slider-width-upper'] = sld_width_upper
    settings['slider-width-steps'] = sld_width_steps

    # For ridge line detection
    sld_ridge_max_dis = Slider(ax_ridge_max_dis, 'max distance', 0, 10, valinit=1, valfmt='%1.0f')
    sld_ridge_gap_thr = Slider(ax_ridge_gap_thr, 'gap threshold', 0, 10, valinit=1, valfmt='%1.0f')
    settings['slider-ridge-max-dis'] = sld_ridge_max_dis
    settings['slider-ridge-gap-thr'] = sld_ridge_gap_thr

    # For ridge line filtering
    sld_filter_min_len = Slider(ax_filter_min_len, 'min_len', 1, 100, valinit=settings['ridge-line-filter-min-length'], valfmt='%1.0f')
    sld_filter_window = Slider(ax_filter_window, 'window', 1, 500, valinit=settings['ridge-line-filter-window-size'], valfmt='%1.0f')
    sld_filter_min_snr = Slider(ax_filter_min_snr, 'min_SNR', 0, 10, valinit=settings['ridge-line-filter-min-snr'])
    sld_filter_noise = Slider(ax_filter_noise, 'noise', 0, 100, valinit=settings['ridge-line-filter-noise_perc'])
    settings['slider-filter-min-len'] = sld_filter_min_len
    settings['slider-filter-window'] = sld_filter_window
    settings['slider-filter-min-snr'] = sld_filter_min_snr
    settings['slider-filter-noise'] = sld_filter_noise

    return settings

"""
def update(i):
    label = 'timestep {0}'.format(i)
    print(label)
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    line.set_ydata(x - 5 + i)
    ax.set_xlabel(label)
    return line, ax
"""

def update(val, clicked, settings):
    """
    The generic function for making updates to the graph.
    """
    # Different function based on the setting/s changed
    if clicked == 'width':
        print('update widths')
    elif clicked == 'ridge':
        print('update ridge detection')
    elif clicked == 'filter':
        print('update ridge filter')

    #print('clicked: '+str(clicked))
    #print('settings: '+str(settings))
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    settings['figure'].canvas.draw_idle()

def drawSignal(settings):
    """

    """
    # Get the axes and clear
    ax = settings['axes-signal']
    ax.clear()

    # Extract the signal and truncate
    ser_data = settings['data'].truncate(*settings['data-truncate'])
    ser_data_raw = settings['data-raw'].truncate(*settings['data-truncate'])
    ser_peaks = settings['data-peaks'].truncate(*settings['data-truncate'])
    print('ser_peaks:\n'+str(ser_peaks))

    # Playing around with indexes to get it right
    x_data = mpl.dates.date2num(ser_data.index.to_pydatetime())
    x_data_raw = mpl.dates.date2num(ser_data_raw.index.to_pydatetime())
    x_data_raw = x_data_raw - x_data[0]
    x_data_raw = (x_data_raw * len(x_data)) / (x_data_raw[-1]-x_data_raw[0])
    x_peaks = mpl.dates.date2num(ser_peaks.index.to_pydatetime())
    x_peaks = x_peaks - x_data[0]
    x_peaks = (x_peaks * len(x_data)) / (x_data[-1]-x_data[0])
    x_data = np.arange(len(x_data))
    print('x_data: ' + str(x_data))

    # Plot the signal and raw signal
    ax.plot(x_data_raw, ser_data_raw.values, color='gray', marker='None', linestyle='-')
    ax.plot(x_data, ser_data.values, color='blue', marker='None', linestyle='-')


    # Remove the x-axis tick labels
    #ax.xaxis.set_ticks_position('none')
    ax.axes.set_xticklabels('')


    # Plot the peaks:
    #if settings.get('data-peaks',False) != False:


    print('x_peaks: ' + str(x_peaks))
    y_peaks = ser_peaks['fl_peakflux'].values
    ax.plot(x_peaks, y_peaks, color='green', marker='*', linestyle='None', markersize=15)

    # Log the plot if asked
    if settings.get('log-signal',False):
        ax.set_yscale("log")

    # Add a title
    #str_title = settings['data-key'] + ' GOES data CWT ricker wavelet widths: ['+str(int_width_lower)+':'+str(int_width_upper)+':'+str(int_width_steps)+']'
    #axarr[0].text(0.5, 1.25, str_title, transform=axarr[0].transAxes, fontsize=24, verticalalignment='top', horizontalalignment ='center')

    # Truncate the plot to only include the given day
    ax.set_xlim(0, len(x_data))
    ax.set_ylim(*settings['signal-ylim'])

    settings['figure'].canvas.draw_idle()

def drawThumbnail(settings):
    """

    """
    # Get the axes and clear
    ax = settings['axes-thumbnail']
    ax.clear()

    # Add the v-span block
    ax.axvspan(mpl.dates.date2num(settings['data-truncate'][0]), mpl.dates.date2num(settings['data-truncate'][1]), facecolor='0.2', alpha=0.2)

    x_data = mpl.dates.date2num(settings['data'].index.to_pydatetime())
    ax.plot(x_data, settings['data'].values, color='blue', marker='None', linestyle='-')

    # Log the plot if asked
    if settings.get('log-signal',False):
        ax.set_yscale("log")

    settings['figure'].canvas.draw_idle()

def drawCWT(settings, imgType = 'imshow'):
    """

    """
    # Clear the axes first
    ax = settings['axes-cwt']
    ax.clear()

    # Add the CWT image
    cwt_image = settings['data-cwt']
    cwt_widths = settings['cwt-widths']

    # Get truncation indices
    int_start = len(settings['data'].truncate(after=settings['data-truncate'][0]))
    int_end = len(settings['data'].truncate(after=settings['data-truncate'][1]))
    print('int_start: '+str(int_start))
    print('int_end: '+str(int_end))

    # Truncate the image
    cwt_image = cwt_image[:,int_start:int_end]

    # If the widths are wider then 1 unit then duplicate the rows to stretch the image
    int_cwt_width_gap = cwt_widths[1] - cwt_widths[0]
    if int_cwt_width_gap > 1:
        # Duplicating the rows
        cwt_image = np.repeat(cwt_image, int_cwt_width_gap, axis=0)

        # Aligning the first row correctly
        #cwt_image_copy = cwt_image_copy[int(int_cwt_width_gap * 0.5) - cwt_widths[0]::]

    # Add the image
    if imgType == 'imshow':
        im = ax.imshow(cwt_image, origin='lower', cmap='PRGn', extent=[0,cwt_image.shape[1],-0.5* int_cwt_width_gap+cwt_widths[0], cwt_widths[-1]+0.5* int_cwt_width_gap+cwt_widths[0]])

    else:
        cmap = plt.get_cmap('PiYG')
        levels = mpl.ticker.MaxNLocator(nbins=150).tick_values(-np.abs(cwt_image).max(), np.abs(cwt_image).max())
        norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        #norm = colors.LogNorm(vmin=-np.abs(cwt_image_copy).max(), vmax=np.abs(cwt_image_copy).max())
        x = np.arange(cwt_image.shape[1])
        y = np.arange(cwt_image.shape[0]) -0.5* int_cwt_width_gap+cwt_widths[0]
        im = ax.pcolormesh(x, y, cwt_image, cmap=cmap, norm=norm, aspect=1)#, offset_position='data', offsets=[50,0,0,0])

    settings['figure'].canvas.draw_idle()


def updateReset(event):
    sfreq.reset()
    samp.reset()

def updateDataSelect(label):
    l.set_color(label)
    fig.canvas.draw_idle()

def onclick(event, settings):
    """
    Function for tracking generic click events.
    Primarily used to track clicks on the mini-map rights now.
    """
    #if event.dblclick:
    #event.button, event.x, event.y, event.xdata, event.ydata

    # Was the thumbnail clicked?
    if event.inaxes is dic_parameters['axes-thumbnail']:
        #print('Thumbnail clicked at x = '+str(event.xdata))
        #update(event, 'thumbnail', dic_parameters={'x':event.xdata})
        dt_selected = mpl.dates.num2date(event.xdata)
        dt_selected = datetime.datetime(dt_selected.year, dt_selected.month, dt_selected.day, dt_selected.hour, dt_selected.minute, dt_selected.second)
        #epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        #dt_selected = (dt_selected - epoch) / datetime.timedelta(seconds=1)
        #dt_selected = pd.Timestamp(mpl.dates.num2date(event.xdata))

        dt_start = settings['data'].index[0]
        dt_end = settings['data'].index[-1]
        print('dt_selected: '+str(dt_selected))
        print('dt_start: '+str(dt_start))
        print('dt_end: '+str(dt_end))

        #
        if dt_selected - 0.5 * settings['data-truncate-duration'] < dt_start:
            dt_end = dt_start + settings['data-truncate-duration']
        elif dt_selected + 0.5 * settings['data-truncate-duration'] > dt_end:
            dt_start = dt_end - settings['data-truncate-duration']
        else:
            dt_start = dt_selected - settings['data-truncate-duration']
            dt_end = dt_selected + settings['data-truncate-duration']

        # Update the truncation settings and redraw
        settings['data-truncate'] = (dt_start, dt_end)
        drawSignal(settings)
        drawThumbnail(settings)
        drawCWT(dic_parameters)



if __name__ == '__main__':

    # The data
    str_savepath = ''
    dic_dates = {'2014-03-28 to 29th': ('2014-03-28 00:00:00', '2014-03-30 00:00:00'),
                 '2012-07-05 to 6th': ('2012-07-05 00:00:00', '2012-07-07 00:00:00'),
                 '2017-09-06': ('2017-09-06 00:00:00', '2017-09-07 00:00:00'),
                 '2017-09-10': ('2017-09-10 00:00:00', '2017-09-11 00:00:00'),
                 '2011-08-09': ('2011-08-09 00:00:00', '2011-08-10 00:00:00')}

    # Check for the data locally


    #


        # Track user clicking




    # Get the data
    dic_data = getGoesData(dic_dates)
    ser_data = dic_data[list(dic_data.keys())[0]][0]
    ser_data_raw = dic_data[list(dic_data.keys())[0]][1]

    # Dictionary to save parameters
    dic_parameters = {'data':dic_data[list(dic_data.keys())[0]][0], 'data-raw':dic_data[list(dic_data.keys())[0]][1], 'data-key':list(dic_data.keys())[0], 'font-size': 16, 'log-signal': True, 'signal-ylim':(10**-10, 10)}
    dic_parameters['data-truncate'] = (ser_data.index[500], ser_data.index[-500])
    dic_parameters['data-truncate-duration'] = datetime.timedelta(days=1)
    # Add CWT parameters
    dic_parameters['cwt-widths'] = (1,100,1)
    # Add ridge-line detection parameters

    # Add ridge-line filter parameters
    int_cwt_widths_len = int((dic_parameters['cwt-widths'][1] - dic_parameters['cwt-widths'][0])/dic_parameters['cwt-widths'][2])
    dic_parameters['ridge-line-filter-min-length'] = np.ceil(int_cwt_widths_len / 4)
    dic_parameters['ridge-line-filter-window-size'] = np.ceil(len(ser_data) / 20)
    dic_parameters['ridge-line-filter-min-snr'] = 1
    dic_parameters['ridge-line-filter-noise_perc'] = 10

    #
    dic_parameters = makePlot(dic_parameters)

    pd_peaks_cwt, cwt_dat, ridge_lines, filtered = modified_get_flare_peaks_cwt(dic_parameters['data'].interpolate(),
                                     widths=np.arange(dic_parameters['cwt-widths'][0],dic_parameters['cwt-widths'][1],dic_parameters['cwt-widths'][2]),# CWT
                                     raw_data=dic_parameters['data-raw'],
                                     ser_minima=None,
                                     get_duration=False,
                                     get_energies=False,
                                     wavelet=None,           # CWT
                                     max_distances=None,     # Ridge-Line Detection
                                     gap_thresh=None,        # Ridge-Line Detection
                                     window_size=None,       # Ridge-Line Filtering
                                     min_length=None,        # Ridge-Line Filtering
                                     min_snr=1,              # Ridge-Line Filtering
                                     noise_perc=10,          # Ridge-Line Filtering
                                     cwt_image=None,          # CWT
                                     ridge_lines=None
                                     )
    print('pd_peaks_cwt: '+ str(pd_peaks_cwt))
    dic_parameters['data-peaks'] = pd_peaks_cwt
    dic_parameters['data-cwt'] = cwt_dat

    #
    drawSignal(dic_parameters)
    drawThumbnail(dic_parameters)
    drawCWT(dic_parameters)






    # Detect slider changes
    dic_parameters['slider-width-lower'].on_changed(lambda x: update(x, 'width', dic_parameters))
    dic_parameters['slider-width-upper'].on_changed(lambda x: update(x, 'width', dic_parameters))
    dic_parameters['slider-width-steps'].on_changed(lambda x: update(x, 'width', dic_parameters))

    dic_parameters['slider-ridge-max-dis'].on_changed(lambda x: update(x, 'ridge', dic_parameters))
    dic_parameters['slider-ridge-gap-thr'].on_changed(lambda x: update(x, 'ridge', dic_parameters))

    dic_parameters['slider-filter-min-len'].on_changed(lambda x: update(x, 'filter', dic_parameters))
    dic_parameters['slider-filter-window'].on_changed(lambda x: update(x, 'filter', dic_parameters))
    dic_parameters['slider-filter-min-snr'].on_changed(lambda x: update(x, 'filter', dic_parameters))
    dic_parameters['slider-filter-noise'].on_changed(lambda x: update(x, 'filter', dic_parameters))











    #fig, ax = plt.subplots()
    #plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    s = a0*np.sin(2*np.pi*f0*t)
    l, = dic_parameters['axes-signal'].plot(t, s, lw=2, color='red')
    plt.axis([0, 1, -10, 10])

    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03])

    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)



    sfreq.on_changed(lambda x: update(x, 'width', dic_parameters))
    samp.on_changed(lambda x: update(x, 'width', dic_parameters))

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')


    button.on_clicked(updateReset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15])
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

    radio.on_clicked(updateDataSelect)

    # Track general click events
    #cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid = dic_parameters['figure'].canvas.mpl_connect('button_press_event', lambda x: onclick(x, dic_parameters))

    plt.show()




