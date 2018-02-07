# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:51:11 2018

@author: alex_
"""

from scipy.signal.wavelets import cwt, ricker
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Basic imports
import pandas as pd
import numpy as np

# Advanced imports
import flarepy.utils as utils
from sunpy.lightcurve import GOESLightCurve
from sunpy.time import TimeRange



# Some of the intermediate values I want
cwt_dat = False
ridge_lines = False
filtered = False
max_locs = False



def _boolrelextrema(data, comparator,
                  axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.
    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.
    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take 2 numbers as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.  'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default 'clip'.  See numpy.take
    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.
    See also
    --------
    argrelmax, argrelmin
    Examples
    --------
    >>> testdata = np.array([1,2,3,2,1])
    >>> _boolrelextrema(testdata, np.greater, axis=0)
    array([False, False,  True, False, False], dtype=bool)
    """
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    #for shift in xrange(1, order + 1):
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if(~results.any()):
            return results
    return results

def _identify_ridge_lines(matr, max_distances, gap_thresh, verbose=True):
    """
    Identify ridges in the 2-D matrix.
    Expect that the width of the wavelet feature increases with increasing row
    number.
    Parameters
    ----------
    matr : 2-D ndarray
        Matrix in which to identify ridge lines.
    max_distances : 1-D sequence
        At each row, a ridge line is only connected
        if the relative max at row[n] is within
        `max_distances`[n] from the relative max at row[n+1].
    gap_thresh : int
        If a relative maximum is not found within `max_distances`,
        there will be a gap. A ridge line is discontinued if
        there are more than `gap_thresh` points without connecting
        a new relative maximum.
    Returns
    -------
    ridge_lines : tuple
        Tuple of 2 1-D sequences. `ridge_lines`[ii][0] are the rows of the ii-th
        ridge-line, `ridge_lines`[ii][1] are the columns. Empty if none found.
        Each ridge-line will be sorted by row (increasing), but the order
        of the ridge lines is not specified.
    References
    ----------
    Bioinformatics (2006) 22 (17): 2059-2065.
    doi: 10.1093/bioinformatics/btl355
    http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
    Examples
    --------
    >>> data = np.random.rand(5,5)
    >>> ridge_lines = _identify_ridge_lines(data, 1, 1)
    Notes
    -----
    This function is intended to be used in conjunction with `cwt`
    as part of `find_peaks_cwt`.
    """
    if(len(max_distances) < matr.shape[0]):
        raise ValueError('Max_distances must have at least as many rows as matr')

    if verbose:
        print('max_distances: '+str(max_distances))
        print('gap_thresh: '+str(gap_thresh))

    all_max_cols = _boolrelextrema(matr, np.greater, axis=1, order=1)
    #Highest row for which there are any relative maxima
    has_relmax = np.where(all_max_cols.any(axis=1))[0]
    if(len(has_relmax) == 0):
        return []
    start_row = has_relmax[-1]
    #Each ridge line is a 3-tuple:
    #rows, cols,Gap number
    ridge_lines = [[[start_row],
                   [col],
                   0] for col in np.where(all_max_cols[start_row])[0]]
    final_lines = []
    rows = np.arange(start_row - 1, -1, -1)
    cols = np.arange(0, matr.shape[1])
    for row in rows:
        this_max_cols = cols[all_max_cols[row]]

        #Increment gap number of each line,
        #set it to zero later if appropriate
        for line in ridge_lines:
            line[2] += 1

        #XXX These should always be all_max_cols[row]
        #But the order might be different. Might be an efficiency gain
        #to make sure the order is the same and avoid this iteration
        prev_ridge_cols = np.array([line[1][-1] for line in ridge_lines])
        #Look through every relative maximum found at current row
        #Attempt to connect them with existing ridge lines.
        for ind, col in enumerate(this_max_cols):
            """
            If there is a previous ridge line within
            the max_distance to connect to, do so.
            Otherwise start a new one.
            """
            line = None
            if(len(prev_ridge_cols) > 0):
                diffs = np.abs(col - prev_ridge_cols)
                closest = np.argmin(diffs)
                if diffs[closest] <= max_distances[row]:
                    line = ridge_lines[closest]
            if(line is not None):
                #Found a point close enough, extend current ridge line
                line[1].append(col)
                line[0].append(row)
                line[2] = 0
            else:
                new_line = [[row],
                            [col],
                            0]
                ridge_lines.append(new_line)

        #Remove the ridge lines with gap_number too high
        #XXX Modifying a list while iterating over it.
        #Should be safe, since we iterate backwards, but
        #still tacky.
        #for ind in xrange(len(ridge_lines) - 1, -1, -1):
        for ind in range(len(ridge_lines) - 1, -1, -1):
            line = ridge_lines[ind]
            if line[2] > gap_thresh:
                final_lines.append(line)
                del ridge_lines[ind]

    out_lines = []
    for line in (final_lines + ridge_lines):
        sortargs = np.array(np.argsort(line[0]))
        rows, cols = np.zeros_like(sortargs), np.zeros_like(sortargs)
        rows[sortargs] = line[0]
        cols[sortargs] = line[1]
        out_lines.append([rows, cols])

    return out_lines

def _filter_ridge_lines(cwt, ridge_lines, window_size=None, min_length=None,
                       min_snr=1, noise_perc=10, verbose=True):
    """
    Filter ridge lines according to prescribed criteria. Intended
    to be used for finding relative maxima.
    Parameters
    ----------
    cwt : 2-D ndarray
        Continuous wavelet transform from which the `ridge_lines` were defined.

    ridge_lines : 1-D sequence
        Each element should contain 2 sequences, the rows and columns
        of the ridge line (respectively).

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

    References
    ----------
    Bioinformatics (2006) 22 (17): 2059-2065. doi: 10.1093/bioinformatics/btl355
    http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
    """
    # Sanitise input parameters
    num_points = cwt.shape[1]
    if min_length is None:
        min_length = np.ceil(cwt.shape[0] / 4)
    if window_size is None:
        window_size = np.ceil(num_points / 20)
    hf_window = window_size / 2

    if verbose:
        print('\nmin_length: '+str(min_length))
        print('window_size: '+str(window_size))
        print('min_snr: '+str(min_snr))
        print('noise_perc: '+str(noise_perc))

    #Filter based on SNR
    row_one = cwt[0, :]
    noises = np.zeros_like(row_one)
    for ind, val in enumerate(row_one):
        window = np.arange(max([ind - hf_window, 0]), min([ind + hf_window, num_points]))
        window = window.astype(int)
        noises[ind] = scoreatpercentile(row_one[window], per=noise_perc)

    def filt_func(line):
        if len(line[0]) < min_length:
            return False
        snr = abs(cwt[line[0][0], line[1][0]] / noises[line[1][0]])
        if snr < min_snr:
            return False
        return True

    return list(filter(filt_func, ridge_lines))



def modified_find_peaks_cwt(vector, widths, wavelet=None, max_distances=None, gap_thresh=None,
                   window_size=None, min_length=None, min_snr=1, noise_perc=10, verbose=False):
    """
    Attempt to find the peaks in a 1-D array.
    The general approach is to smooth `vector` by convolving it with
    `wavelet(width)` for each width in `widths`. Relative maxima which
    appear at enough length scales, and with sufficiently high SNR, are
    accepted.
    .. versionadded:: 0.11.0
    Parameters
    ----------
    vector : ndarray
        1-D array in which to find the peaks.
    widths : sequence
        1-D array of widths to use for calculating the CWT matrix. In general,
        this range should cover the expected width of peaks of interest.
    wavelet : callable, optional
        Should take a single variable and return a 1-D array to convolve
        with `vector`.  Should be normalized to unit area.
        Default is the ricker wavelet.
    max_distances : ndarray, optional
        At each row, a ridge line is only connected if the relative max at
        row[n] is within ``max_distances[n]`` from the relative max at
        ``row[n+1]``.  Default value is ``widths/4``.
    gap_thresh : float, optional
        If a relative maximum is not found within `max_distances`,
        there will be a gap. A ridge line is discontinued if there are more
        than `gap_thresh` points without connecting a new relative maximum.
        Default is 2.
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
        `stats.scoreatpercentile`.  Default is 10.
    See Also
    --------
    cwt
    Notes
    -----
    This approach was designed for finding sharp peaks among noisy data,
    however with proper parameter selection it should function well for
    different peak shapes.
    The algorithm is as follows:
     1. Perform a continuous wavelet transform on `vector`, for the supplied
        `widths`. This is a convolution of `vector` with `wavelet(width)` for
        each width in `widths`. See `cwt`
     2. Identify "ridge lines" in the cwt matrix. These are relative maxima
        at each row, connected across adjacent rows. See identify_ridge_lines
     3. Filter the ridge_lines using filter_ridge_lines.
    References
    ----------
    .. [1] Bioinformatics (2006) 22 (17): 2059-2065.
        doi: 10.1093/bioinformatics/btl355
        http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
    Examples
    --------
    >>> from scipy import signal
    >>> xs = np.arange(0, np.pi, 0.05)
    >>> data = np.sin(xs)
    >>> peakind = signal.find_peaks_cwt(data, np.arange(1,10))
    >>> peakind, xs[peakind], data[peakind]
    ([32], array([ 1.6]), array([ 0.9995736]))
    """
    # If no gap_thresh is defined then use the maximum width
    if gap_thresh is None:
        gap_thresh = np.ceil(widths[0])
    # If no max_distance is defined then...
    if max_distances is None:
        max_distances = widths / 4.0
    # Set the default wavelet to the ricker wavelet
    if wavelet is None:
        wavelet = ricker

    #
    cwt_dat = cwt(vector, wavelet, widths)

    #
    ridge_lines = _identify_ridge_lines(cwt_dat, max_distances, gap_thresh)

    #
    filtered = _filter_ridge_lines(cwt_dat, ridge_lines, window_size=window_size, min_length=min_length,
                                   min_snr=min_snr, noise_perc=noise_perc)

    #
    max_locs = [x[1][0] for x in filtered]
    return sorted(max_locs), cwt_dat, ridge_lines, filtered


def modified_get_flare_peaks_cwt(ser_data, widths=np.arange(1,100), raw_data=None, ser_minima=None, get_duration=True, get_energies=True,wavelet=None,
                                                                                max_distances=None,
                                                                                gap_thresh=None,
                                                                                window_size=None,
                                                                                min_length=None,
                                                                                min_snr=1,
                                                                                noise_perc=10):
    """
    Implment SciPy CWT to find peaks in the given data.
    Note: input data is expected to be pre-processed (generally resampled and averaged).

    Parameters
    ----------
    ser_data: ~`pandas.Series`
        The dataset to look for flare peaks in.
        Generally GOES XRS B-channel X-Ray data.
        Should be pre-processed to exclude data spikes, data gaps (interpolate),
        rebinned to a common bin width and ideally averaged.

    raw_data: ~`pandas.Series`
        The raw dataset, used for getting the intensity at the time of each peak
        (and thus the flare classification).
        This is generally the un-averaged data, because that'll tend to give
        results closer to the HEK listings.
        Note: defaults to using the ser_data if no raw_data is given.

    widths : (M,) sequence
        The widths to check within the CWT routine.
        See `scipy.signal.cwt` for mor details.

    get_duration : `bool`
        When True the start and end times will be found and then the duraction
        calculated.
        The current implmentation finds the local minima before and after the peak
        and deems these the start and end.
        This must be true if you calculate the energy via numerical integration.

    get_energies : `bool`
        When True use the start/end times and data to use numerical integration
        to interpret the energy detected at the detector.

    Returns
    -------
    result: ~`pandas.DataFrame`
        The table of results, ordered/indexed by peak time.
    """

    # Make the data a pandas.Series if it isn't already
    ser_raw_data = raw_data
    if not isinstance(raw_data, pd.Series):
        ser_raw_data = ser_data

    # Get the peaks
    ###arr_peak_time_indices = signal.find_peaks_cwt(ser_data.values, widths)
    arr_peak_time_indices, cwt_dat, ridge_lines, filtered = modified_find_peaks_cwt(ser_data.values, widths, wavelet=wavelet, max_distances=max_distances, gap_thresh=gap_thresh,
                                                                                    window_size=window_size,
                   min_length=min_length, min_snr=min_snr, noise_perc=noise_perc)
    ser_cwt_peaks = ser_raw_data[arr_peak_time_indices]

    # As a dataframe
    pd_peaks_cwt = pd.DataFrame(data={'fl_peakflux': ser_cwt_peaks})
    pd_peaks_cwt['fl_goescls'] = utils.arr_to_cla(pd_peaks_cwt['fl_peakflux'].values, int_dp=1)
    pd_peaks_cwt['event_peaktime'] = pd_peaks_cwt.index
    pd_peaks_cwt['i_index'] = arr_peak_time_indices

    # Assuming we want the time/energy details
    if get_duration:
        # Get local minima if not given.
        if ser_minima == None:
            ser_minima = ser_data[utils.find_minima_fast(ser_data.interpolate().values)]

        # Now get the star/end time details and add to the DataFrame
        pd_durations = utils.get_flare_start_end_using_min_min(ser_data=ser_data, ser_minima=ser_minima, ser_peaks=ser_cwt_peaks)
        # Add to the original DataFrame
        pd_peaks_cwt = pd.concat([pd_peaks_cwt, pd_durations], axis=1)
        """
        print('\n')
        print(pd_peaks_cwt)
        print('\n')
        """
        # Now get the energies if requested.
        if get_energies:
            pd_energies = utils.get_flare_energy_trap_inte(ser_data, pd_durations['event_starttime'], pd_durations['event_endtime'], pd_peaks_cwt.index)
            # Add to the original DataFrame
            pd_peaks_cwt = pd.concat([pd_peaks_cwt, pd_energies], axis=1)

    # Return the results
    return pd_peaks_cwt, cwt_dat, ridge_lines, filtered



def plot_cwt_components(show=['linear','log','image','ridges','image/ridges'], savepath=None, data=None, peaks=None, cwt_image=None, ridge_lines=None, filtered_ridge_lines=None, title=''):
    """
    Convenience function to make a figure with CWT components.
    """
    # Sanitise the inputs, allowing for arrays or dataframes
    # The line data
    x_line = np.arange(len(data))
    if isinstance(data, pd.core.series.Series):
        y_line = data.values
    elif isinstance(data, np.ndarray):
        y_line = data

    # The peak data
    if isinstance(peaks, pd.core.frame.DataFrame):
        x_peaks = df_peaks_cwt['i_index'].values
        y_peaks = df_peaks_cwt['fl_peakflux'].values
    elif isinstance(peaks, np.ndarray):
        x_peaks = peaks
        y_peaks = y_line[peaks]

    # Two subplots, the axes array is 1-d
    fig, axarr = plt.subplots(len(show), sharex=True)
    count = 0

    # Linear data plots
    if 'linear' in show and not isinstance(data, type(None)):
        # Plot the data line in the top:
        plt_line = axarr[count].plot(x_line, y_line, color='blue', marker='None', linestyle='-')

        # Plot the peaks:
        x_peaks = df_peaks_cwt['i_index'].values
        y_peaks = df_peaks_cwt['fl_peakflux'].values
        plt_peaks = axarr[count].plot(x_peaks, y_peaks, color='green', marker='x', linestyle='None', markersize=5)

        # Show we added a component to the figure
        count = count + 1

    # Log plots
    if 'log' in show:
        # Plot the logged data line:
        plt_line_log = axarr[count].plot(x_line, y_line, color='blue', marker='None', linestyle='-')
        # Plot the logged peaks:
        plt_peaks_log = axarr[count].plot(x_peaks, y_peaks, color='green', marker='x', linestyle='None', markersize=5)
        # Set the scale to logged:
        axarr[count].set_yscale("log")

        # Show we added a component to the figure
        count = count + 1

    # Plot the CWT image
    if 'image' in show:
        plt_img = axarr[count].imshow(cwt_image, origin='lower')#, extent=[x_line[0],x_line[-1],0,100])
        # Trying to strecth vertically, but I can't figure it out:
        #axarr[2].set_ylim((0,100))
        #axarr[2].set_ymargin(0)

        # Show we added a component to the figure
        count = count + 1

    # Plotting the ridge plot
    if ('ridges' in show or 'image/ridges' in show) and not isinstance(ridge_lines, type(None)):
        # Adding all ridge points
        x_all = []
        y_all = []
        for i, ridge_line in enumerate(ridge_lines):
            #print('i: '+str(i))
            for j in range(len(ridge_line[0])):
                #print('    j: '+str(j))
                y_all.append(ridge_lines[i][0][j])
                x_all.append(ridge_lines[i][1][j])

        # Adding the filtered ridge points, those associated with a peak detection
        x_filtered = []
        y_filtered = []
        for i, ridge_line in enumerate(filtered_ridge_lines):
            #print('i: '+str(i))
            for j in range(len(ridge_line[0])):
                #print('    j: '+str(j))
                y_filtered.append(filtered_ridge_lines[i][0][j])
                x_filtered.append(filtered_ridge_lines[i][1][j])

        if 'ridges' in show:
            # Adding these values to the lowest plot:
            axarr[count].plot(x_all, y_all, color='k', marker='.', linestyle='None', markersize=1)
            axarr[count].plot(x_filtered, y_filtered, color='blue', marker='.', linestyle='None', markersize=1)

            # Show we added a component to the figure
            count = count + 1

        if 'image/ridges' in show:
            # Add the image
            axarr[count].imshow(cwt_image, origin='lower')

            # Add the ridge lines
            axarr[count].plot(x_all, y_all, color='k', marker='.', linestyle='None', markersize=1)
            axarr[count].plot(x_filtered, y_filtered, color='blue', marker='.', linestyle='None', markersize=1)

            # Show we added a component to the figure
            count = count + 1

    # Add a title to the whole figure
    axarr[0].set_title(title)

    # Save the figure for viewing
    fig.savefig(savepath, dpi=900)#, bbox_inches='tight')

    return fig



