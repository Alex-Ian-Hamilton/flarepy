# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:28:04 2017

@author: Alex
"""

import flarepy.utils as utils
import numpy as np
import pandas as pd


def get_flare_start_indices_by_consecutive_rises_np(arr_in, N=20, start_threshold=0.4):
    """
    For GOES HEK, a flare is detected when:
        1. there are four consecutive minutes of increasing flux;
        2. the flux at the end of the fourth minute is at least 40%
           greater than the flux in the first minute.
    This method uses numpy primatives to try and find these values efficiently.

    You can adjust the threshold from 0.4 (40%) to other values and adjust the
    consecutive required increases.

    Parameters
    ----------
    arr_in : array
        The dataset to look for flare peaks in.
        Generally GOES XRS B-channel X-Ray data.
        Should be pre-processed to exclude data spikes, data gaps (interpolate),
        rebinned to a common bin width and ideally averaged.

    N : `int`
        The number of consecutive rises that would constitute the start of a flare.
        Generally you want 4 minutes, so for 12s cadence N=20 and for 60s cadence
        n=4.

    start_threshold : `float`
        The required proportional rise over the N data readings that consitutes
        the start of a flare.

    Returns
    -------
    result : array
        The list of indices of all flare peak points.
    """
    # Find the change of value over the last N cells and divide by the
    arr_flo_diff_20 = (arr_in - np.roll(arr_in,N))
    # Get these as a proportion relative to the initial value
    arr_flo_diff_20_prop = arr_flo_diff_20 / np.roll(arr_in,N)

    # Has the change of value passed the given threshold
    arr_boo_above_threshold = arr_flo_diff_20_prop >= start_threshold

    # What cells have increasing value
    arr_boo_increasing = np.diff(arr_in) > 0

    # Rolling arrays of increasing
    lis_rolling_increasing_arrays = [arr_boo_increasing]
    for i in range(1,N):
        # Add each
        lis_rolling_increasing_arrays.append(np.roll(arr_boo_increasing,i))

    # From the combination of the rolling increasing arrays we have if increasing for N
    arr_boo_increasing_for_N = np.logical_and.reduce(lis_rolling_increasing_arrays)
    arr_boo_incr_N_and_thresh = np.logical_and(arr_boo_increasing_for_N,arr_boo_above_threshold[:-1])

    # Find the difference of this so we only get the start/ends of the chains
    arr_boo_diff__incr_N_and_thresh = np.diff(arr_boo_incr_N_and_thresh)

    # Check if we have the start of a chain (not the end)
    arr_bool_pos = np.logical_and(arr_boo_incr_N_and_thresh[1:], arr_boo_diff__incr_N_and_thresh)

    # Need to get the index positions of these re-map for lost start cells from diff
    # then subtract N to the start of the N-consecutive increases
    arr_int_start = np.nonzero(arr_bool_pos)[0] + 1 - N

    """
    # For testing the output from each step.
    df_temp = pd.DataFrame(data={'datetime':arr_index,
                                 'xrsb':arr_in,
                                 'arr_flo_diff_20': arr_flo_diff_20,
                                 'arr_flo_diff_20_prop': arr_flo_diff_20_prop,
                                 'arr_boo_above_threshold': arr_boo_above_threshold,
                                 'arr_boo_increasing': np.insert(arr_boo_increasing, 0, np.nan),
                                 'arr_boo_increasing_for_N': np.insert(arr_boo_increasing_for_N, 0, np.nan),
                                 'arr_boo_incr_N_and_thresh': np.insert(arr_boo_incr_N_and_thresh, 0, np.nan),
                                 'arr_bool_pos': np.insert(arr_bool_pos, 0, [np.nan, np.nan])})
    df_temp.to_csv('c:\\temp.csv')
    """
    return arr_int_start


def find_end_indices_and_flux(arr_data, arr_peak_indices, arr_end_threshold):
    """
    Given the data, the peak indices and the end threshold (below which the flare stops)
    as arrays this function will find the index and flux of all ends.

    Note: if the last flare never reaches the end threshold within the data then the
    last index and data value is given as the end of that flare.

    Note: I have added the ability to ignore NaN values (for missing data), this is
    not necessary for data without gaps, such as if it's been interpolated.

    Parameters
    ----------
    arr_data : array
        The dataset to look for flare ends in.
        Generally GOES XRS B-channel X-Ray data.
        Should be pre-processed to exclude data spikes, data gaps (interpolate),
        rebinned to a common bin width and ideally averaged.

    arr_peak_indices : array
        The array in i indices for the peaks in the data.

    arr_end_threshold : array
        The array of absolute (not proportional) end threshold for each given peak.

    Returns
    -------
    result : array
        The list of indices of all flare peak points.
    """
    arr_int_end_index = np.empty(arr_peak_indices.shape, dtype=int)
    arr_flo_end_flux = np.empty(arr_peak_indices.shape, dtype=int)
    # For each flare detection (index i)
    for i in range(0,len(arr_peak_indices)):
        # Get the flux threshold to return.
        end_index = arr_peak_indices[i]
        end_flux = arr_data[end_index]
        end_flux_threshold = arr_end_threshold[i]

        # Keep stepping through the dataset after the peak until we drop below the threshold.
        while (end_flux > end_flux_threshold) and (end_index < arr_data.shape[0] - 1):
            end_index = end_index + 1
            if not np.isnan(arr_data[end_index]):
                end_flux = arr_data[end_index]
        arr_int_end_index[i] = end_index
        arr_flo_end_flux[i] = end_flux

    return arr_int_end_index, arr_flo_end_flux


def get_flares_goes_event_list(ser_data, N=20, start_threshold=0.4, end_threshold=0.5, raw_data=None, get_duration=True, get_energies=True):
    """
    Method to return a DataFrame of all the flares within a given region.
    Also gives metadata for the parameters used.

    Parameters
    ----------
    arr_in : array
        The dataset to look for flare peaks in.
        Generally GOES XRS B-channel X-Ray data.
        Should be pre-processed to exclude data spikes, data gaps (interpolate),
        rebinned to a common bin width and ideally averaged.

    N : `int`
        The number of consecutive rises that would constitute the start of a flare.
        Generally you want 4 minutes, so for 12s cadence N=20 and for 60s cadence
        n=4.

    start_threshold : `float`
        The required proportional rise over the N data readings that consitutes
        the start of a flare.

    get_duration : `bool`
        When True the start and end times will be found.
        Note: with this methof the start time is always found.
        Note: This must be true if you calculate the energy via numerical integration.

    get_energies : `bool`
        When True use the start/end times and data to use numerical integration
        to interpret the energy detected at the detector.

    Returns
    -------
    result : ~`pandas.DataFrame`
        The pandas.DataFrame of all detected  events within the given data.
        Indexed by the flare peak time, which is expected to be unique for each
        flare.
    """
    # Get the start index/time/flux of each flare
    arr_int_flare_starts = get_flare_start_indices_by_consecutive_rises_np(ser_data.values, N=N, start_threshold=start_threshold)

    # Convert flare start indices to datetimes
    arr_dt_flare_starts = ser_data.index.values[arr_int_flare_starts]

    # Get the flux for each start
    arr_flo_flare_start_flux = ser_data.values[arr_int_flare_starts]

    # Find all the maxima in this series
    #####arr_int_maxima = find_extrema(ser_data.values, extrema='maxima')
    arr_int_maxima = utils.find_maxima_fast(ser_data.values)

    # Get the first peaks for each flare
    arr_int_first_peaks = np.empty(arr_int_flare_starts.shape, dtype=int)
    for i in range(0,len(arr_int_flare_starts)):
        arr_int_first_peaks[i] = arr_int_maxima[np.argmax(arr_int_maxima>arr_int_flare_starts[i])]
    arr_dt_first_peaks = ser_data.index.values[arr_int_first_peaks]
    arr_flo_first_peak_flux = ser_data.values[arr_int_first_peaks]

    """
    if get_duration != None:
    NEED TO Implement
    """
    # calculate end thresholds and thus end indices/times
    #arr_flo_flare_end_threshold = arr_int_first_peaks - (arr_int_first_peaks - (arr_flo_flare_start_flux * end_drop))
    arr_flo_flare_end_threshold = (arr_flo_first_peak_flux + arr_flo_flare_start_flux) * end_threshold
    arr_int_flare_end_index, arr_flo_flare_end_flux = find_end_indices_and_flux(ser_data.values, arr_int_first_peaks, arr_flo_flare_end_threshold)
    arr_dt_flare_end = ser_data.index.values[arr_int_flare_end_index]

    ####
    # Could look for subsequent peaks
    ####

    # Create the Pandas DataFrame of this data
    dic_data = {'event_startindex': arr_int_flare_starts,
                'event_starttime': pd.to_datetime(arr_dt_flare_starts),
                'fl_strtflux': arr_flo_flare_start_flux,
                'event_endindex': arr_int_flare_end_index,
                'event_endtime': pd.to_datetime(arr_dt_flare_end),
                'event_peakindex': arr_int_first_peaks,
                'event_peaktime': pd.to_datetime(arr_dt_first_peaks),
                'fl_peakflux': arr_flo_first_peak_flux,
                'fl_peakflux_end_threshold': arr_flo_flare_end_threshold,
                'fl_duration_(td)': arr_dt_flare_end - arr_dt_flare_starts,
                'fl_duration_(s)': ((arr_dt_flare_end - arr_dt_flare_starts)/1000000000.0).astype(int),
                'fl_duration_initial_rise_(s)': ((arr_dt_first_peaks - arr_dt_flare_starts)/1000000000.0).astype(int),
                #'fl_goescls':  utils.intensity_to_flare_class_arr(arr_flo_first_peak_flux, dp=1)
                'fl_goescls':  utils.arr_to_cla(arr_flo_first_peak_flux, int_dp=1)}
    df_4min_flares = pd.DataFrame(dic_data, index=arr_dt_first_peaks)

    # Assuming we want the energy details
    if get_energies:
        pd_energies = utils.get_flare_energy_trap_inte(ser_data, df_4min_flares['event_starttime'], df_4min_flares['event_endtime'], df_4min_flares.index)
        # Add to the original DataFrame
        df_4min_flares = pd.concat([df_4min_flares, pd_energies], axis=1)

    #lis_meta = [['flare_detection_parameter_N', N],['flare_detection_parameter_threshold',threshold],['flare_detection_parameter_end_threshold',end_drop]]
    return df_4min_flares#, lis_meta