# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:58:37 2017

@author: Alex
"""

import pandas as pd
import numpy as np
#import flarepy.utils as utils

def get_flare_start_end_using_min_min(ser_data=None, ser_minima=None, ser_peaks=None, raw_data=None):
    """
    Basic method for getting the start and end times for CWT detected peaks.
    In this case we use the local maxima and minima before and after the given
    peaks as the start and end.
    Note; this will be hevily effected by noise in the initial data.

    Parameters
    ----------
    ser_data : ~`pandas.Series`
        The dataset for the flare peaks.
        Generally GOES XRS B-channel X-Ray data.
        Should be pre-processed to exclude data spikes, data gaps (interpolate),
        rebinned to a common bin width and ideally averaged.

    ser_minima : ~`pandas.Series`
        The pre-calculated series of local minima.
        This is can be calcuated in-function, but if it's used many times in the
        base script then passing it is more efficient (then calculating each time).

    ser_peaks : ~`pandas.Series`
        The pandas.Series of all flare peak times.

    raw_data : ~`pandas.Series`
        #######

    Returns
    -------
    result: ~`pandas.DataFrame`
        The table of results, ordered/indexed by peak time.
    """
    # Are we given a pre-calculated series of minima (if it's needed many times thise is more efficient)
    if not isinstance(ser_minima, pd.Series):
        if not isinstance(ser_data, pd.Series):
            raise ValueError("Invalid arguments used for get_flare_start_end_for_cwt(), you must either include a series of local minima or a series of the original data.")
        else:
            ser_minima = ser_data[utils.find_minima_fast(ser_data.interpolate().values)]

    # Now find the duration for each flare
    #arr_min_before =
    lis_tsp_min_before = []
    lis_tsp_min_after = []
    #arr_flo_min_before = np.zeros([len(ser_peaks)])
    #arr_flo_min_after = np.zeros([len(ser_peaks)])
    #arr_flo_min_min_energy_tot = np.zeros([len(ser_peaks)])
    #arr_flo_min_min_energy_peak_only = np.zeros([len(ser_peaks)])

    # For each peak, find the start and end times
    for i in range(0, len(ser_peaks)):
        # Get the nearest minima by removing all minima after the flare peak time
        ser_minima_before = ser_minima.truncate(ser_data.index[0], ser_peaks.index[i])
        if len(ser_minima_before) > 1:
            # We have a minima before the given peak
            lis_tsp_min_before.append(ser_minima_before.index[-2])
        else:
            # There's no minima before the peak in the time window, assume the start value in a minima
            lis_tsp_min_before.append(ser_peaks.index[0])

        # Get the nearest minima by removing all minima before the flare peak time
        ser_minima_after = ser_minima.truncate(ser_peaks.index[i], ser_data.index[-1])
        if len(ser_minima_after) > 1:
            # We have a minima before the given peak
            lis_tsp_min_after.append(ser_minima_after.index[1])
        else:
            # There's no minima before the peak in the time window, assume the start value in a minima
            lis_tsp_min_after.append(ser_peaks.index[-1])

    # Create a DataFrame for flare start/end details
    arr_tsp_min_before = np.array(lis_tsp_min_before)
    arr_tsp_min_after = np.array(lis_tsp_min_after)
    df_flare_durations = pd.DataFrame(data={'event_starttime': arr_tsp_min_before,#'pre_flare_min_time'
                                            'event_endtime': arr_tsp_min_after,#'post_flare_min_time'
                                            'fl_duration_(td)': arr_tsp_min_after - arr_tsp_min_before,
                                            #'fl_duration_(s)': ((arr_tsp_min_after - arr_tsp_min_before).seconds),
                                            #'pre_flare_min_intensity': arr_flo_min_before,
                                            #'post_flare_min_intensity': arr_flo_min_after,
                                            #'min_min_energy_total': arr_flo_min_min_energy_tot,
                                            #'min_min_energy_peak_only': arr_flo_min_min_energy_peak_only
                                            }, index=ser_peaks.index)

    return df_flare_durations