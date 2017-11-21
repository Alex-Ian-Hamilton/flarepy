# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:18:17 2017

@author: Alex
"""
# Basic imports
import pandas as pd
import numpy as np
#from datetime import timedelta
from scipy import signal

# Advanced imports
import flarepy.utils as utils
#from sunpy.lightcurve import GOESLightCurve
#from sunpy.time import TimeRange


def get_flare_peaks_cwt(ser_data, widths=np.arange(1,100), raw_data=None, ser_minima=None, get_duration=True, get_energies=True):
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
    arr_peak_time_indices = signal.find_peaks_cwt(ser_data.values, widths)
    ser_cwt_peaks = ser_raw_data[arr_peak_time_indices]

    # As a dataframe
    pd_peaks_cwt = pd.DataFrame(data={'fl_peakflux': ser_cwt_peaks})
    pd_peaks_cwt['fl_goescls'] = utils.arr_to_cla(pd_peaks_cwt['fl_peakflux'].values, int_dp=1)
    pd_peaks_cwt['event_peaktime'] = pd_peaks_cwt.index

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
    return pd_peaks_cwt
