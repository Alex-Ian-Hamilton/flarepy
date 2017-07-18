# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:55:59 2017

@author: Alex
"""

import numpy as np
import pandas as pd

def get_flare_energy_trap_inte(ser_data, arr_starttimes, arr_endtimes, arr_index):
    """
    Parameters
    ----------
    ser_data : ~`pandas.Series`
        The dataset of flux.
        Generally GOES XRS B-channel X-Ray data.
        Should be pre-processed to exclude data spikes, data gaps (interpolate),
        rebinned to a common bin width and ideally averaged.

    arr_starttimes : array
        An array of the flare start times.

    arr_endtimes : array
        An array of the flare end times.

    arr_index : array
        An array of the flare index, generally peak times.

    Returns
    -------
    result: ~`pandas.DataFrame`
        The table of results, ordered/indexed by peak time.
    """
    # Lists to hold the results
    lis_energies_total = []
    lis_energies_over_minima = []

    # Get the energy between start and end of each flare
    for i in range(0, len(arr_starttimes)):
        # get the parameters
        dt_starttime = arr_starttimes[i]
        dt_endtime = arr_endtimes[i]
        flo_min_start = ser_data.values[i]
        flo_min_end = ser_data.values[i]

        # For each flare, get the truncated data
        ser_flare = ser_data.truncate(dt_starttime, dt_endtime)

        # Assume the first time bin is representative of the whole set
        dx = (ser_flare.index[1] - ser_flare.index[0]).seconds

        # Interpolate the energy
        flo_energy_tot = np.trapz(ser_flare.values, dx=dx)
        lis_energies_total.append(flo_energy_tot)
        lis_energies_over_minima.append(flo_energy_tot - (((flo_min_start + flo_min_end) * 0.5 ) * (dt_endtime - dt_starttime).seconds))

    return pd.DataFrame(data={'fl_energy_total': lis_energies_total}, index=arr_index)# 'fl_energy_above_flat': lis_energies_over_minima