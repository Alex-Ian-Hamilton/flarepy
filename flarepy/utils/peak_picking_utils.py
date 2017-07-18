# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:32:33 2016

@author: Alex
"""
import numpy as np
from scipy import signal

__all__ = ['find_maxima_fast', 'find_maxima_mid', 'find_maxima_slow',
           'find_minima_fast', 'find_minima_mid', 'find_minima_slow']

###############################################################################
#
# Maxima algorithms
#
###############################################################################


def find_maxima_fast(arr_inputs):
    """
    Find the indices of all single-width maxima in an array and return the
    indices as an array.
    Uses numpy.roll(x,n) to get the values ahead and behind for each elements,
    then it will check if both are below the current values and return.

    Parameters
    ----------
    arr_inputs: array
        The dataset to look for maxima in.

    Returns
    -------
    result: array
        The list of indices for local maxima.
    """
    arr_boo_roll_plus = np.roll(arr_inputs,1)
    arr_boo_roll_minus = np.roll(arr_inputs,-1)

    # Find peaks
    arr_boo_peak = np.logical_and((arr_boo_roll_plus < arr_inputs),
                                  (arr_boo_roll_minus < arr_inputs))

    return np.where(arr_boo_peak)[0]

def find_maxima_mid(arr_inputs):
    """
    Find the indices of maxima in an array and return the indices as an array.
    Finds single-width, double-width (returns first, second or maybe both) and
    3-wide (returns middle) peaks.
    Uses numpy.roll(x,n) to get the values ahead and behind for each elements,
    then it will check if both are below the current values and return.

    Parameters
    ----------
    arr_inputs: array
        The dataset to look for maxima in.

    Returns
    -------
    result: array
        The list of indices for local maxima.
    """
    arr_boo_roll_plus_1 = np.roll(arr_inputs,1)
    arr_boo_roll_plus_2 = np.roll(arr_inputs,2)
    arr_boo_roll_minus_1 = np.roll(arr_inputs,-1)
    arr_boo_roll_minus_2 = np.roll(arr_inputs,-2)

    # Find peaks
    arr_boo_peak_roll_1 = np.logical_and((arr_boo_roll_plus_1 < arr_inputs),
                                  (arr_boo_roll_minus_1 < arr_inputs))
    arr_boo_peak_roll_2 = np.logical_and((arr_boo_roll_plus_2 < arr_inputs),
                                  (arr_boo_roll_minus_2 < arr_inputs))

    return np.where(np.logical_or(arr_boo_peak_roll_1, arr_boo_peak_roll_2))

def find_maxima_slow(array):
    """
    Find the indices of all maxima in an array and return the indices as an
    array.
    For maxima spread across multiple elements it will return only the index of
    the first.

    Uses Python loops to do this, so this is a slow implementation.

    Parameters
    ----------
    arr_inputs: array
        The dataset to look for maxima in.

    Returns
    -------
    result: array
        The list of indices for local maxima.
    """
    output = []
    for i in range(1, len(array)-1):

        # Find the index of the values before and after
        i_before = 1
        i_after = 1
        # Use indices to get values
        val_before = array[i-i_before]
        val = array[i]
        val_after = array[i+i_after]

        # Roam either side of values if we have a flat part
        while val_before == val:
            i_before = i_before + 1
            if array[i-i_before] == np.nan or i_before < 0:
                break
            val_before = array[i-i_before]
        while val_after == val:
            i_after = i_after + 1
            if array[i+i_after] == np.nan or i_after >= len(array):
                break
            val_after = array[i+i_after]

        # Find maxima
        if val > val_before and val > val_after:
            # Check we dont have a flat, if so ignore all but the first value
            if array[i-1] != val:
                output.append(i)

    return np.array(output)

def find_maxima_cwt(arr_inputs, arr_cwt_widths=np.arange(1,10)):
    """
    Detect flares using CWT (implmented in scipy).
    Parameters:
        ser_data: the data to look for the peaks.
            Note: must be constant binned with no data gaps.
        arr_cwt_widths: parameter for the CWT method.

    """
    arr_peaktime_indices = signal.find_peaks_cwt(arr_inputs, arr_cwt_widths)
    return arr_peaktime_indices



###############################################################################
#
# Minima algorithms
#
###############################################################################

def find_minima_fast(arr_inputs):
    """
    Find the indices of all single-width minima in an array and return the
    indices as an array.
    Uses numpy.roll(x,n) to get the values ahead and behind for each elements,
    then it will check if both are below the current values and return.

    Parameters
    ----------
    arr_inputs: arr
        The dataset to look for minima in.

    Returns
    -------
    result: array
        The list of indices for local minima.
    """
    arr_boo_roll_plus = np.roll(arr_inputs,1)
    arr_boo_roll_minus = np.roll(arr_inputs,-1)

    # Find peaks
    arr_boo_peak = np.logical_and((arr_boo_roll_plus > arr_inputs),
                                  (arr_boo_roll_minus > arr_inputs))

    return np.where(arr_boo_peak)[0]

def find_minima_mid(arr_inputs):
    """
    Find the indices of minima in an array and return the indices as an array.
    Finds single-width, double-width (returns first, second or maybe both) and
    3-wide (returns middle) peaks.
    Uses numpy.roll(x,n) to get the values ahead and behind for each elements,
    then it will check if both are below the current values and return.

    Parameters
    ----------
    arr_inputs: arr
        The dataset to look for minima in.

    Returns
    -------
    result: array
        The list of indices for local minima.
    """
    arr_boo_roll_plus_1 = np.roll(arr_inputs,1)
    arr_boo_roll_plus_2 = np.roll(arr_inputs,2)
    arr_boo_roll_minus_1 = np.roll(arr_inputs,-1)
    arr_boo_roll_minus_2 = np.roll(arr_inputs,-2)

    # Find peaks
    arr_boo_peak_roll_1 = np.logical_and((arr_boo_roll_plus_1 > arr_inputs),
                                  (arr_boo_roll_minus_1 > arr_inputs))
    arr_boo_peak_roll_2 = np.logical_and((arr_boo_roll_plus_2 > arr_inputs),
                                  (arr_boo_roll_minus_2 > arr_inputs))

    return np.where(np.logical_or(arr_boo_peak_roll_1, arr_boo_peak_roll_2))

def find_minima_slow(array):
    """
    Find the indices of all minima in an array and return the indices as an
    array.
    For minima spread across multiple elements it will return only the index of
    the first.

    Uses Python loops to do this, so this is a slow implementation.

    Parameters
    ----------
    arr_inputs: array
        The dataset to look for maxima in.

    Returns
    -------
    result: array
        The list of indices for local maxima.
    """
    output = []
    for i in range(1, len(array)-1):
        # Find the index of the values before and after
        i_before = 1
        i_after = 1

        # Use indices to get values
        val_before = array[i-i_before]
        val = array[i]
        val_after = array[i+i_after]


        # Roam either side of values if we have a flat part
        while val_before == val:
            i_before = i_before + 1
            if array[i-i_before] == np.nan or i_before < 0:
                break
            val_before = array[i-i_before]
            #print('i_before: ' + str(i_before))
            #print('val_before: ' + str(val_before) + '\n')
        while val_after == val:
            i_after = i_after + 1
            if array[i+i_after] == np.nan or i_after >= len(array):
                break
            val_after = array[i+i_after]

        # Find minima
        if val < val_before and val < val_after:
            # Check we dont have a flat, if so ignore all but the first value
            if array[i-1] != val:
                output.append(i)

    return np.array(output)
