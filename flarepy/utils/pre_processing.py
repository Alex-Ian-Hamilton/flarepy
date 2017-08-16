# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 08:40:02 2017

@author: alex_
"""

import astropy.units as u
import pandas as pd
import numpy as np

def pre_process(data, resample_bins=60*u.s, resample_method='median', average_method='mean', int_cart=5):
    """
    A convenience function, designed to make generating pre-processed data easier
    to write and to support astropy units.
    
    Note: when doing the boxcart averaging I use the parameter center=True so
    the boxcart gets the average of values before and after to get a cells
    value and the parameter min_periods=1 so the edges (where you can get an
    average over the full cart) will get the average over all of the values avalible.
    
    Parameters
    ----------
    data : ~`pandas.Series`
        The data.

    resample_bins : `astropy.Quantity` or `str`
        The size of the bins to use as either a quantity or a pandas string.

    resample_method : `str`
        The method used to rebin/resample.

    average_method : `str`
        The method used for averaging.

    int_cart : `int`
        The width of the boxcart to average over.
        
    Returns
    -------
    result : ~`numpy.ndarray`
        The array of indices of the data spikes.
    """
    # Sanitise arguments
    if isinstance(resample_bins, u.Quantity):
        str_bins = str(int(resample_bins.to(u.s).value)) + 'S'
    else:
        str_bins = resample_bins

    # Make a mask
    ser_raw_mask = pd.Series(data=np.logical_or(np.isnan(data.values), data.values == 0.0), index=data.index)
    ser_raw_int = data.replace({0.0:np.nan}).interpolate()

    # Resample/Rebin
    ser_raw_int_res = ser_raw_int.resample(str_bins).median()
    ser_raw_int_res_mask = ser_raw_mask.resample(str_bins).max()
    
    # Rolling Average
    ser_raw_int_res_box = ser_raw_int_res.rolling(int_cart, center=True, min_periods=1).mean()
    
    # Make series for plots
    # Make nan holes where data gaps were originally.
    ser_plt_fil = pd.Series(ser_raw_int_res_box)
    ser_plt_fil.iloc[np.where(ser_raw_int_res_mask != 0.0)] = np.nan
    ser_plt_raw = pd.Series(data)
    ser_plt_raw.iloc[np.where(ser_raw_mask != 0.0)] = np.nan
    
    return ser_raw_int_res_box, ser_raw_int_res, ser_plt_fil
    

def find_data_spikes(data, threshold=5, method='relative'):
    """
    A basic function to find data spikes in an array or series of data.
    A spike is defined as a single datapoint that is much higher then the values
    both before and after.
    The amount higher is given by the threshold which can be either absolute or
    relative, as controlled by the method.

    Parameters
    ----------
    data : ~`pandas.Series` or ~`numpy.ndarray`
        The data to look for spikes in.

    threshold : `float`
        The amount by which the value has to change to be considered a spike.

    method : `str`
        Defines if the threshold is absolute or relative.

    Returns
    -------
    result : ~`numpy.ndarray`
        The array of indices of the data spikes.
    """
    # Sanitise arguments
    if isinstance(data, pd.Series):
        arr_data = data.values
    else:
        # Otherwise we have a numpy array
        arr_data = data

    # Get the rolling values
    arr_roll_1 = np.roll(arr_data,1)
    arr_roll_n1 = np.roll(arr_data,-1)

    # Find the difference before and after
    arr_diff_before = arr_data - arr_roll_1
    arr_diff_after = arr_data - arr_roll_n1
    
    # If we're looking for relative changes then divide by the initial value
    if method == 'relative':
        arr_diff_before = arr_diff_before / arr_roll_1
        arr_diff_after = arr_diff_after / arr_roll_n1
    
    # Find sudden jumps/drops
    arr_jumps = arr_diff_before >= threshold
    arr_drops = arr_diff_after >= threshold
    
    # A spike is a single point with a sudden jump and then drop
    arr_spikes = np.logical_and(arr_jumps, arr_drops)
    
    # Remove the invalid first/last values
    arr_spikes[0] = False
    arr_spikes[-1] = False
    
    # Get the integer indices
    arr_indices = np.where(arr_spikes)

    # If given a pandas series then convert the integer indices to datetimes
    if isinstance(data, pd.Series):
        arr_indices = data.index[arr_indices]
    
    return arr_indices

def interpolate_data_spikes(data, threshold=5, method='relative'):
    """
    Finds data spikes (using find_data_spikes) and interpolates values for them.

    Parameters
    ----------
    data : ~`pandas.Series` or ~`numpy.ndarray`
        The data to look for spikes in.

    threshold : `float`
        The amount by which the value has to change to be considered a spike.

    method : `str`
        Defines if the threshold is absolute or relative.

    Returns
    -------
    result : ~`numpy.ndarray`
        The array of indices of the data spikes.
    """    
    # Make a mask
    data_mask = pd.Series(data=np.logical_or(np.isnan(data.values), data.values == 0.0), index=data.index)
    
    # Now replace spikes with NaN values and interpolate
    data[find_data_spikes(data, threshold=5, method='relative')] = np.nan
    raw_int = data.replace({0.0:np.nan}).interpolate()
        
    # Remove the non-spike interpolated values using the mask
    plt_raw.iloc[np.where(data_mask != 0.0)] = np.nan