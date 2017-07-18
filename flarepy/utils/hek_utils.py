# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:01:10 2017

@author: Alex
"""
import pandas as pd
from sunpy.net import hek
import numpy as np
from datetime import timedelta


def hek_results_to_df(hek_results, minimise=True, event_duration=True):
    """
    A basic function to return a peak time indexed pandas DataFrame object from
    the results list of the SunPy's HEK API.
    It's also got the ability to remove all columns with nothing but nan, None,
    0 and '' values, to try and make it neater.

    Parameters
    ----------
    hek_results : `list`
        The results list generated by the sunpy.net.hek client.

    minimise : `bool`
        When True the returned pandas.DataFrame with have several columns removed
        that have nothing by NaN, None and 0 values.
        Designed to neaten the output a bit.

    event_duration : `bool`
        When True calculate and add a column for duration.

    Returns
    -------
    result : ~`pandas.DataFrame`
        The pandas.DataFrame of all HEK events from the results.
        Indexed by the flare peak time, which is expected to be unique for each
        flare.
    """
    df_hek_temp = pd.DataFrame(hek_results)
    hek_index = pd.to_datetime(df_hek_temp['event_peaktime'].values)
    df_hek = pd.DataFrame(df_hek_temp)
    df_hek.index = hek_index

    # If asked to minimise, then drop redundant columns
    if minimise:
        # Drop all NaN columns
        df_hek = df_hek.dropna(how='all')

        # Look for other useless columns
        for str_col in list(df_hek.columns):
            # Extract column values
            vals = df_hek[str_col].values

            # Remove all columns with only NoneType
            if np.all(vals) == None:
                df_hek = df_hek.drop(str_col, axis=1)

        # And remove some columns manually
        set_col_remove = set(['obs_dataprepurl', 'obs_firstprocessingdate', 'obs_includesnrt', 'event_clippedspatial', 'event_clippedtemporal', 'event_expires', 'event_importance_num_ratings', 'event_mapurl', 'event_maskurl', 'event_type', 'fl_efoldtimeunit', 'fl_fluenceunit', 'fl_halphaclass', 'fl_peakemunit', 'fl_peakfluxunit', 'fl_peaktempunit', 'obs_lastprocessingdate', 'chaincodetype', 'search_frm_name', 'ar_compactnesscls', 'obs_title', 'bound_chaincode', 'bound_chaincode', 'ar_penumbracls', 'ar_zurichcls', 'area_unit', 'rasterscan', 'rasterscantype', 'ar_penumbracl', 'ar_zurichcls area_unit', 'ar_mcintoshcls', 'ar_mtwilsoncls', 'ar_noaaclass', 'search_instrument', 'search_observatory', 'skel_chaincode'])
        set_col_remove = set_col_remove.intersection(set(df_hek.columns))
        for str_col in set_col_remove:
            df_hek = df_hek.drop(str_col, axis=1)

    # Add flare duration column if specified
    if event_duration:
        df_hek['event_duration'] = pd.to_datetime(df_hek['event_endtime']) - pd.to_datetime(df_hek['event_starttime'])

    return df_hek


def get_hek_goes_flares(str_start, str_end, minimise=True, fail_safe=True):
    """
    A wrapper for SunPy's HEK lookup, to make it quicker to get the relivant
    data in a useful form. (peak time indexed pandas DataFrame)

    Parameters
    ----------
    str_start : `str`
        The string defining the start of the time range we want HEK data for.

    str_start : `str`
        The string defining the end of the time range we want HEK data for.

    minimise : `bool`
        When True the returned pandas.DataFrame with have several columns removed
        that have nothing by NaN, None and 0 values.
        Designed to neaten the output a bit.

    fail_safe : `bool`
        When True if an exception is returned (for example HEK download timeout)
        then you'll have an empty pandas.DataFrame returned.
        Designed to allow code to work offline, thou it'll not show HEK results
        in outputs.

    Returns
    -------
    result : ~`pandas.DataFrame`
        The pandas.DataFrame of all HEK events within the given timerange.
        Indexed by the flare peak time, which is expected to be unique for each
        flare.
    """
    client = hek.HEKClient()

    # Search filters
    event_type = 'FL'
    event_observatory = 'GOES'

    # Get the values for a given daterange
    try:
        hek_results = client.query(hek.attrs.Time(str_start,str_end),
                               hek.attrs.EventType(event_type),
                               hek.attrs.OBS.Observatory == event_observatory)

        # Make a peak time-indexed pandas.DataFrame
        df_hek = hek_results_to_df(hek_results, minimise=minimise)
    except:
        df_hek = pd.DataFrame()

    # return the DataFrame
    return df_hek

def get_equiv_hek_results_OLD(data, hek_data=None, widths=[60, 120], hek_padding=timedelta(minutes=3)):
    """
    Try to correlate detected flares to HEK flares.

    Parameters
    ----------
    data: array or ~pandas.Series or ~pandas.DataFrame
        The input data, at minimum this must be an array of datetime-like values
        giving the peak times (flare start/finish times are not deeemed unique
        enough), ideally this is a pandas DataFrame with columns for flare class,
        peak intensity and such.

    hek_data: `list` or `~pandas.DataFrame`
        The HEK data to compare too, either in original list form or a sanatised
        pandas.DataFrame, if ommited the the function will DL automatically
        (which wastes time if you already have done).

    widths: `list`
        A list of integers (in seconds) for the widths to check.
        Assuming data has 60 second bins, you can set this to 60 and it will find
        flares within a inute of eachother. (or 120 to find them within 2 minutes
        of eachother).
        Setting an individual width may cause issues if more then 1 flare fit
        within the same bin/window (#### Not sure the behavour), so instead add
        multiple widths and the function will search sucessivly further away for
        each flare, upto the maximum width defined.

    hek_padding: `~datetime.timedelta`
        If HEK data not supplied then it will be automatically downloaded.
        This parameter adds time either side to the search window to ensure of
        the flare is detected near the edges then it can still correlated to a
        HEK flare slightly outside the range or original flares.

    Returns
    -------
    result: array
        ###########
    """
    # Sanitise input data
    df_data = data
    if isinstance(data, np.ndarray):
        df_data = pd.DataFrame({'event_peaktime': data}, index=pd.to_datetime(data))
    elif isinstance(data, pd.Series):
        df_data = pd.DataFrame(data)
    # Get HEK data if necessary
    if not isinstance(hek_data, pd.DataFrame):
        df_hek = hek_data
    else:
        df_hek = get_hek_goes_flares(df_data.index[0]-hek_padding, df_data.index[-1]+hek_padding, minimise=True)


    # Make pandas Series to hold the original peaktime and index peaktime (that'll be resampled)
    #ser_det = pd.Series(data=pd.to_datetime(df_data['event_peaktime'].values), index=pd.to_datetime(df_data['event_peaktime'].values))
    #ser_hek = pd.Series(data=pd.to_datetime(hek_data['event_peaktime'].values), index=pd.to_datetime(hek_data['event_peaktime'].values))
    ser_det = df_data['event_peaktime']
    ser_hek = hek_data['event_peaktime']

    # Holders for matched/unmatched flares
    ser_matched = pd.Series()
    ser_unmatched = pd.Series(data=ser_det.index, index=ser_det.index)

    for int_width in widths:
        print(ser_unmatched)
        ser_det_resampled = ser_unmatched.resample(str(2*int_width)+'S').max()
        ser_hek_resampled = ser_hek.resample(str(2*int_width)+'S').max()

        # Remove empty rows
        ser_det_resampled = ser_det_resampled.dropna(axis=0)


        # Try each remaining unmatched flare
        for i in range(0, len(ser_unmatched)):
            dt_det_resampled = ser_det_resampled.index[i]
            dt_det = ser_det_resampled.values[i]
            if not isinstance(ser_hek_resampled[dt_det_resampled], pd._libs.tslib.NaTType):
                # Match found
                dt_hek = ser_hek_resampled[dt_det_resampled]

                # Add result to the matched Series
                ser_matched[dt_det] = dt_hek

                # Remove from unmatched Series
                ser_unmatched = ser_unmatched.drop(dt_det)


    # Add data for matched to the data DF
    df_matched = pd.DataFrame(data={'HEK_event_peaktime': ser_matched.values}, index=ser_matched.index)
    df_out = pd.concat([df_data, df_matched], axis=1)

    # Add comparitive values where possible
    lis_cols_to_copy = ['fl_goescls']
    for str_col in lis_cols_to_copy:
        if str_col in list(df_hek.columns):
            ser_temp = df_hek[str_col][ser_matched.index]
            ser_temp.index = ser_matched.index
            df_out['HEK_'+str_col] = ser_temp

    return df_matched, ser_unmatched


def get_equiv_hek_results(data, hek_data=None, windows=[timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=3)], hek_padding=timedelta(minutes=3)):
    """
    Try to correlate detected flares to HEK flares.

    Parameters
    ----------
    data: array or ~pandas.Series or ~pandas.DataFrame
        The input data, at minimum this must be an array of datetime-like values
        giving the peak times (flare start/finish times are not deeemed unique
        enough), ideally this is a pandas DataFrame with columns for flare class,
        peak intensity and such.

    hek_data: `list` or `~pandas.DataFrame`
        The HEK data to compare too, either in original list form or a sanatised
        pandas.DataFrame, if ommited the the function will DL automatically
        (which wastes time if you already have done).

    widths: `list`
        A list of integers (in seconds) for the widths to check.
        Assuming data has 60 second bins, you can set this to 60 and it will find
        flares within a inute of eachother. (or 120 to find them within 2 minutes
        of eachother).
        Setting an individual width may cause issues if more then 1 flare fit
        within the same bin/window (#### Not sure the behavour), so instead add
        multiple widths and the function will search sucessivly further away for
        each flare, upto the maximum width defined.

    hek_padding: `~datetime.timedelta`
        If HEK data not supplied then it will be automatically downloaded.
        This parameter adds time either side to the search window to ensure of
        the flare is detected near the edges then it can still correlated to a
        HEK flare slightly outside the range or original flares.

    Returns
    -------
    result: array
        The list of indices for local maxima.
    """
    # Sanitise input data
    df_data = data
    if isinstance(data, np.ndarray):
        df_data = pd.DataFrame({'event_peaktime': data}, index=pd.to_datetime(data))
    elif isinstance(data, pd.Series):
        df_data = pd.DataFrame(data)
    # Get HEK data if necessary
    if not isinstance(hek_data, pd.DataFrame):
        df_hek = hek_data
    else:
        df_hek = get_hek_goes_flares(df_data.index[0]-hek_padding, df_data.index[-1]+hek_padding, minimise=True)


    # Make pandas Series to hold the original peaktime and index peaktime (that'll be resampled)
    #ser_det = pd.Series(data=pd.to_datetime(df_data['event_peaktime'].values), index=pd.to_datetime(df_data['event_peaktime'].values))
    #ser_hek = pd.Series(data=pd.to_datetime(hek_data['event_peaktime'].values), index=pd.to_datetime(hek_data['event_peaktime'].values))
    ser_det = df_data['event_peaktime']
    ser_hek = hek_data['event_peaktime']

    # Holders for matched/unmatched flares
    ser_matched = pd.Series()
    ser_unmatched = pd.Series(data=np.nan, index=ser_det.index)

    # For each of the flares
    for i in range(0, len(ser_det)):
        dt_det = ser_det.index[i]

        # Check if the flare detection is the same as a HEK entry
        if dt_det in ser_hek.index:
            ser_matched[dt_det] = dt_det
            ser_unmatched = ser_unmatched.drop(dt_det)
        else:
            # If not exactly the same, check within the given windows
            for td_window in windows:
                dt_window_min = dt_det - td_window
                dt_window_max = dt_det + td_window

                # truncate to this window and see if any HEK results are present
                ser_hek_trun = ser_hek.truncate(dt_window_min, dt_window_max)
                if len(ser_hek_trun) == 1:
                    # We have a unique match
                    ser_matched[dt_det] = ser_hek_trun.index[0]
                    ser_unmatched = ser_unmatched.drop(dt_det)
                    break
                elif len(ser_hek_trun) == 2:
                    # We have 2 matches, if diff between windows is small enough
                    # these are equidistant from the detection. Choose the first.
                    ser_matched[dt_det] = ser_hek_trun.index[0]
                    ser_unmatched = ser_unmatched.drop(dt_det)
                    break
                elif len(ser_hek_trun) > 2:
                    # Too many matches, windows are 2 large.
                    break

    """
    # This is meant to be a father method using resampling, it doesn't work yet.
    for int_width in widths:
        print(ser_unmatched)
        ser_det_resampled = ser_unmatched.resample(str(2*int_width)+'S').max()
        ser_hek_resampled = ser_hek.resample(str(2*int_width)+'S').max()

        # Remove empty rows
        ser_det_resampled = ser_det_resampled.dropna(axis=0)


        # Try each remaining unmatched flare
        for i in range(0, len(ser_unmatched)):
            dt_det_resampled = ser_det_resampled.index[i]
            dt_det = ser_det_resampled.values[i]
            if not isinstance(ser_hek_resampled[dt_det_resampled], pd._libs.tslib.NaTType):
                # Match found
                dt_hek = ser_hek_resampled[dt_det_resampled]

                # Add result to the matched Series
                ser_matched[dt_det] = dt_hek

                # Remove from unmatched Series
                ser_unmatched = ser_unmatched.drop(dt_det)
    """

    # Add data for matched to the data DF
    df_matched = pd.DataFrame(data={'HEK_event_peaktime': ser_matched.values}, index=ser_matched.index)
    df_out = pd.concat([df_data, df_matched], axis=1)

    # Add comparitive values where possible
    lis_cols_to_copy = ['fl_goescls']
    for str_col in lis_cols_to_copy:
        if str_col in list(df_hek.columns):
            ser_temp = df_hek[str_col][ser_matched.index]
            ser_temp.index = ser_matched.index
            df_out['HEK_'+str_col] = ser_temp

    return df_matched, ser_unmatched




