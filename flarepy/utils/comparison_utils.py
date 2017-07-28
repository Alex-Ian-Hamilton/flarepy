# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:27:14 2017

@author: Alex
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from flarepy import utils
import warnings

def compare_to_primary(data_ref, data, windows=[timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=3)]):
    """
    Basic function to compare a secondary dataset to a primary reference.

    Parameters
    ----------
    data_ref : ~`pandas.Series` or ~`pandas.Series`
        The primary/reference list of flares indexed by peaktime.
        The actual data valus arn't relivent, it compares based on the index.
        Note: currently expects values in this index to be unique.

    data : ~`pandas.Series`
        The secondary flare list, that will be compared to the primary (reference).

    windows : `list` of `datetime.timedelta`
        The list of windows to search in, these get progressively larger.
        Ideally you want the increase per window to be small enough to only add
        a single match.

    Returns
    -------
    ser_matched : ~`pandas.Series`
        The list of matches between the primary (reference) and the secondary list.
        The index is the index from the secondary, the values are the associated
        index from the primary reference list.

    ser_not_in_primary : ~`pandas.Series`
        The list of flares in the secondary and not in the primary (reference).
        The index is the index from the secondary, the values are all NaN.

    ser_not_in_secondary : ~`pandas.Series`
        The list of flares in the primary (reference) and not in the secondary.
        The index is the index from the secondary, the values are all NaN.
    """
    # Sanatise inputs
    ser_ref = data_ref
    if isinstance(ser_ref, pd.DataFrame):
        ser_ref = pd.Series(data=np.nan, index=ser_ref.index)
    elif isinstance(ser_ref, np.ndarray):
        ser_ref = pd.Series(data=np.nan, index=pd.to_datetime(ser_ref))
    ser_data = data
    if isinstance(ser_data, pd.DataFrame):
        ser_data = pd.Series(data=np.nan, index=ser_data.index)
    elif isinstance(ser_ref, np.ndarray):
        ser_data = pd.Series(data=np.nan, index=pd.to_datetime(ser_data))

    # Check for duplicates, remove if detected
    if not (np.all(np.invert(ser_ref.index.duplicated()))):
        warnings.warn("HEK data has flares with dupicate peak times. First entries have been kept.")
        ser_ref = ser_ref.groupby(ser_ref.index).first()
    if not (np.all(np.invert(ser_data.index.duplicated()))):
        warnings.warn("HEK data has flares with dupicate peak times. First entries have been kept.")
        ser_data = ser_data.groupby(ser_data.index).first()

    # For each of the values in the primary (reference) data
    ser_matched, ser_not_in_primary = compare_ser_results(ser_ref, ser_data, windows=windows)

    # For each of the values in the secondary
    ser_matched_alt, ser_not_in_secondary = compare_ser_results(ser_data, ser_ref, windows=windows)

    # Check if the matched series are equivilent. I'm not sure if this will be true.
    if not ser_matched.equals(ser_matched_alt):
        print('Note: the two matched lists don\'t match.')

    # Return the results
    return ser_matched, ser_not_in_primary, ser_not_in_secondary

def get_col_multiindex(windows, metrics, names=['Window','Metric']):
    """
    A function to create the correct column multiindex for a given set of windows
    and metrics.

    Parameters
    ----------

    data : ~`pandas.Series`
        The secondary flare list, that will be compared to the primary (reference).

    windows : `list` of `datetime.timedelta`
        The list of windows to search in, these get progressively larger.
        Ideally you want the increase per window to be small enough to only add
        a single match.

    names : `list` of `str` (['Window','Metric'])
        The list of names for each level of the multiindex.
        Generally left blank to use the default, otherwise the resulting pandas
        DataFrame may not work with other FlarePy tools, but added to give more
        versatility to the function.

    Returns
    -------
    output : ~`pandas.core.indexes.multi.MultiIndex`
        The multiindex correlating to the given windows and metrics.
    """
    # Add the 0s window if not included.
    if not windows[0].seconds is 0:
        windows.insert(0, timedelta(minutes=0))

    # Create list for each level
    lis_level_0 = []
    lis_level_1 = []
    for window in windows:
        # Window string
        str_win = str(window.seconds) + 's'

        # Add to the two lists
        for i in range(0, len(metrics)):
            lis_level_0.append(str_win)
            lis_level_1.append(metrics[i])

    # return the resulting multiindex
    return pd.MultiIndex.from_tuples(list(zip(*[lis_level_0, lis_level_1])), names=names)

def get_varied_window_stats(ser_data_ref, dic_ser_data, percentage=False, windows=[timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=3)], keydelimiter=';'):
    """
    Take all the results, perform comparisons and output a multiindex pandas
    DataFrame of them.

    Parameters
    ----------

    ser_data_ref : ~`pandas.Series`
        The primary/reference flare list, that will be used for comparison.

    dic_ser_data : `dict`
        The dictionary of all the results datasets to be compared.
        Note, for good measure it's generally worth having the reference in here.

    percentage : (False)
        ####### Needs to be implmented

    windows : `list` of `datetime.timedelta`
        The list of windows to search in, these get progressively larger.
        Each window will be checked and results added seperately so you can
        compare the results between windows for each method.

    keydelimiter : (';')
        The keys in the dictionary are created as a merger of the pre-processing
        method and method parameters details.
        This is the delimiter used to seperate out these parts.

    Returns
    -------
    ser_matched : ~`pandas.core.indexes.multi.MultiIndex`
        The multiindex correlating to the given windows and metrics.
    """
    #df_cwt_compared_to_hek, ser_in_cwt_not_in_hek, ser_in_hek_not_in_cwt = utils.compare_to_HEK(df_hek, df_peaks_cwt, windows=lis_windows)

    # Create the data and row index array
    lis_data = []
    lis_row_index = [[],[],[]]

    for key, df_detections in dic_ser_data.items():
        # Get the row index parameters and add to the row index array
        lis_index = key.split(keydelimiter)
        lis_row_index[0].append(lis_index[0])
        lis_row_index[1].append(lis_index[1])
        lis_row_index[2].append(lis_index[2])

        # Get the stats data values
        lis_data_line = []
        for i in range(0,len(windows)+1):#td_windows in windows:
            #print('windows[0:i]: ' + str(windows[0:i]))
            ser_matched, ser_not_in_primary, ser_not_in_secondary = compare_to_primary(ser_data_ref, df_detections, windows=windows[0:i])

            # Add the values
            lis_data_line.append(len(ser_not_in_primary))
            lis_data_line.append(len(ser_matched))
            lis_data_line.append(len(ser_not_in_secondary))

        # Add this line to the data
        lis_data.append(lis_data_line)

    # Create the column and row multiindices
    col_index = get_col_multiindex(windows, ['FA', 'M', 'FR'])
    #[np.array(['0s','0s','0s','2s','1s','1s','2s','2s','2s','3s','3s','3s']),np.array(['FA','M','FR','False Acceptance','Matched','False Rejection','False Acceptance','Matched','False Rejection','False Acceptance','Matched','False Rejection'])]
    row_index = pd.MultiIndex.from_tuples(list(zip(*lis_row_index)), names=['Pre-Processing','Method','Parameters'])

    """
    lis_data=[['bin=60s','HEK - Reference','',0,int_hek,0,0,int_hek,0,0,int_hek,0,0,int_hek,0],
          ['bin=60s boxcart=5','4min','N=4',3,12,8,3,12,8,3,12,8,3,12,8],
          ['bin=60s boxcart=5','CWT','W=1…49',4,16,4,4,16,4,4,16,4,4,16,4],['bin=60s boxcart=5','CWT','W=1…100',3,12,8,3,12,8,3,12,8,3,12,8]]
    lis_data=[[2,15,5,3,16,3,4,17,1,5,18,0],[3,12,8,4,13,6,5,14,4,6,15,2],[4,16,4,5,17,2,6,18,0,7,19,0],[3,12,8,4,13,6,5,14,4,6,15,2]]
    col_index = pd.MultiIndex.from_tuples(list(zip(*lis_col_index)), names=['Window','Metric'])
    lis_row_index = [np.array(['bin=12s boxcart=5','bin=60s boxcart=5','bin=60s boxcart=5','bin=60s boxcart=5']),np.array(['4min','4min','CWT','CWT']),np.array(['N=20','N=4','W=1…49','W=1…100'])]
    row_index = pd.MultiIndex.from_tuples(list(zip(*lis_row_index)), names=['Pre-Processing','Method','Parameters'])

    df_temp = pd.DataFrame(data=np.zeros([4,12]), columns=col_index, index=row_index)
    """
    return pd.DataFrame(data=lis_data, columns=col_index, index=row_index)



def compare_ser_results(ser_pri, ser_sec, windows=[timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=3)]):
    """
    The underlaying method used to compare one pandas.Series to another.
    The algorithm starts by checking for an exact match, if no match is found
    then this checks progressively wider windows (arround the primary/reference)
    as defined by the windows argument.
    Windows should increase in such a way as to reduce the chances of multiple
    matches, if 2 matches are found in the lowest matching window then we return
    the first.
    If we get more then 2 matches we assume it's too big a window and don't
    return matches.

    Parameters
    ----------
    ser_pri : ~`pandas.Series`
        The primary/reference list of flares indexed by peaktime.
        The actual data valus arn't relivent, it compares based on the index.
        Note: currently expects values in this index to be unique.

    ser_sec : ~`pandas.Series`
        The secondary flare list, that will be compared to the primary (reference).

    windows : `list`
        The list of windows to search in, these get progressively larger.
        Ideally you want the increase per window to be small enough to only add
        a single match.

    Returns
    -------
    ser_matched : ~`pandas.Series`
        The list of matches between the primary (reference) and the secondary list.
        The index is the index from the secondary, the values are the associated
        index from the primary (reference) list.

    ser_not_in_primary : ~`pandas.Series`
        The list of flares in the secondary and not in the primary (reference).
        The index is the index from the secondary, the values are all NaN.
    """
    ####print('ser_pri: '+str(len(ser_pri)))
    ####print(ser_pri)
    # Holders for matched/unmatched flares
    ser_matched = pd.Series()
    ser_not_in_primary = pd.Series(data=np.nan, index=ser_sec.index)

    #print('compare_ser_results: windows: '+str(windows))

    # For each of the values in the primary (reference) dataset
    ####print('\n')
    ####print('str(len(ser_sec)): '+str(len(ser_sec)))
    ####print(ser_sec)
    ####print('\n\n')
    for i in range(0, len(ser_sec)):
        dt_det = ser_sec.index[i]
        #print('dt_det: '+str(dt_det)+'\n')

        # Check if the flare detection is the same as a HEK entry
        if dt_det in ser_pri:
            #print('here0')
            ser_matched[dt_det] = dt_det
            ser_not_in_primary = ser_not_in_primary.drop(dt_det)
        else:
            ####print('here1')
            # If it's not exactly the same, check within the given windows
            for td_window in windows:
                ####print('here2')
                dt_window_min = dt_det - td_window
                dt_window_max = dt_det + td_window
                ####print('dt_window_min: '+str(dt_window_min))
                ####print('dt_window_max: '+str(dt_window_max))

                # truncate to this window and see if any HEK results are present
                ser_pri_trun = ser_pri.truncate(dt_window_min, dt_window_max)
                ####print('ser_pri_trun: '+str(ser_pri_trun))

                if len(ser_pri_trun) == 1:
                    #print('here3')
                    # We have a unique match
                    ser_matched[dt_det] = ser_pri_trun.index[0]
                    ser_not_in_primary = ser_not_in_primary.drop(dt_det)
                    break
                elif len(ser_pri_trun) == 2:
                    ####print('here4')
                    # We have 2 matches, if diff between windows is small enough
                    # these are equidistant from the detection. Choose the first.
                    ser_matched[dt_det] = ser_pri_trun.index[0]
                    ser_not_in_primary = ser_not_in_primary.drop(dt_det)
                    break
                elif len(ser_pri_trun) > 2:
                    ####print('here5')
                    # Too many matches, windows are 2 large.
                    break

    return ser_matched, ser_not_in_primary


def compare_to_HEK(df_all_hek, df_all_detections, windows=[timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=3)]):
    """

    Parameters
    ----------
    df_all_hek : ~`pandas.DataFrame`
        The dataframe containing HEK flare results for the same timerange and
        the detections, indexed by peaktimes.

        Note: HEK sometimes has multiple flares with the same peaktime, generally
        one of these will end up being ill-formed, for example have 0 duration.

    df_all_detections : ~`pandas.DataFrame`
        The pandas DataFrame of flare detections, that will be compared to the
        HEK (reference), should be indexed by flare peaktime and can optionally
        have start/end times.

    windows : `list`
        The list of windows to search in, these get progressively larger.
        Ideally you want the increase per window to be small enough to only add
        a single match.

    Returns
    -------
    df_out  : ~`pandas.DataFrame`
        The list of matches between the primary (reference) and the secondary list.
        The index is the index from the secondary, the values are the associated
        index from the primary (reference) list.

    ser_not_in_primary : ~`pandas.Series`
        The list of detected flares not in the HEK (reference) data.
        The index is the index from the detection data, the values are all NaN.

    ser_not_in_secondary : ~`pandas.Series`
        The list of flares in the HEK (reference) data and not in the detections.
        The index is the index from the HEK data, the values are all NaN.
    """
    # Does HEK data have any duplicates
    if not (np.all(np.invert(df_all_hek.index.duplicated()))):
        warnings.warn("HEK data has flares with dupicate peak times. First entries have been kept.")
        df_all_hek = df_all_hek.groupby(df_all_hek.index).first()

    # Find matches
    ser_matched, ser_not_in_primary, ser_not_in_secondary = compare_to_primary(df_all_hek['event_peaktime'], df_all_detections['event_peaktime'], windows=[timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=3)])

    # For matches, find values of event_peaktime, fl_peakflux, event_starttime, event_endtime, ... relative to HEK
    ####print('df_all_hek.index: \n'+str(df_all_hek.index))
    ####print('ser_matched.values: \n'+str(ser_matched.values))
    ####print('type(ser_matched.values): \n'+str(type(ser_matched.values)))
    #### The fllowing doesn't work, I'm doing a hack to get arround it
    #df_matched_hek = df_all_hek[ser_matched.values]
    #df_matched_det = df_all_detections[ser_matched.index]
    lis_indices = []
    for dt_index in ser_matched.values:
        lis_indices.append(df_all_hek.index.get_loc(dt_index))
    df_matched_hek = pd.DataFrame(df_all_hek.iloc[lis_indices])
    #print('\n\n')
    #print(df_matched_hek)
    #print('\n\n')
    lis_indices = []
    for dt_index in ser_matched.index:
        lis_indices.append(ser_matched.index.get_loc(dt_index))
    df_matched_det = pd.DataFrame(df_all_detections.iloc[lis_indices])
    arr_rel_event_peaktime = pd.to_datetime(ser_matched.index) - pd.to_datetime(ser_matched.values)
    ####print('arr_rel_event_peaktime:\n'+str(arr_rel_event_peaktime))

    # If we haven't calculated the HEK peak flux from the flare class then do so
    if not 'fl_peakflux' in df_all_hek.columns:
        df_matched_hek['fl_peakflux'] = utils.flare_class_to_intensity(df_matched_hek['fl_goescls'].values)
        #print('HEK fl_peakflux:\n'+str(df_matched_hek['fl_peakflux']))
    arr_rel_fl_peakflux = df_matched_det['fl_peakflux'].values / df_matched_hek['fl_peakflux'].values

    # Note: if the detected flares don't have start/end times determined, then return zero values. (would rather NaN's)
    if 'event_starttime' in df_all_detections.columns:
        arr_rel_event_starttime = pd.to_datetime(df_matched_det['event_starttime'].values) - pd.to_datetime(df_matched_hek['event_starttime'].values)
    else:
        arr_rel_event_starttime = np.zeros([len(df_matched_hek)])
    if 'event_endtime' in df_all_detections.columns:
        arr_rel_event_endtime = pd.to_datetime(df_matched_det['event_endtime'].values) - pd.to_datetime(df_matched_hek['event_endtime'].values)
    else:
        arr_rel_event_endtime = np.zeros([len(df_matched_hek)])

    # Make the dataframe for the relative values
    dic_matched_rel_data = { 'det_event_peaktime': ser_matched.index,
                             'hek_event_peaktime': ser_matched.values,
                             'fl_peakflux_rel_hek': arr_rel_fl_peakflux,
                             'event_peaktime_rel_hek': arr_rel_event_peaktime,
                             'event_peaktime_rel_hek_(s)': (arr_rel_event_peaktime/1000000000.0).astype(int),
                             'event_peaktime_rel_hek_(min)': (arr_rel_event_peaktime/60000000000.0).astype(int),
                             'event_starttime_rel_hek': arr_rel_event_starttime,
                             'event_starttime_rel_hek_(s)': (arr_rel_event_starttime/1000000000.0).astype(int),
                             'event_starttime_rel_hek_(min)': (arr_rel_event_starttime/60000000000.0).astype(int),
                             'event_endtime_rel_hek': arr_rel_event_endtime,
                             'event_endtime_rel_hek_(s)': (arr_rel_event_endtime/1000000000.0).astype(int),
                             'event_endtime_rel_hek_(min)': (arr_rel_event_endtime/60000000000.0).astype(int)
                             }
    df_out = pd.DataFrame(data=dic_matched_rel_data, index=ser_matched.index)

    return df_out, ser_not_in_primary, ser_not_in_secondary

def compare_many_to_primary(data_ref, dic_datasets, windows=[timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=3)]):
    """
    Basic function to compare multiple datasets to one primary (reference).

    ####This isn't complete yet!
    """
    # Sanatise inputs
    ser_ref = data_ref
    if isinstance(ser_ref, pd.DataFrame):
        ser_ref = pd.Series(data=np.nan, index=ser_ref.index)
    elif isinstance(ser_ref, np.ndarray):
        ser_ref = pd.Series(data=np.nan, index=pd.to_datetime(ser_ref))

    # Check all datasets
    lis_ser_data = []
    for dataset in datasets:
        ser_dataset = dataset
        if isinstance(ser_dataset, pd.DataFrame):
            ser_dataset = pd.Series(data=np.nan, index=ser_data.index)
        elif isinstance(ser_dataset, np.ndarray):
            ser_dataset = pd.Series(data=np.nan, index=pd.to_datetime(ser_data))
        lis_ser_data.append(ser_dataset)

    # Holders for matched/unmatched flares
    dic_temp = {}

    ser_matched = pd.Series()
    ser_unmatched = pd.Series(data=np.nan, index=ser_data.index)

    # For each dataset.
    for dataset in datasets:
        # For each of the values in the primary (reference) data
        for i in range(0, len(ser_data)):
            dt_det = ser_data.index[i]

            # Check if the flare detection is the same as a HEK entry
            if dt_det in ser_ref.index:
                ser_matched[dt_det] = dt_det
                ser_unmatched = ser_unmatched.drop(dt_det)
            else:
                # If not exactly the same, check within the given windows
                for td_window in windows:
                    dt_window_min = dt_det - td_window
                    dt_window_max = dt_det + td_window

                    # truncate to this window and see if any HEK results are present
                    ser_ref_trun = ser_ref.truncate(dt_window_min, dt_window_max)
                    if len(ser_ref_trun) == 1:
                        # We have a unique match
                        ser_matched[dt_det] = ser_ref_trun.index[0]
                        ser_unmatched = ser_unmatched.drop(dt_det)
                        break
                    elif len(ser_ref_trun) == 2:
                        # We have 2 matches, if diff between windows is small enough
                        # these are equidistant from the detection. Choose the first.
                        ser_matched[dt_det] = ser_ref_trun.index[0]
                        ser_unmatched = ser_unmatched.drop(dt_det)
                        break
                    elif len(ser_ref_trun) > 2:
                        # Too many matches, windows are 2 large.
                        break
    # Return the results
    return df_matched, ser_unmatched