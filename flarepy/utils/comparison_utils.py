# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:27:14 2017

@author: Alex
"""

import pandas as pd
import numpy as np

def compare_to_primary(data_ref, data, windows=[timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=3)]):
    """
    Basic function to compare 2 datasets.
    """
    # Sanatise inputs
    ser_ref = data_ref
    if isinstance(ser_ref, pd.DataFrame):
        ser_ref = pd.Series(data=np.nan, index=ser_ref.index)
    elif isinstance(ser_ref, np.ndarray):
        ser_ref = pd.Series(data=np.nan, index==pd.to_datetime(ser_ref))
    ser_data = data
    if isinstance(ser_data, pd.DataFrame):
        ser_data = pd.Series(data=np.nan, index=ser_data.index)
    elif isinstance(ser_ref, np.ndarray):
        ser_data = pd.Series(data=np.nan, index==pd.to_datetime(ser_data))

    # Holders for matched/unmatched flares
    ser_matched = pd.Series()
    ser_unmatched = pd.Series(data=np.nan, index=ser_data.index)

    # For each of the values in the reference data
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


def compare_many_to_primary(data_ref, datasets, windows=[timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=3)]):
    """
    Basic function to compare 2 datasets.
    """
    # Sanatise inputs
    ser_ref = data_ref
    if isinstance(ser_ref, pd.DataFrame):
        ser_ref = pd.Series(data=np.nan, index=ser_ref.index)
    elif isinstance(ser_ref, np.ndarray):
        ser_ref = pd.Series(data=np.nan, index==pd.to_datetime(ser_ref))
    # Check all datasets
    lis_ser_data = []
    for dataset in datasets:
        ser_dataset = dataset
        if isinstance(ser_dataset, pd.DataFrame):
            ser_dataset = pd.Series(data=np.nan, index=ser_data.index)
        elif isinstance(ser_dataset, np.ndarray):
            ser_dataset = pd.Series(data=np.nan, index==pd.to_datetime(ser_data))
        lis_ser_data.append(ser_dataset)

    # Holders for matched/unmatched flares
    ser_matched = pd.Series()
    ser_unmatched = pd.Series(data=np.nan, index=ser_data.index)

    # For each dataset.
    for dataset in datasets:
        # For each of the values in the reference data
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