# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:41:27 2017

@author: Alex
"""

import pandas as pd

def from_dataframe(data, indices):
    """
    Retrieves one or more results from a pandas DataFrame using the .iloc method
    on the given index/indices.
    This is a convenience function, as sometimes pandad DataFrames has issues
    using a datetime/timestamp index.
    I'm looking for better solutions, but right now this is a quick fix.

    Parameters
    ----------
    data : `~pandas.DataFrame` or `~pandas.Series`
        The DataFrame to look in.

    indices : `list` of `datetime.datetime` or timestamp
        The list of indices that are to be retrieved.

    Returns
    -------
    result : ~`pandas.DataFrame`
        The pandas.DataFrame with all the matching results.
    """
    lis_i_indices = []
    for dt_index in indices:
        lis_i_indices.append(data.index.get_loc(dt_index))

    return pd.DataFrame(data.iloc[lis_i_indices])