# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:08:44 2018

@author: alex_
"""

#==============================================================================
# Test Save/Load Functionality
#==============================================================================

#import sunpy.data
#sunpy.data.download_sample_data()

import os
import sunpy.data.sample
from sunpy import timeseries as ts

def saveLoadTest(ts_str_load, str_path, source):
    #
    ts_original = ts.TimeSeries(ts_str_load, source=source)

    # Save the file
    str_savepath = os.path.join(str_path, source+'.fits')
    ts_original.save(str_savepath)
    ts_loaded = ts.TimeSeries()

    return ts_original == ts_loaded

# Make the sample TS
lis_data = [
[sunpy.data.sample.GOES_XRS_TIMESERIES, 'xrs'],
[sunpy.data.sample.EVE_TIMESERIES, 'EVE'],
[sunpy.data.sample.GBM_TIMESERIES, 'GBMSummary'],
[sunpy.data.sample.LYRA_LEVEL3_TIMESERIES, 'lyra'],
[sunpy.data.sample.NOAAINDICES_TIMESERIES, 'NOAAIndices'],
[sunpy.data.sample.NOAAPREDICT_TIMESERIES, 'NOAAPredictIndices'],
[sunpy.data.sample.NORH_TIMESERIES, 'NoRH'],
[sunpy.data.sample.RHESSI_TIMESERIES, 'rhessi'],
]
str_temppath = '\\fits'


for val in lis_data:
    boo_result = saveLoadTest(val[0], str_temppath, source=val[1])

    if boo_result:
        print('pass: '+val[1])
    else:
        print('fail: '+val[1])






"""
def saveLoadTest(ts_str_load, str_path, source):
    ts_data.save(str_path)
    ts_loaded = ts.TimeSeries(str_path)

    if ts_data == ts_ loaded:
        print('pass: '+str(ts_goes.__class__))
    else:
        print('pass: '+str(ts_goes.__class__))
"""

"""
def test_save_load_eve(eve_test_ts):
    afilename = 'test_eve.fits'
    eve_test_ts.save(afilename)
    eve_test_ts_loaded = sunpy.timeseries.TimeSeries(afilename)
    assert eve_test_ts == eve_test_ts_loaded

def test_save_load_fermi_gbm(fermi_gbm_test_ts):
    afilename = 'test_gbm.fits'
    fermi_gbm_test_ts.save(afilename)
    fermi_gbm_test_ts_loaded = sunpy.timeseries.TimeSeries(afilename)
    assert fermi_gbm_test_ts == fermi_gbm_test_ts_loaded

def test_save_load_norh(norh_test_ts):
    afilename = 'test_norh.fits'
    norh_test_ts.save(afilename)
    norh_test_ts_loaded = sunpy.timeseries.TimeSeries(afilename)
    assert norh_test_ts == norh_test_ts_loaded

def test_save_load_goes(goes_test_ts):
    afilename = 'test_goes.fits'
    goes_test_ts.save(afilename)
    goes_test_ts_loaded = sunpy.timeseries.TimeSeries(afilename)
    assert goes_test_ts == goes_test_ts_loaded

def test_save_load_lyra(lyra_test_ts):
    afilename = 'test_lyra.fits'
    lyra_test_ts.save(afilename)
    lyra_test_ts_loaded = sunpy.timeseries.TimeSeries(afilename)
    assert lyra_test_ts == lyra_test_ts_loaded

def test_save_load_rhessi(rhessi_test_ts):
    afilename = 'test_rhessi.fits'
    rhessi_test_ts.save(afilename)
    rhessi_test_ts_loaded = sunpy.timeseries.TimeSeries(afilename)
    assert rhessi_test_ts == rhessi_test_ts_loaded

def test_save_load_noaa_ind(noaa_ind_test_ts):
    afilename = 'test_noaa_ind.fits'
    noaa_ind_test_ts.save(afilename)
    noaa_ind_test_ts_loaded = sunpy.timeseries.TimeSeries(afilename)
    assert noaa_ind_test_ts == noaa_ind_test_ts_loaded

def test_save_load_noaa_pre(noaa_pre_test_ts):
    afilename = 'test_noaa_pre.fits'
    noaa_pre_test_ts.save(afilename)
    noaa_pre_test_ts_loaded = sunpy.timeseries.TimeSeries(afilename)
    assert noaa_pre_test_ts == noaa_pre_test_ts_loaded

def test_save_load_generic(generic_ts):
    afilename = 'test_generic.fits'
    generic_ts.save(afilename)
    generic_ts_loaded = sunpy.timeseries.TimeSeries(afilename)
    assert generic_ts == generic_ts_loaded

def test_save_load_table(table_ts):
    afilename = 'test_.fits'
    table_ts.save(afilename)
    table_ts_loaded = sunpy.timeseries.TimeSeries(afilename)
    assert table_ts == table_ts_loaded
"""

#==============================================================================
# Test Other Functions
#==============================================================================
