# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:46:28 2017

@author: alex_
"""

import sunpy.timeseries as ts
import sunpy.data.sample
from astropy.time import Time
from astropy.table import Table, Column
import pandas as pd
import os
import astropy.units as u
import numpy as np

import astropy
print(astropy.__version__)
print(astropy.__file__)

goes = ts.TimeSeries(sunpy.data.sample.GOES_XRS_TIMESERIES, source='XRS')
df_data = goes.data[::int(len(goes.data)/7)]

# Make a table of this

# Get the start and end dates
tim_start = Time(df_data.index[0])
tim_start.format = 'isot' # Not currently correct using 'fits'
tim_end = Time(df_data.index[-1])
tim_end.format = 'isot'

tim_other = Time(df_data.index[int(len(df_data)/2)])
tim_other.format = 'isot' # Not currently correct using 'fits'

# First, the metadata
str_sat = goes.meta.metadata[0][2]['telescop']
dic_metadata = {
                'telescop': str_sat,
                'instrume': str_sat,
                #'date': datetime.today().strftime('%Y/%m/%d'),#.strftime('%d/%m/%Y'),
                'DATE-BEG': str(tim_start),
                'DATE-OBS': str(tim_other),
                'DATE-END': str(tim_end),
                #'TSTOP': str(tim_end),
                #'DATE-OBS': df_data.index[0].to_pydatetime().strftime('%Y/%m/%d'),#.strftime('%d/%m/%Y'),
                #'DATE-END': df_data.index[-1].to_pydatetime().strftime('%Y/%m/%d'),#.strftime('%d/%m/%Y'),
                #'timezero':  (df_data.index[0].to_pydatetime() - datetime(1979,1,1)).total_seconds(), #DATE-OBS in seconds from 79/1/1,0
                #'TIME-OBS': '00:00:00.000000',#df_data.index[0].to_datetime().strftime('%H:%M:%S.%f'),
                #'TIME-END': '23:59:59.999999',#df_data.index[-1].to_datetime().strftime('%H:%M:%S.%f')
                'object': 'Sun',
                #bitpix': lc_goes.meta['BITPIX'],#ts_goes.meta.metadata[0][2]['bitpix'],
                'origin': 'SDAC/GSFC',
                #'naxis': 0,
                #'CTYPE1': 'seconds ',#     / seconds into DATE-OBS of 3s interval (see comments)
                #'CTYPE2': 'watts / m^2',#  / in 1. - 8. Angstrom band
                #'CTYPE3': 'watts / m^2',#  / in .5 - 4. Angstrom band
                }



# Make an astropy table to save the data
tbl_from_ts = Table([(df_data.index - df_data.index[0]).total_seconds(), df_data['xrsa'].values, df_data['xrsb'].values], names=('TIME', 'xrsa', 'xrsb'), meta=dic_metadata)
#tbl_data.write('D:\work_data\\goes_xrs_pre_processed\\table.fits', format='fits')
str_path1 = 'fits_to_fits_float_time.fits'
tbl_from_ts.write(str_path1, format='fits', overwrite=True)#os.path.join(str_year_folderpath,str_year+'_'+str_sat.replace(' ','-')+'.fits'), format='fits')

tbl_from_ts_time = Table([Time(df_data.index.to_pydatetime()), df_data['xrsa'].values, df_data['xrsb'].values], names=('TIME', 'xrsa', 'xrsb'), meta=dic_metadata)
#tbl_data.write('D:\work_data\\goes_xrs_pre_processed\\table.fits', format='fits')
str_path2 = 'fits_to_fits_Time_time.fits'
tbl_from_ts_time.write(str_path2, format='fits', overwrite=True)#os.path.join(str_year_folderpath,str_year+'_'+str_sat.replace(' ','-')+'.fits'), format='fits')

tbl_from_fits_float_time = Table.read(str_path1)
tbl_from_fits_Time_time = Table.read(str_path2)





# Make an astropy table to save the data
tbl_from_ts_W_units = Table([np.array((df_data.index - df_data.index[0]).total_seconds()) *u.s, df_data['xrsa'].values * (u.Watt / u.m**2), df_data['xrsb'].values * (u.Watt / u.m**2)], names=('buzz', 'xrsa', 'xrsb'), meta=dic_metadata)
str_path3 = 'fits_to_fits_float_time_W_units.fits'
tbl_from_ts_W_units.write(str_path3, format='fits', overwrite=True)#os.path.join(str_year_folderpath,str_year+'_'+str_sat.replace(' ','-')+'.fits'), format='fits')

tbl_from_ts_time_W_units = Table([Time(df_data.index.to_pydatetime()), df_data['xrsa'].values * (u.Watt / u.m**2), df_data['xrsb'].values * (u.Watt / u.m**2)], names=('TIME', 'xrsa', 'xrsb'), meta=dic_metadata)
#tbl_data.write('D:\work_data\\goes_xrs_pre_processed\\table.fits', format='fits')
str_path4 = 'fits_to_fits_Time_time_W_units.fits'
tbl_from_ts_time_W_units.write(str_path4, format='fits', overwrite=True)#os.path.join(str_year_folderpath,str_year+'_'+str_sat.replace(' ','-')+'.fits'), format='fits')

tbl_from_fits_float_time_W_Units = Table.read(str_path3)
tbl_from_fits_Time_time_W_Units = Table.read(str_path4)

ts_1 = ts.TimeSeries(str_path1)
ts_2 = ts.TimeSeries(str_path2)
ts_3 = ts.TimeSeries(str_path3)
ts_4 = ts.TimeSeries(str_path4)

# Turn these into tables
tbl_1 = ts_1.to_table()
tbl_2 = ts_2.to_table()
tbl_3 = ts_3.to_table()
tbl_4 = ts_4.to_table()

# Save them as TS (uses .to_table and then saves that.)
ts_1.save('ts_1.fits', overwrite=True)
ts_2.save('ts_2.fits', overwrite=True)
ts_3.save('ts_3.fits', overwrite=True)
ts_4.save('ts_4.fits', overwrite=True)

# Load them as TS
ts_1_loaded = ts.TimeSeries('ts_1.fits') # this has a float index
ts_2_loaded = ts.TimeSeries('ts_2.fits') # this has a TIME index
ts_3_loaded = ts.TimeSeries('ts_3.fits')
ts_4_loaded = ts.TimeSeries('ts_4.fits')

# Check these match
ts_1_loaded.data.equals(ts_1.data)
ts_2_loaded.data.equals(ts_2.data)
ts_3_loaded.data.equals(ts_3.data)
ts_4_loaded.data.equals(ts_4.data)
