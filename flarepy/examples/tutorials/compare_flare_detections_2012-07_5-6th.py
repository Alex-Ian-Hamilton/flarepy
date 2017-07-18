# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:53:17 2017

@author: Alex
"""

# Basic imports
import pandas as pd
import numpy as np
from datetime import timedelta

# Advanced imports
import flarepy.utils as utils
import flarepy.flare_detection as det
from sunpy.lightcurve import GOESLightCurve
from sunpy.time import TimeRange

# Parameters
# Specify the start/end times
str_start = '2012-07-05 00:00:00'
str_mid = '2012-07-06 00:00:00' # Only necessary because only DL GOES for single days
str_end = '2012-07-07 00:00:00'

str_save_path = 'C:\\flare_outputs\\2017-07-18\\'

############
#
#   Download GOES XRS Data
#
############

# Get and open GOES data
h5 = pd.HDFStore('C:\\goes_h5\\2012_goes.h5')

lc_goes_5th = GOESLightCurve.create(TimeRange(str_start, str_mid))
lc_goes_6th = GOESLightCurve.create(TimeRange(str_mid, str_end))
df_goes_XRS = pd.concat([lc_goes_5th.data, lc_goes_6th.data])


############
#
#   XRSA Data Pre-Processing
#   Note: not used in the flare detection, just for the plots.
#
############

# Get raw dataset as a series and make a mask
ser_xrsa_raw = df_goes_XRS['xrsa'].truncate(str_start, str_end)
ser_xrsa_raw_mask = pd.Series(data=np.logical_or(np.isnan(ser_xrsa_raw.values), ser_xrsa_raw.values == 0.0), index=ser_xrsa_raw.index)
ser_xrsa_raw_int = ser_xrsa_raw.replace({0.0:np.nan}).interpolate()

# Resample
str_bins = '60S'
ser_xrsa_raw_int_60S = ser_xrsa_raw_int.resample(str_bins).median()
ser_xrsa_raw_int_60S_mask = ser_xrsa_raw_mask.resample(str_bins).max()

# Rebin
int_cart = 5
ser_xrsa_raw_int_60S_box5 = ser_xrsa_raw_int_60S.rolling(int_cart).mean()
ser_xrsa_raw_int_60S_box5 = ser_xrsa_raw_int_60S_box5[int_cart - 1: 1- int_cart] # remove NaNs
ser_xrsa_raw_int_60S_mask = ser_xrsa_raw_int_60S_mask[int_cart - 1: 1- int_cart]

# Make series for plots (basically making nan holes where data gaps were)
ser_xrsa_plt_fil = pd.Series(ser_xrsa_raw_int_60S_box5)
ser_xrsa_plt_fil.iloc[np.where(ser_xrsa_raw_int_60S_mask != 0.0)] = np.nan
ser_xrsa_plt_raw = pd.Series(ser_xrsa_raw)
ser_xrsa_plt_raw.iloc[np.where(ser_xrsa_raw_mask != 0.0)] = np.nan


############
#
#   XRSB Data Pre-Processing
#
############

# Get raw dataset as a series and make a mask
ser_xrsb_raw = df_goes_XRS['xrsb'].truncate(str_start, str_end)
ser_xrsb_raw_mask = pd.Series(data=np.logical_or(np.isnan(ser_xrsb_raw.values), ser_xrsb_raw.values == 0.0), index=ser_xrsb_raw.index)
ser_xrsb_raw_int = ser_xrsb_raw.replace({0.0:np.nan}).interpolate()

# Resample
str_bins = '60S'
ser_xrsb_raw_int_60S = ser_xrsb_raw_int.resample(str_bins).median()
ser_xrsb_raw_int_60S_mask = ser_xrsb_raw_mask.resample(str_bins).max()

# Rebin
int_cart = 5
ser_xrsb_raw_int_60S_box5 = ser_xrsb_raw_int_60S.rolling(int_cart).mean()
ser_xrsb_raw_int_60S_box5 = ser_xrsb_raw_int_60S_box5[int_cart - 1: 1- int_cart] # remove NaNs
ser_xrsb_raw_int_60S_mask = ser_xrsb_raw_int_60S_mask[int_cart - 1: 1- int_cart]
ser_xrsb_raw_int_60S_box5_int = ser_xrsb_raw_int_60S_box5.interpolate()

# Make series for plots (basically making nan holes where data gaps were)
ser_xrsb_plt_fil = pd.Series(ser_xrsb_raw_int_60S_box5)
ser_xrsb_plt_fil.iloc[np.where(ser_xrsb_raw_int_60S_mask != 0.0)] = np.nan
ser_xrsb_plt_raw = pd.Series(ser_xrsb_raw)
ser_xrsb_plt_raw.iloc[np.where(ser_xrsb_raw_mask != 0.0)] = np.nan


############
#
#   Download HEK Peaks
#
############

df_hek = utils.get_hek_goes_flares(str_start, str_end, fail_safe=True)
arr_hek_peaks = utils.flare_class_to_intensity(df_hek['fl_goescls'].values)
ser_hek_peaks = pd.Series(data=arr_hek_peaks,index=df_hek.index)
ser_hek_peaks.to_csv(str_save_path+'2012_july_5-6th_hek.csv')


############
#
#   Fnd All Minima (For Pre-flare)
#
############

ser_minima = ser_xrsb_raw_int_60S_box5[utils.find_minima_fast(ser_xrsb_raw_int_60S_box5.interpolate().values)]


############
#
#   Calculate CWT Peaks
#
############

# Now use boxcart method
# CWT parameters
int_max_width = 100 # Could use int_max_width = 50 to increase sensitivity
arr_cwt_widths = np.arange(1,int_max_width)

# Get the peaks
df_peaks_cwt = det.get_flare_peaks_cwt(ser_xrsb_raw_int_60S_box5.interpolate(), raw_data=ser_xrsb_raw_int_60S.interpolate(), widths=arr_cwt_widths, get_energies=True)

# Get estimated flare start/end times (use closest local minima)


# Get the flare energies


############
#
#   Calculate 4-min Rise Peaks
#
############

# You can quickly find flares using the 4min Rise method
df_peaks_4min = det.get_flares_goes_event_list(ser_xrsb_raw_int_60S_box5.interpolate(), N=4, start_threshold=0.4, end_threshold=0.5, raw_data=None, get_duration=True, get_energies=True)


############
#
#   Save adata and plot
#
############

# Save data
df_peaks_cwt.to_csv(str_save_path+'2012_july_5-6th___cwt_peaks_[1-'+str(int_max_width)+'].csv')
df_peaks_4min.to_csv(str_save_path+'2012_july_5-6th___4min_peaks.csv')
df_hek.to_csv(str_save_path+'2012_july_5-6th___HEK_events.csv')

# Plot this as a single figure
fig = utils.plot_goes({'xrsa':ser_xrsa_plt_fil, 'xrsa - raw': ser_xrsa_plt_raw, 'xrsb': ser_xrsb_plt_fil, 'xrsb - raw': ser_xrsb_plt_raw},
              {'CWT': df_peaks_cwt['fl_peakflux'], 'HEK': ser_hek_peaks, '4-min Rise': df_peaks_4min['fl_peakflux']},
              title='5-6th July 2012 - GOES XRS Data',#title='2 X-Class Flares in March 2012 - GOES XRS Data (CWT: 1 to ' + str(int_max_width) + ' for ' + str(len(df_peaks_cwt['xrsb'])) + ' peaks)',
              ylim=(1e-9, 1e-3))
fig.savefig(str_save_path+'2012_july_5-6th_cwt_[1-'+str(int_max_width)+'].png', dpi=900, bbox_inches='tight')

# Now a figure per day
# July 5th
fig = utils.plot_goes({'xrsa':ser_xrsa_plt_fil.truncate(str_start, str_mid), 'xrsa - raw': ser_xrsa_plt_raw.truncate(str_start, str_mid), 'xrsb': ser_xrsb_plt_fil.truncate(str_start, str_mid), 'xrsb - raw': ser_xrsb_plt_raw.truncate(str_start, str_mid)},
              {'CWT': df_peaks_cwt['fl_peakflux'].truncate(str_start, str_mid), 'HEK': ser_hek_peaks.truncate(str_start, str_mid), '4-min Rise': df_peaks_4min['fl_peakflux'].truncate(str_start, str_mid)},
              title='5th July 2012 - GOES XRS Data',#title='2 X-Class Flares in March 2012 - GOES XRS Data (CWT: 1 to ' + str(int_max_width) + ' for ' + str(len(df_peaks_cwt['xrsb'])) + ' peaks)',
              ylim=(1e-9, 1e-3))
fig.savefig(str_save_path+'2012_july_5th_cwt_[1-'+str(int_max_width)+'].png', dpi=900, bbox_inches='tight')
fig = utils.plot_goes({'xrsa':ser_xrsa_plt_fil.truncate(str_mid, str_end), 'xrsa - raw': ser_xrsa_plt_raw.truncate(str_mid, str_end), 'xrsb': ser_xrsb_plt_fil.truncate(str_mid, str_end), 'xrsb - raw': ser_xrsb_plt_raw.truncate(str_mid, str_end)},
              {'CWT':df_peaks_cwt['fl_peakflux'].truncate(str_mid, str_end), 'HEK': ser_hek_peaks.truncate(str_mid, str_end), '4-min Rise': df_peaks_4min['fl_peakflux'].truncate(str_mid, str_end)},
              title='6th July 2012 - GOES XRS Data',#title='2 X-Class Flares in March 2012 - GOES XRS Data (CWT: 1 to ' + str(int_max_width) + ' for ' + str(len(df_peaks_cwt['xrsb'])) + ' peaks)',
              ylim=(1e-9, 1e-3))
fig.savefig(str_save_path+'2012_july_6th_cwt_[1-'+str(int_max_width)+'].png', dpi=900, bbox_inches='tight')

windows = [timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=3)]
df_matched, df_unmatched = utils.get_equiv_hek_results(df_peaks_cwt, hek_data=df_hek, windows=windows)
df_matched.to_csv(str_save_path+'2012_july_5-6th_cwt_peaks_[1-'+str(int_max_width)+']_matched_to_hek.csv')
