# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:53:17 2017

@author: Alex
"""

# Basic imports
import pandas as pd
import numpy as np
from datetime import timedelta
import os.path
import astropy.units as u
import datetime

# Advanced imports
import flarepy.utils as utils
import flarepy.flare_detection as det
import flarepy.plotting as plot
from sunpy.lightcurve import GOESLightCurve
from sunpy.time import TimeRange

# Parameters
# Specify the start/end times
str_start = '2014-03-28 00:00:00'
str_mid = '2014-03-29 00:00:00' # Only necessary because only DL GOES for single days
str_end = '2014-03-30 00:00:00'

str_save_path = os.path.join('D:\\','flare_outputs',datetime.datetime.now().strftime("%Y-%m-%d"))
str_plots_dir = os.path.join(str_save_path,'plots_other')
str_comparisons_dir = os.path.join(str_save_path,'comparisons')
str_detections_dir = os.path.join(str_save_path,'detections')
str_miniplots_dir = os.path.join(str_save_path,'plots_peaks')
str_dayplots_dir = os.path.join(str_save_path,'plots_days')


str_file_prefix = '2014_mar_28-29th___'
str_day_1_prefix = '2014_mar_28th___'
str_day_2_prefix = '2014_mar_29th___'
str_heading = '28-29th mar 2014'
str_day_1_heading = '28th mar 2014'
str_day_2_heading = '29th mar 2014'

# Create any folders necessary
lis_str_folders = [str_save_path,
                   str_plots_dir,
                   str_comparisons_dir,
                   str_detections_dir,
                   str_miniplots_dir,
                   str_dayplots_dir]
for str_folder in lis_str_folders:
    if not os.path.exists(str_folder):
        os.makedirs(str_folder)

############
#
#   Download GOES XRS Data
#
############

# Get and open GOES data
#str_year = str_start[0:4]
#h5 = pd.HDFStore('C:\\goes_h5\\str_year_goes.h5')

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

# Make series for plots (basically making nan holes where data gaps were)
ser_xrsa_plt_raw = pd.Series(ser_xrsa_raw)
ser_xrsa_plt_raw.iloc[np.where(ser_xrsa_raw_mask != 0.0)] = np.nan

# Pre-process the data
ser_xrsa_raw_int_60S_box5, ser_xrsa_raw_int_60S, ser_xrsa_plt_fil = utils.pre_process(ser_xrsa_raw, resample_bins=60*u.s, resample_method='median', average_method='mean', int_cart=5)

############
#
#   XRSB Data Pre-Processing
#
############

# Get raw dataset as a series and make a mask
ser_xrsb_raw = df_goes_XRS['xrsb'].truncate(str_start, str_end)
ser_xrsb_raw_mask = pd.Series(data=np.logical_or(np.isnan(ser_xrsb_raw.values), ser_xrsb_raw.values == 0.0), index=ser_xrsb_raw.index)
ser_xrsb_raw_int = ser_xrsb_raw.replace({0.0:np.nan}).interpolate()

# Make series for plots (basically making nan holes where data gaps were)
ser_xrsb_plt_raw = pd.Series(ser_xrsb_raw)
ser_xrsb_plt_raw.iloc[np.where(ser_xrsb_raw_mask != 0.0)] = np.nan

# Pre-process the data
ser_xrsb_raw_int_60S_box5, ser_xrsb_raw_int_60S, ser_xrsb_plt_fil = utils.pre_process(ser_xrsb_raw, resample_bins=60*u.s, resample_method='median', average_method='mean', int_cart=5)


############
#
#   Download HEK Peaks
#
############

# Check if the file is already present
if os.path.isfile(os.path.join(str_detections_dir,str_file_prefix+'_hek.csv')):
    # Simply open the file
    df_hek = pd.read_csv(os.path.join(str_detections_dir,str_file_prefix+'_hek.csv'))
    df_hek.index = pd.to_datetime(df_hek['event_peaktime'].values)
else:
    # Download
    df_hek = utils.get_hek_goes_flares(str_start, str_end, fail_safe=False, add_peaks=True)
    df_hek.to_csv(os.path.join(str_detections_dir,str_file_prefix+'_hek.csv'))
ser_hek_peaks = df_hek['fl_peakflux']

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
int_max_width = 50 # Could use int_max_width = 50 to increase sensitivity
arr_cwt_widths = np.arange(1,int_max_width)

# Get the peaks
df_peaks_cwt = det.get_flare_peaks_cwt(ser_xrsb_raw_int_60S_box5.interpolate(), raw_data=ser_xrsb_raw_int_60S.interpolate(), widths=arr_cwt_widths, get_energies=True)

# Save
df_peaks_cwt.to_csv(os.path.join(str_detections_dir,str_file_prefix+'CWT_detections.csv'), header=True)

# Get estimated flare start/end times (use closest local minima)


# Get the flare energies


############
#
#   Calculate 4-min Rise Peaks
#
############

# You can quickly find flares using the 4min Rise method
df_peaks_4min = det.get_flares_goes_event_list(ser_xrsb_raw_int_60S_box5.interpolate(), N=4, start_threshold=0.4, end_threshold=0.5, raw_data=None, get_duration=True, get_energies=True)

# Save
df_peaks_4min.to_csv(os.path.join(str_detections_dir,str_file_prefix+'4min_rise_detections.csv'), header=True)


############
#
#   Compare results
#
############

# Allow matches if within 3 mins, split into 3 windows to prioritise closer matches
lis_windows = [timedelta(minutes=1), timedelta(minutes=2), timedelta(minutes=3)]

# Get the basic matches
#df_cwt_matched_hek, ser_in_hek_not_in_cwt, ser_in_cwt_not_in_hek = utils.compare_to_primary(df_hek, df_peaks_cwt, windows=lis_windows)
#df_cwt_matched_hek.to_csv(str_save_path+str_file_prefix+'temp1.csv')
#ser_in_hek_not_in_cwt.to_csv(str_save_path+str_file_prefix+'temp2.csv')
#ser_in_cwt_not_in_hek.to_csv(str_save_path+str_file_prefix+'temp3.csv')
#df_4min_matched_hek, ser_in_hek_not_in_4min, ser_in_4min_not_in_hek = utils.compare_to_primary(df_hek, df_peaks_4min.groupby(df_peaks_4min.index).last(), windows=lis_windows)
#df_4min_matched_hek.rename("HEK peaktime").to_csv(str_save_path+str_file_prefix+'temp4.csv', header=True, index_label='4 min peaktime')

# Get the detailed match details
df_cwt_compared_to_hek, ser_in_cwt_not_in_hek, ser_in_hek_not_in_cwt = utils.compare_to_HEK(df_hek, df_peaks_cwt, windows=lis_windows)
df_4min_compared_to_hek, ser_in_4min_not_in_hek, ser_in_hek_not_in_4min = utils.compare_to_HEK(df_hek, df_peaks_4min.groupby(df_peaks_4min.index).last(), windows=lis_windows)
# Note: for the 4min rise method we have had to remove duplicates before comparison.

# Save these
df_cwt_compared_to_hek.to_csv(os.path.join(str_comparisons_dir,str_file_prefix+'cwt_compared_to_hek.csv'), header=True)
df_4min_compared_to_hek.to_csv(os.path.join(str_comparisons_dir,str_file_prefix+'compared_to_hek.csv'), header=True)
ser_in_hek_not_in_cwt.rename("N/A").to_csv(os.path.join(str_comparisons_dir,str_file_prefix+'HEK_flares_not_in_CWT_detections.csv'), header=True, index_label='HEK peaktime')
ser_in_hek_not_in_4min.rename("N/A").to_csv(os.path.join(str_comparisons_dir,str_file_prefix+'HEK_flares_not_in_4min_detections.csv'), header=True, index_label='HEK peaktime')
ser_in_cwt_not_in_hek.rename("N/A").to_csv(os.path.join(str_comparisons_dir,str_file_prefix+'CWT_detections_not_in_HEK.csv'), header=True, index_label='CWT peaktime')
ser_in_4min_not_in_hek.rename("N/A").to_csv(os.path.join(str_comparisons_dir,str_file_prefix+'4min_detections_not_in_HEK.csv'), header=True, index_label='4 min peaktime')

# The statistsics for all windows
dic_results = { 'bin=60s;HEK - Reference;': ser_hek_peaks,
                'bin=60s;4 min;N=4': df_peaks_4min['fl_peakflux'],
                'bin=60s;CWT;Widths=[1,...,49]': df_peaks_cwt['fl_peakflux'] }
df_varied_windows_stats = utils.get_varied_window_stats(ser_hek_peaks, dic_results, windows=lis_windows)
df_varied_windows_stats.to_csv(os.path.join(str_comparisons_dir,str_file_prefix+'varied_windows_statistsics.csv'), header=True)
fig_stats = plot.plot_varied_window_stats(df_varied_windows_stats, percentage=False)
fig_stats.savefig(os.path.join(str_plots_dir,str_file_prefix+'varied_windows_statistsics.png'), dpi=900, bbox_inches='tight')


############
#
#   Full Plots
#
############

# Plot this as a single figure
fig = plot.plot_goes({'xrsa':ser_xrsa_plt_fil, 'xrsa - raw': ser_xrsa_plt_raw, 'xrsb': ser_xrsb_plt_fil, 'xrsb - raw': ser_xrsb_plt_raw},
              {'CWT': df_peaks_cwt['fl_peakflux'], 'HEK': ser_hek_peaks, '4-min Rise': df_peaks_4min['fl_peakflux']},
              title=str_heading+' - GOES XRS Data',
              ylim=(1e-9, 1e-3))
fig.savefig(os.path.join(str_plots_dir,str_file_prefix+'cwt_[1-'+str(int_max_width)+'].png'), dpi=900, bbox_inches='tight')

# Now a figure per day
# July 5th
fig = plot.plot_goes({'xrsa':ser_xrsa_plt_fil.truncate(str_start, str_mid), 'xrsa - raw': ser_xrsa_plt_raw.truncate(str_start, str_mid), 'xrsb': ser_xrsb_plt_fil.truncate(str_start, str_mid), 'xrsb - raw': ser_xrsb_plt_raw.truncate(str_start, str_mid)},
              {'CWT': df_peaks_cwt['fl_peakflux'].truncate(str_start, str_mid), 'HEK': ser_hek_peaks.truncate(str_start, str_mid), '4-min Rise': df_peaks_4min['fl_peakflux'].truncate(str_start, str_mid)},
              title=str_day_1_heading+' - GOES XRS Data',
              ylim=(1e-9, 1e-3))
fig.savefig(os.path.join(str_plots_dir,str_day_1_prefix+'cwt_[1-'+str(int_max_width)+'].png'), dpi=900, bbox_inches='tight')
fig = plot.plot_goes({'xrsa':ser_xrsa_plt_fil.truncate(str_mid, str_end), 'xrsa - raw': ser_xrsa_plt_raw.truncate(str_mid, str_end), 'xrsb': ser_xrsb_plt_fil.truncate(str_mid, str_end), 'xrsb - raw': ser_xrsb_plt_raw.truncate(str_mid, str_end)},
              {'CWT':df_peaks_cwt['fl_peakflux'].truncate(str_mid, str_end), 'HEK': ser_hek_peaks.truncate(str_mid, str_end), '4-min Rise': df_peaks_4min['fl_peakflux'].truncate(str_mid, str_end)},
              title=str_day_2_heading+' - GOES XRS Data',
              ylim=(1e-9, 1e-3))
fig.savefig(os.path.join(str_plots_dir,str_day_2_prefix+'cwt_[1-'+str(int_max_width)+'].png'), dpi=900, bbox_inches='tight')


############
#
#   Mini Plots - For each Flare - This will run slowly
#
############

lis_minifigs = plot.plot_goes_miniplots({'xrsa':ser_xrsa_plt_fil, 'xrsa - raw': ser_xrsa_plt_raw, 'xrsb': ser_xrsb_plt_fil, 'xrsb - raw': ser_xrsb_plt_raw},
              {'CWT': df_peaks_cwt['fl_peakflux'], 'HEK': ser_hek_peaks, '4-min Rise': df_peaks_4min['fl_peakflux']},
              title=str_heading+' - GOES XRS Data',
              ylim=(1e-9, 1e-3),
              miniplot_peaks=ser_hek_peaks[0:1],
              miniplot_windows=timedelta(hours=1),
              miniplot_save_loc=os.path.join(str_miniplots_dir))
