# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:55:59 2017

@author: alex_
"""

import flarepy.utils as utils
from sunpy.database import Database
import sunpy.database.attrs as dbattrs
import os
import shutil
from datetime import datetime, timedelta
import glob
from sunpy.net import vso
import sunpy.timeseries as ts
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from sunpy import lightcurve as lc # This shouldn't be needed
from astropy import units as u
from astropy.time import Time
import numpy as np
import copy
import flarepy.flare_detection as det

import warnings
warnings.filterwarnings("ignore")

# Parameters
str_path_sour = os.path.join('D:\\','work_data','goes_xrs_source')
str_path_raw_years = os.path.join('D:\\','work_data','goes_xrs_raw_years')
str_path_pre_proc = os.path.join('D:\\','work_data','goes_xrs_pre_processed')
str_path_db_sour = os.path.join('D:\\','work_data','db','db_flarepy_source.sqlite')#db_flarepy_test.sqlite
str_path_db_data_raw = os.path.join('D:\\','work_data','db','db_flarepy_data_raw.sqlite')
str_path_db_data = os.path.join('D:\\','work_data','db','db_flarepy_data.sqlite')
str_path_flare_outputs = os.path.join('D:\\','work_data','flare_outputs')
str_path_log = os.path.join('D:\\','work_data','db','log_goes_sunpy_db_v03.log')
boo_delete_all_gen_files = False
boo_download_data = False
boo_add_sour_to_db = True
boo_concat_data_for_years = True
boo_overwite_years = False
boo_import_years_into_raw_db = True
boo_neaten_data_db = True
boo_pre_proc_years = True
boo_detect_flares = True
boo_verbose = True
tup_str_bins = ('12S','30S','60S',)#'12S', '60S')
tup_int_boxcart_widths = (5, 11, 21)


# Delimiters (for storing parameters in filenames)
str_par = '-'
str_del = '_'

# For timing performance
dt_start = datetime.today()

# Download files?
lis_int_years = list(range(2013,2017))#list(range(1981,2000))
if boo_download_data:
    utils.goes_utils.get_goes_xrs_year_data(lis_int_years, path=str_path_sour)

if boo_delete_all_gen_files:
    if os.path.exists(str_path_log):
        os.remove(str_path_log)
    if os.path.exists(str_path_db_sour):
        os.remove(str_path_db_sour)
    if os.path.exists(str_path_db_data_raw):
        os.remove(str_path_db_data_raw)
    if os.path.exists(str_path_db_data):
        os.remove(str_path_db_data)
    for int_year in lis_int_years:
        if os.path.exists(os.path.join(str_path_raw_years,str(int_year))):
            shutil.rmtree(os.path.join(str_path_raw_years,str(int_year)))
        if os.path.exists(os.path.join(str_path_pre_proc,str(int_year))):
            shutil.rmtree(os.path.join(str_path_pre_proc,str(int_year)))
        if os.path.exists(os.path.join(str_path_flare_outputs,str(int_year))):
            shutil.rmtree(os.path.join(str_path_flare_outputs,str(int_year)))

#db = Database('sqlite:///C:\\goes_data\\db_all_goes.sqlite')
# Make/open the source data database
db_source = Database('sqlite:///'+str_path_db_sour)
db_data_raw = Database('sqlite:///'+str_path_db_data_raw)
db_data = Database('sqlite:///'+str_path_db_data)
#db_detections = Database('sqlite:///C:\\Users\\alex_\\sunpy\\db_flarepy_detections.sqlite')

# Get the folders
lis_str_folders = os.walk(str_path_sour)

# Start the log file
fil_log = open(str_path_log,'a')

# Adding the source data files into the database
if boo_add_sour_to_db:
    dt_start_make_source_db = datetime.today()

    # Add for each year
    for int_year in lis_int_years:
        # Get the folder path
        str_folder_path = os.path.join(str_path_sour,str(int_year))
        print('  ' + str(int_year)+': ',end='')
        fil_log.write('  ' + str(int_year)+': ')

        # Check if there are any files
        #if len(glob.glob(str_folder_path+str_patt)) > 0:
        lis_files = list(glob.glob(os.path.join(str_folder_path,'*.fits')))

        # Split into batches, because whole years can take hours to add
        for i in range(0,10):
            # Files for this batch
            lis_files_batch = lis_files[i::10]
            if len(lis_files_batch) > 0:
                #print(lis_files_batch)
                for str_filepath in lis_files_batch:
                    #print(str_filepath)
                    new_entries = utils.entries_from_goes_file(str_filepath)
                    db_source.add_many(new_entries)
            print('#',end='')
        """
        if len(glob.glob(str_folder_path+str_patt)) > 0:

            # Split into batches, because whole years can take hours to add
            for lis_sat in lis_sat_date:
                str_patt = lis_sat[0]
                str_format = lis_sat[1]

                # Add the files
                db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern=str_patt, time_string_parse_format=str_format)

                # Commit the changes to the DB
                db_source.commit()
                print(str_patt[3:5]+',', end='')
        """
        if boo_verbose: print(' Added to source DB.')
        fil_log.write(' Added to source DB.\n')
    db_source.commit()
    if boo_verbose: print('Finished adding source files to source DB in ' + str(datetime.today() - dt_start_make_source_db) + '\n')
    fil_log.write('Finished adding source files to source DB in ' + str(datetime.today() - dt_start_make_source_db)+'\n\n')

if boo_concat_data_for_years:
    if boo_verbose: print('Concatenating by year:')
    fil_log.write('Concatenating by year:\n')
    dt_start_concat = datetime.today()

    # Work on each year (within given range)
    for int_year in lis_int_years:
        # Debug text
        dt_start_year = datetime.today()
        str_year = str(int_year)
        if boo_verbose: print('  ' + str_year + ': ', end='')
        fil_log.write('  ' + str_year + ': ')

        # Check if there is a folder for the year
        str_year_folderpath = os.path.join(str_path_raw_years,str_year)
        if not os.path.exists(str_year_folderpath):
            os.makedirs(str_year_folderpath)

        for int_sat in range(1,16):
            str_sat = 'GOES '+str(int_sat)
            str_year_sat_path = os.path.join(str_year_folderpath,str_year+'__instrument-'+str_sat+'.fits')

            # Only do anything if the year-long FITS file isn't there or we want to overwrite
            if (not os.path.isfile(str_year_sat_path)) or boo_overwite_years:
                # Get the results for this year by this satelite
                results = db_source.search(vso.attrs.Time(str(int_year-1)+'-12-31 23:59:59.99', str(int_year+1)+'-01-01 00:00:00.00'), vso.attrs.Instrument(str_sat))

                # Only do anything if we have some matching data
                if len(results) > 0:
                    dt_start_task = datetime.today()
                    # Get the paths of all the files
                    lis_df_data = []
                    for file_entry in results:
                        str_path = file_entry.path
                        #ts_goes = ts.TimeSeries(str_path)
                        #lis_df_data.append(ts_goes.data)
                        #lc_goes = False
                        ts_goes = False

                        try:
                            #lc_goes = lc.GOESLightCurve.create(str_path)
                            ts_goes = ts.TimeSeries(str_path)
                        except:
                            print('Error reading: '+str_path)
                            fil_log.write('Error reading: '+str_path+'\n')

                        # Only add to the year if there is valid data within the file
                        if ts_goes:#lc_goes:
                            #if not (np.all(np.isnan(lc_goes.data['xrsa'].values)) and np.all(np.isnan(lc_goes.data['xrsb'].values))):
                            #    lis_df_data.append(lc_goes.data)
                            if not (np.all(np.isnan(ts_goes.data['xrsa'].values)) and np.all(np.isnan(ts_goes.data['xrsb'].values))):
                                lis_df_data.append(ts_goes.data)

                    # The full years data for the given satelite
                    if len(lis_df_data) > 0:
                        df_data = pd.concat(lis_df_data)

                        # Get the start and end dates
                        tim_start = Time(df_data.index[0])
                        tim_start.format = 'isot' # Not currently correct using 'fits'
                        tim_end = Time(df_data.index[-1])
                        tim_end.format = 'isot'

                        flo_task_time = (datetime.today() - dt_start_task).total_seconds()

                        #tim_other = Time(df_data.index[500])
                        #tim_other.format = 'isot' # Not currently correct using 'fits'

                        # Metadata
                        dic_metadata = {
                                        'telescop': str_sat,
                                        'instrume': str_sat, # Normally 'X-ray Detector'
                                        #'date': datetime.today().strftime('%Y/%m/%d'),#.strftime('%d/%m/%Y'),
                                        'DATE-BEG': str(tim_start),
                                        'DATE-OBS': str(tim_start),
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
                                        'CTYPE1': 'seconds',#     / seconds into DATE-OBS of 3s interval (see comments)
                                        'CTYPE2': 'watts / m^2',#  / in 1. - 8. Angstrom band
                                        'CTYPE3': 'watts / m^2',#  / in .5 - 4. Angstrom band
                                        'secs-concat-year': str(flo_task_time),
                                        }

                        """
                        # Make an astropy table to save the data
                        tbl_data = Table([np.array((df_data.index - df_data.index[0]).total_seconds()) *u.s, df_data['xrsa'].values * (u.Watt / u.m**2), df_data['xrsb'].values * (u.Watt / u.m**2)], names=('time', 'xrsa', 'xrsb'), meta=dic_metadata)
                        #tbl_data.write('D:\work_data\\goes_xrs_pre_processed\\table.fits', format='fits')
                        tbl_data.write(str_year_sat_path, format='fits', overwrite=True)
                        """

                        ts_full = ts.TimeSeries(df_data, dic_metadata, {'xrsa':(u.Watt / u.m**2), 'xrsb':(u.Watt / u.m**2)})
                        ts_full.save(str_year_sat_path, overwrite=True)
                        ts_full.save(str_year_sat_path.replace('.fits','.tsv'), filetype='tsv')
                        ts_full.save(str_year_sat_path.replace('.fits','.h5'), filetype='h5')

                        if boo_verbose: print(str(int_sat)+',', end='')
                        fil_log.write(str(int_sat)+',')

        # Import this file into the db
        if boo_import_years_into_raw_db:
            int_len = len(db_data_raw)
            #db_data_raw.add_from_dir(str_year_folderpath, ignore_already_added=True)
            lis_paths = list(glob.glob(os.path.join(str_year_folderpath,'*.fits')))
            utils.add_entries_from_goes_ts_files(db_data_raw, *lis_paths, tags=['raw']) # Note, will add a duplicate if the file is already present.
            db_data_raw.commit()
            #print('Added ' + str(len(db_data) - int_len) + ' to the db')


        if boo_verbose: print('    Finished '+str_year+' in ' + str(datetime.today() - dt_start_year))
        fil_log.write('    Finished '+str_year+' in ' + str(datetime.today() - dt_start_year)+'\n\n')
    if boo_verbose: print('Finished concatenating all years in ' + str(datetime.today() - dt_start_concat) + '\n')
    fil_log.write('Finished concatenating all years in ' + str(datetime.today() - dt_start_concat)+'\n\n')

# Clean the data database
if boo_neaten_data_db:
    dt_start_neaten = datetime.today()
    # Remove non-primary headers
    for database_entry in db_data:
       if database_entry.observation_time_start is None and database_entry.observation_time_end is None:
          db_data.remove(database_entry)
    db_data.commit()
    if boo_verbose: print('Finished neatening data DB in ' + str(datetime.today() - dt_start_neaten))
    fil_log.write('Finished neatening data DB in ' + str(datetime.today() - dt_start_neaten)+'\n')


if boo_pre_proc_years:
    if boo_verbose: print('Pre-processing years:')
    dt_start_pre_proc = datetime.today()

    # Search DB for matching results
    db_results = db_data_raw.search(vso.attrs.Time(str(lis_int_years[0]-1)+'-12-30 23:59:59', str(lis_int_years[0]+1)+'-01-02 00:00:00'))

    for result in db_results:
        # Path details
        str_filepath = result.path
        str_folderpath = os.path.split(str_filepath)[0]
        str_filename = os.path.split(str_filepath)[-1]
        str_year = os.path.split(str_folderpath)[-1]

        # Check if there is a folder for the year
        str_year_folderpath = os.path.join(str_path_pre_proc,str_year)
        print('\n\n\n'+str_year_folderpath+'\n\n\n')
        if not os.path.exists(str_year_folderpath):
            os.makedirs(str_year_folderpath)

        # Load as SunPy TimeSeries
        ts_result = ts.TimeSeries(str_filepath)

        # Make a mask for all the NaN and zero values in the original dataset
        #ser_xrsa_raw_mask = pd.Series(data=np.logical_or(np.isnan(ts_result.data['xrsa'].values), ts_result.data['xrsa'].values == 0.0), ts_result.data['xrsa']=data.index)
        #ser_xrsb_raw_mask = pd.Series(data=np.logical_or(np.isnan(ts_result.data['xrsb'].values), ts_result.data['xrsb'].values == 0.0), ts_result.data['xrsb']=data.index)
        df_raw_mask = copy.deepcopy(ts_result.data)
        df_raw_mask.columns = ['xrsa mask', 'xrsb mask']
        df_raw_mask['xrsa mask'] = np.logical_or(np.isnan(ts_result.data['xrsa'].values), ts_result.data['xrsa'].values == 0.0)
        df_raw_mask['xrsb mask'] = np.logical_or(np.isnan(ts_result.data['xrsb'].values), ts_result.data['xrsb'].values == 0.0)


        # Interpolate for all NaN and zero values in the original dataset (will be removed later)
        #ser_xrsa_raw_int = ts_result.data['xrsa'].replace({0.0:np.nan}).interpolate()
        #ser_xrsb_raw_int = ts_result.data['xrsb'].replace({0.0:np.nan}).interpolate()
        ts_result.data = ts_result.data.replace({0.0:np.nan}).interpolate()

        # Now pre-process
        for str_bin in tup_str_bins:
            str_rebinned_filename = str_filename.split('.')[0] + '__ppbinwid_' + str_bin + '__ppbinmet_median' + '.fits'
            ts_rebinned = copy.deepcopy(ts_result)

            # Rebin
            #ts_rebinned.data = ts_rebinned.data.resample(str_bin).median()
            # Resample/Rebin
            #ser_xrsa_raw_int_res = ser_xrsa_raw_int.resample(str_bins).median()
            #ser_xrsa_raw_int_res_mask = ser_xrsa_raw_mask.resample(str_bins).max()
            #ser_xrsb_raw_int_res = ser_xrsb_raw_int.resample(str_bins).median()
            #ser_xrsb_raw_int_res_mask = ser_xrsb_raw_mask.resample(str_bins).max()
            dt_start_task = datetime.today()
            df_rebinned = ts_rebinned.data.resample(str_bin).median()
            df_rebinned_mask = df_raw_mask.resample(str_bin).max()
            ts_rebinned.data = pd.concat([df_rebinned, df_rebinned_mask], axis=1, join='inner')
            flo_task_time = (datetime.today() - dt_start_task).total_seconds()

            # Add Rebinned Pre-Process Parameter and save
            ts_rebinned.meta.update({'ppbinwid':str_bin, 'ppbinmet':'median', 'secs-rebin': str(flo_task_time), 'rawpath':str_filepath}, overwrite=True)
            str_rebinned_filepath = os.path.join(str_year_folderpath,str_rebinned_filename )
            ts_rebinned.save(str_rebinned_filepath)
            ts_rebinned.save(str_rebinned_filepath.replace('.fits','.tsv'), filetype='tsv')
            ts_rebinned.save(str_rebinned_filepath.replace('.fits','.h5'), filetype='h5')

            # Add to the database
            #new_entries = utils.entries_from_goes_file(str_rebinned_filepath)
            #db_data.add_many(new_entries)
            utils.add_entries_from_goes_ts_files(db_data, str_rebinned_filepath, tags=['ppbinwid='+str_bin, 'ppbinmet=median'], source=str_filepath)
            db_data.commit()

            # Now Boxcart Average
            for int_boxcart_width in tup_int_boxcart_widths:
                dt_start_task = datetime.today()
                str_averaged_filename = str_rebinned_filename.split('.')[0] + '__ppbcwid_' + str(int_boxcart_width) + '__ppbcmet_mean' + '__ppbccen_True' + '__ppbcmin_1' + '.fits'
                ts_averaged = copy.deepcopy(ts_rebinned)

                # Average
                #ts_averaged.data = ts_averaged.data.rolling(int_boxcart_width, center=True, min_periods=1).mean()
                dt_start_task = datetime.today()
                df_averaged = df_rebinned.rolling(int_boxcart_width, center=True, min_periods=1).mean()
                ts_averaged.data = pd.concat([df_averaged, df_rebinned_mask], axis=1, join='inner')
                flo_task_time = (datetime.today() - dt_start_task).total_seconds()

                # Add Rebinned Pre-Process Parameter and save
                ts_averaged.meta.update({'ppbcwid':int_boxcart_width, 'ppbcmet': 'mean', 'ppbccen': 'True', 'ppbcmin': '1', 'secs-boxcart-average': str(flo_task_time),}, overwrite=True)
                str_averaged_filepath = os.path.join(str_year_folderpath,str_averaged_filename )
                ts_averaged.save(str_averaged_filepath)
                ts_averaged.save(str_averaged_filepath.replace('.fits','.tsv'), filetype='tsv')
                ts_averaged.save(str_averaged_filepath.replace('.fits','.h5'), filetype='h5')

                # Add to DB
                utils.add_entries_from_goes_ts_files(db_data, str_averaged_filepath, tags=['ppbinwid='+str_bin, 'ppbinmet=median', 'ppbcwid='+str(int_boxcart_width), 'ppbcmet=mean', 'ppbccen=True', 'ppbcmin=1'], source=str_filepath)
                db_data.commit()

    if boo_verbose: print('Finished preprocessing all years in ' + str(datetime.today() - dt_start_pre_proc) + '\n')
    fil_log.write('Finished preprocessing all years in ' + str(datetime.today() - dt_start_pre_proc)+'\n\n')


if boo_detect_flares:
    if boo_verbose: print('Detect flares:')
    dt_start_det_flares = datetime.today()

    # Search DB for matching results
    db_results = db_data.search(vso.attrs.Time(str(lis_int_years[0]-1)+'-12-30 23:59:59', str(lis_int_years[0]+1)+'-01-02 00:00:00'))

    for result in db_results:
        # Path details
        str_filepath = result.path
        str_folderpath = os.path.split(str_filepath)[0]
        str_filename = os.path.split(str_filepath)[-1]
        str_year = os.path.split(str_folderpath)[-1]
        str_year_folderpath = os.path.join(str_path_pre_proc,str_year)

        # Load as SunPy TimeSeries
        ts_result = ts.TimeSeries(str_filepath)
        if ts_result.meta.get('ppbinwid').values() != []:
            str_ppbinwid = ts_result.meta.get('ppbinwid').values()[0]
        else:
            str_ppbinwid = ''
        if ts_result.meta.get('ppbinmet').values() != []:
            str_ppbinmet = ts_result.meta.get('ppbinmet').values()[0]
        else:
            str_ppbinmet = ''
        if ts_result.meta.get('ppbcwid').values() != []:
            str_ppbcwid = ts_result.meta.get('ppbcwid').values()[0]
        else:
            str_ppbcwid = ''
        if ts_result.meta.get('ppbcmet').values() != []:
            str_ppbcmet = ts_result.meta.get('ppbcmet').values()[0]
        else:
            str_ppbcmet = ''
        if ts_result.meta.get('ppbccen').values() != []:
            str_ppbccen = ts_result.meta.get('ppbccen').values()[0]
        else:
            str_ppbccen = ''
        if ts_result.meta.get('ppbcmin').values() != []:
            str_ppbcmin = ts_result.meta.get('ppbcmin').values()[0]
        else:
            str_ppbcmin = ''

        # Get the raw data TS
        str_raw_data_path = result.source
        ts_raw = ts.TimeSeries(str_raw_data_path)

        str_detections_path = os.path.join(str_path_flare_outputs,str_year,str_year+'_flare_detections' + '__instrument-' + result.instrument + '__ppbinwid_' + str_ppbinwid + '__ppbinmet_' + str_ppbinmet + '__ppbcwid_' + str_ppbcwid + '__ppbcmet_' + str_ppbcmet + '__ppbccen_' + str_ppbccen + '__ppbcmin_' + str_ppbcmin)
        #print('str_detections_path: '+str_detections_path)

        # Check if there is a folder for the years detections
        str_year_folderpath = os.path.join(str_path_flare_outputs,str_year)
        if not os.path.exists(str_year_folderpath):
            os.makedirs(str_year_folderpath)

        # Get flare detections
        # Get the CWT peaks
        #int_max_width = 50 # Could use int_max_width = 50 to increase sensitivity arr_cwt_widths = np.arange(1,int_max_width)
        lis_arr_widths = [np.arange(1,26), np.arange(1,26,2), np.arange(1,26,4), np.arange(1,51), np.arange(1,51,2), np.arange(1,51,4), np.arange(1,101), np.arange(1,101,2), np.arange(1,101,4)]
        for arr_cwt_widths in lis_arr_widths:
            str_detections_path_full = str_detections_path+'__fdmet=cwt__fdwid_['+str(arr_cwt_widths[0])+','+str(arr_cwt_widths[1])+'...'+str(arr_cwt_widths[-1])+'].fits'

            # Make the flare event list
            dt_start_task = datetime.today()
            df_peaks_cwt = det.get_flare_peaks_cwt(ts_result.data['xrsb'].interpolate(), raw_data=ts_raw.data['xrsb'].interpolate(), widths=arr_cwt_widths, get_energies=True)
            df_peaks_cwt['fl_duration'] = df_peaks_cwt['fl_duration_(td)'].values.astype('float64')/1e9
            flo_task_time = (datetime.today() - dt_start_task).total_seconds()

            # Make this a TS with metadata
            md_flares = copy.copy(ts_result.meta.metadata[0][2])
            md_flares['secs-flare-det'] = str(flo_task_time)
            md_flares['fdmet'] = 'cwt'
            md_flares['fdwids'] = str(arr_cwt_widths)

            ts_flares = ts.TimeSeries(df_peaks_cwt, md_flares)
            #ts_flares.save(str_detections_path_full+'.fits')
            ts_flares.save(str_detections_path_full.replace('.fits','.tsv'), filetype='tsv')
            #ts_flares.save(str_detections_path_full.replace('.fits','.h5'), filetype='h5')
            #df_peaks_cwt.to_csv(str_detections_path_full.replace('.fits','.tsv'), header=True, sep='\t')
            fil_meta = open(str_detections_path_full.replace('.fits','__meta.tsv'),'w')
            fil_meta.write(str(ts_flares.meta.metadata[0][2]))
            fil_meta.close()


    if boo_verbose: print('Finished detecting flares in ' + str(datetime.today() - dt_start_det_flares) + '\n')
    fil_log.write('Finished detecting flares in ' + str(datetime.today() - dt_start_det_flares)+'\n\n')










fil_log.close()
#print(Database.display_entries(db_data, ['id', 'observation_time_start', 'observation_time_end', 'instrument', 'tags'], sort=True))
#print(Database.display_entries(db_data, ['id', 'observation_time_start', 'observation_time_end', 'instrument', 'source'], sort=True))