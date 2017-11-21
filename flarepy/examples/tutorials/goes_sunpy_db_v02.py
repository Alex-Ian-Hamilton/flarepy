# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:55:59 2017

@author: alex_
"""

import flarepy.utils as utils
from sunpy.database import Database
import sunpy.database.attrs as dbattrs
import os
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

import warnings
warnings.filterwarnings("ignore")

# Parameters
str_path_sour = os.path.join('D:\\','work_data','goes_xrs_source')
str_path_pre_proc = os.path.join('D:\\','work_data','goes_xrs_pre_processed')
str_path_db_sour = os.path.join('D:\\','work_data','db','db_flarepy_source.sqlite')#db_flarepy_test.sqlite
str_path_db_data = os.path.join('D:\\','work_data','db','db_flarepy_data.sqlite')
boo_download_data = False
boo_add_sour_to_db = False
boo_concat_data_for_years = True
boo_import_years_into_pre_proc_db = False
boo_pre_proc_years = False
boo_verbose = True
tup_bins = ('12s', '60s')

# Delimiters (for storing parameters in filenames)
str_par = '-'
str_del = '_'

# For timing performance
dt_start = datetime.today()

#db = Database('sqlite:///C:\\goes_data\\db_all_goes.sqlite')
# Make/open the source data database
db_source = Database('sqlite:///'+str_path_db_sour)
db_data = Database('sqlite:///'+str_path_db_data)
#db_detections = Database('sqlite:///C:\\Users\\alex_\\sunpy\\db_flarepy_detections.sqlite')

# Download files?
lis_int_years = list(range(1999,2000))#list(range(1981,2000))
if boo_download_data:
    utils.goes_utils.get_goes_xrs_year_data(lis_int_years, path=str_path_sour)

# Get the folders
lis_str_folders = os.walk(str_path_sour)

# Adding the source data files into the database
if boo_add_sour_to_db:
    dt_start_make_source_db = datetime.today()

    # Add for each year
    for int_year in lis_int_years:
        # Get the folder path
        str_folder_path = os.path.join(str_path_sour,str(int_year))
        print('  ' + str(int_year)+': ',end='')

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
                    new_entry = utils.entries_from_goes_file(str_filepath)
                    db_source.add_many(new_entry)
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
    db_source.commit()
    if boo_verbose: print('Finished adding source files to source DB in ' + str(datetime.today() - dt_start_make_source_db))

if boo_concat_data_for_years:
    if boo_verbose: print('Concatenating by year:')
    dt_start_concat = datetime.today()

    #
    for int_year in lis_int_years:
        # Debug text
        dt_start_year = datetime.today()
        str_year = str(int_year)
        if boo_verbose: print('  ' + str_year + ': ', end='')

        # Check if there is a folder for the year
        str_year_folderpath = os.path.join(str_path_pre_proc,str_year)
        if not os.path.exists(str_year_folderpath):
            os.makedirs(str_year_folderpath)

        for int_sat in range(1,16):
            str_sat = 'GOES '+str(int_sat)

            # Get the results for this year by this satelite
            results = db_source.search(vso.attrs.Time(str(int_year-1)+'-12-31 23:59:59.99', str(int_year+1)+'-01-01 00:00:00.00'), vso.attrs.Instrument(str_sat))

            # Only do anything if we have some matching data
            if len(results) > 0:
                # Get the paths of all the files
                lis_df_data = []
                for file_entry in results:
                    str_path = file_entry.path
                    #ts_goes = ts.TimeSeries(str_path)
                    #lis_df_data.append(ts_goes.data)
                    lc_goes = lc.GOESLightCurve.create(str_path)
                    lis_df_data.append(lc_goes.data)

                # The full years data for the given satelite
                df_data = pd.concat(lis_df_data)

                # Get the start and end dates
                tim_start = Time(df_data.index[0])
                tim_start.format = 'isot' # Not currently correct using 'fits'
                tim_end = Time(df_data.index[-1])
                tim_end.format = 'isot'

                tim_other = Time(df_data.index[500])
                tim_other.format = 'isot' # Not currently correct using 'fits'

                # Metadata
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
                                'CTYPE1': 'seconds ',#     / seconds into DATE-OBS of 3s interval (see comments)
                                'CTYPE2': 'watts / m^2',#  / in 1. - 8. Angstrom band
                                'CTYPE3': 'watts / m^2',#  / in .5 - 4. Angstrom band
                                }

                # Make an astropy table to save the data
                tbl_data = Table([(df_data.index - df_data.index[0]).total_seconds(), df_data['xrsa'].values, df_data['xrsb'].values], names=('seconds', 'xrsa', 'xrsb'), meta=dic_metadata)
                #tbl_data.write('D:\work_data\\goes_xrs_pre_processed\\table.fits', format='fits')
                tbl_data.write(os.path.join(str_year_folderpath,str_year+'_'+str_sat.replace(' ','-')+'.fits'), format='fits')
                if boo_verbose: print(str(int_sat)+',', end='')


                """
                ###############################################################
                # Temporarily do pre-processing in here
                ###############################################################
                # For every required bin size
                for str_bin in tup_bins:

                    # A dictionary to hold all the pre-process parameters
                    dic_pre_proc = OrderedDict(tbl_year.meta)

                    # Convert into a Pandas DataFrame to pre-process
                    df_year = pd.DataFrame({'xrsa':tbl_year['xrsa'], 'xrsb':tbl_year['xrsb']}, index=pd.to_datetime(tbl_year['index']))
                    #df_year = df_year[0:1000]

                    # Make a mask for all the NaN and zero values in the original dataset
                    #pd_raw_mask = pd.Series(data=np.logical_or(np.isnan(df_year.values), df_year.values == 0.0), index=df_year.index)# For pandas Series
                    df_raw_mask = pd.DataFrame(data={'xrsa[mask]':np.logical_or(np.isnan(df_year['xrsa']), df_year['xrsa'] == 0.0),
                                                     'xrsb[mask]':np.logical_or(np.isnan(df_year['xrsb']), df_year['xrsb'] == 0.0)},
                                                    index=df_year.index)# For pandas dataframe
                    #df_year['xrsa[mask]'] = np.logical_or(np.isnan(df_year['xrsa']), df_year['xrsa'] == 0.0)
                    #df_year['xrsb[mask]'] = np.logical_or(np.isnan(df_year['xrsb']), df_year['xrsb'] == 0.0)

                    # Interpolate for all NaN and zero values in the original dataset (will be removed later)
                    df_raw_int = df_year.replace({0.0:np.nan}).interpolate()

                    # Resample/Rebin data an mask using the given method
                    str_bins = '12S'
                    df_raw_int_res = df_raw_int.resample(str_bins).median()
                    df_raw_int_res_mask = df_raw_mask.resample(str_bins).max()

                    # Add resample/rebin parameters to the metadata
                    dic_pre_proc['RSWIDTH'] = str_bins
                    dic_pre_proc['RSMETHOD'] = 'median'
                    dic_pre_proc['RSLOFFST'] = 'str_bins'

                    # Save the resample/rebin data
                    dic_year_resampled = {'index': np.array(df_raw_int_res.index).tolist(),
                                          'xrsa':df_raw_int_res['xrsa'], 'xrsb':df_raw_int_res['xrsb'],
                                          'xrsa[mask]':df_raw_int_res_mask['xrsa[mask]'], 'xrsb[mask]':df_raw_int_res_mask['xrsb[mask]']}
                    tbl_year_resampled = Table(dic_year_resampled, meta=dic_pre_proc)
                    tbl_year_resampled.write(str_path[0:-5]+
                                             str_del+'RSWIDTH' +str_par+str(dic_pre_proc['RSWIDTH'])+
                                             str_del+'RSMETHOD'+str_par+str(dic_pre_proc['RSMETHOD'])+
                                             str_del+'RSLOFFST'+str_par+str(dic_pre_proc['RSLOFFST'])+
                                             '.fits', overwrite=True)

                    # Find the rolling average
                    int_cart = 5
                    df_raw_int_res_box = df_raw_int_res.rolling(int_cart, center=True, min_periods=1).mean()

                    # Add boxcart averaging parameters to the metadata
                    dic_pre_proc['BCWIDTH'] = int_cart
                    dic_pre_proc['BCMETHOD'] = 'mean'
                    dic_pre_proc['BCCENTER'] = True
                    dic_pre_proc['BCMINPER'] = 1

                    # Save the averaged data
                    dic_year_averaged = {'index': np.array(df_raw_int_res.index).tolist(),
                                          'xrsa':df_raw_int_res_box['xrsa'], 'xrsb':df_raw_int_res_box['xrsb'],
                                          'xrsa[mask]':df_raw_int_res_mask['xrsa[mask]'], 'xrsb[mask]':df_raw_int_res_mask['xrsb[mask]']}
                    tbl_year_averaged = Table(dic_year_averaged, meta=dic_pre_proc)
                    tbl_year_averaged.write(str_path[0:-5]+
                                             str_del+'RSWIDTH' +str_par+str(dic_pre_proc['RSWIDTH'])+
                                             str_del+'RSMETHOD'+str_par+str(dic_pre_proc['RSMETHOD'])+
                                             str_del+'RSLOFFST'+str_par+str(dic_pre_proc['RSLOFFST'])+
                                             str_del+'BCWIDTH' +str_par+str(dic_pre_proc['BCWIDTH'])+
                                             str_del+'BCMETHOD'+str_par+str(dic_pre_proc['BCMETHOD'])+
                                             str_del+'BCCENTER'+str_par+str(dic_pre_proc['BCCENTER'])+
                                             str_del+'BCMINPER'+str_par+str(dic_pre_proc['BCMINPER'])+
                                             '.fits', overwrite=True)
                """

###t = Table.read('D:\\work_data\\goes_xrs_pre_processed\\1999\\1999_GOES 10.fits')
###db_data.add_from_dir('D:\\work_data\\goes_xrs_pre_processed\\1999\\', time_string_parse_format='%d/%m/%Y')



        if boo_verbose: print('    Finished '+str_year+' in ' + str(datetime.today() - dt_start_year))
    if boo_verbose: print('Finished concatenating all years in ' + str(datetime.today() - dt_start_concat))
db_data.add_from_dir('D:\\work_data\\goes_xrs_pre_processed\\1999\\')