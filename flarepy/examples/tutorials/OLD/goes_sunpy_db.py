# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
import astropy.units as u
from sunpy.net import Fido, attrs as a
from sunpy.database import Database

db = Database()
db.fetch(a.Time("2011-09-20T01:00:00", "2011-09-20T02:00:00"),
             a.Instrument('AIA'), a.vso.Sample(15*u.min))
db.commit()
"""
import glob
from sunpy.database import Database
import sunpy.database.attrs as dbattrs
import datetime
import os
import flarepy.utils as utils
from sunpy.net import vso
import sunpy.timeseries as ts
from astropy.table import Table
import numpy as np
import pandas as pd
from collections import OrderedDict

# Parameters
str_path_sour = 'D:\\work_data\\goes_xrs_source'
str_path_pre_proc = 'D:\\work_data\\goes_xrs_pre_processed'
str_path_db_sour = 'D:\\work_data\\db\\db_flarepy_source.sqlite'
str_path_db_pre_proc = 'D:\\work_data\\db\\db_flarepy_pre-process.sqlite'
boo_download_data = False
boo_add_sour_to_db = True
boo_neaten_sour_db = False
boo_concat_data_for_years = False
boo_import_years_into_pre_proc_db = False
boo_pre_proc_years = False
boo_verbose = True

# Delimiters (for storing parameters in filenames)
str_par = '-'
str_del = '_'

# For timing performance
dt_start = datetime.datetime.today()

#db = Database('sqlite:///C:\\goes_data\\db_all_goes.sqlite')
# Make/open the source data database
db_source = Database('sqlite:///'+str_path_db_sour)
#db_detections = Database('sqlite:///C:\\Users\\alex_\\sunpy\\db_flarepy_detections.sqlite')

# Download files?
lis_int_years = list(range(1996,2017))#list(range(1981,2000))
if boo_download_data:
    utils.goes_utils.get_goes_xrs_year_data(lis_int_years, path=str_path_sour)

# Get the folders
lis_str_folders = os.walk(str_path_sour)

# Adding the source data files into the database
if boo_add_sour_to_db:
    dt_start_make_source_db = datetime.datetime.today()
    # Import each folder into the source db
    #for str_year in next(os.walk(str_path_sour))[1]: # Automatically gets all year folders

    # The format by satellite
    lis_sat_date = [ ['*go02*.fits', '%d/%m/%y'],
                     ['*go05*.fits', '%d/%m/%y'],
                     ['*go06*.fits', '%d/%m/%y'],
                     ['*go07*.fits', '%d/%m/%y'],
                     ['*go08*.fits', '%d/%m/%y'],
                     ['*go09*.fits', '%d/%m/%y'],
                     ['*go10*.fits', '%d/%m/%y'],
                     ['*go11*.fits', '%d/%m/%Y'],
                     ['*go12*.fits', '%d/%m/%Y'],
                     ['*go13*.fits', '%d/%m/%Y'],
                     ['*go14*.fits', '%d/%m/%Y'],
                     ['*go15*.fits', '%d/%m/%Y'] ]

    # Add for each year
    for int_year in lis_int_years:
        # Get the folder path
        str_folder_path = str_path_sour + '//' + str(int_year) + '//'

        # Split into batches, because whole years can take hours to add
        for lis_sat in lis_sat_date:
            str_patt = lis_sat[0]
            str_format = lis_sat[1]

            # Check if there are any files
            if len(glob.glob(str_folder_path+str_patt)) > 0:
                # Add the files
                db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern=str_patt, time_string_parse_format=str_format)

                # Commit the changes to the DB
                db_source.commit()
                print(str_patt[3:5]+',', end='')

        """
        # Add all the FITS files for that year
        # GOES 2 has a different datetime format:
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go02*.fits', time_string_parse_format='%d/%m/%y')
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go05*.fits', time_string_parse_format='%d/%m/%y')
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go06*.fits', time_string_parse_format='%d/%m/%y')
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go07*.fits', time_string_parse_format='%d/%m/%y')
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go08*.fits', time_string_parse_format='%d/%m/%y')#1996
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go09*.fits', time_string_parse_format='%d/%m/%y')#1996
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go10*.fits', time_string_parse_format='%d/%m/%y')#1998
        # GOES 2 has a different datetime format
        ###Issue adding 1999.
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go12*.fits', time_string_parse_format='%d/%m/%Y')#2002
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go11*.fits', time_string_parse_format='%d/%m/%Y')#2006
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go14*.fits', time_string_parse_format='%d/%m/%Y')#2009
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go15*.fits', time_string_parse_format='%d/%m/%Y')#2010
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, pattern='*go13*.fits', time_string_parse_format='%d/%m/%Y')#2015
        """

        if boo_verbose: print('\nAdded ' + str(int_year) + ' to source DB.')
    if boo_verbose: print('Finished changing instrument column for each sat in ' + str(datetime.datetime.today() - dt_start_make_source_db))

# Now run the pre-proccesing
# Pre-prcess by year
#for int_year in lis_int_years:
#    # Now pre-process by satellite


"""
str_file = 'D://work_data//goes_xrs_source//1981//go02810101.fits
#db_source.add_from_file(str_folder_path, ignore_already_added=True, time_string_parse_format='%d/%m/%Y')

str_folder = 'D://work_data//goes_xrs_source//1981//'
db_source.add_from_dir(str_folder , ignore_already_added=True, pattern='(?s:go02.*\\.fits)\\Z', time_string_parse_format='%d/%m/%Y')
db_source.add_from_dir(str_folder , ignore_already_added=True, time_string_parse_format='%d/%m/%y')

print('Finished in ' + str(datetime.datetime.today() - dt_start))#

print(Database.display_entries(database.query(vso.attrs.Time('2012-08-05', '2012-08-05 00:00:05'), vso.attrs.Instrument('AIA')),
                      ['id', 'observation_time_start', 'observation_time_end', 'instrument', 'wavemin', 'wavemax'],
                      sort=True)
    )

db_source.search(vso.attrs.Time('2013-08-05', '2013-08-07'))
db_source.search(vso.attrs.Time('2013-08-05', '2013-08-07'), vso.attrs.Instrument('X-ray Detector'))
print(Database.display_entries(db_source.search(vso.attrs.Time('2013-08-05', '2013-08-07'), vso.attrs.Instrument('X-ray Detector')), ['id', 'observation_time_start', 'observation_time_end', 'instrument'],sort=True))
print(Database.display_entries(db_source.search(vso.attrs.Time('2013-08-05', '2013-08-07'), vso.attrs.Instrument('X-ray Detector')), ['id', 'observation_time_start', 'observation_time_end', 'instrument', 'path'],sort=True))
Database.display_entries(db_source.search(vso.attrs.Time('2013-08-05', '2013-08-07'), vso.attrs.Instrument('X-ray Detector')), ['path'],sort=True)
Database.display_entries(db_source.search(vso.attrs.Time('2013-08-05', '2013-08-07'), vso.attrs.Instrument('X-ray Detector')), ['path'],sort=True)

print(Database.display_entries(db_source.query( dbattrs.FitsHeaderEntry('TELESCOP', 'GOES 15')), ['id', 'observation_time_start', 'observation_time_end', 'instrument', 'wavemin', 'wavemax', 'tags', 'starred'], sort=True))
print(Database.display_entries(db_source.search( dbattrs.FitsHeaderEntry('TELESCOP', 'GOES 2')), ['id', 'observation_time_start', 'observation_time_end', 'instrument', 'wavemin', 'wavemax', 'tags', 'starred'], sort=True))

print(Database.display_entries(db_source, ['id', 'observation_time_start', 'observation_time_end', 'instrument', 'wavemin', 'wavemax', 'tags', 'starred'], sort=True))

"""

if boo_neaten_sour_db:
    dt_start_neaten = datetime.datetime.today()
    # Remove non-primary headers
    for database_entry in db_source:
       if database_entry.observation_time_start is None and database_entry.observation_time_end is None:
          db_source.remove(database_entry)
    db_source.commit()

    # Now change the instrument col value to show the satellite for that file
    # Note: took about 3 mins for 5 years
    for database_entry in db_source:
        for fits_entry in database_entry.fits_header_entries:
            if fits_entry.key == 'TELESCOP':
                #fits_entry.print(fits_entry.value)
                database_entry.instrument = fits_entry.value
                #database_entry.source = fits_entry.value
                break
    db_source.commit()
    if boo_verbose: print('Finished changing instrument column for each sat in ' + str(datetime.datetime.today() - dt_start_neaten))


if boo_concat_data_for_years:
    if boo_verbose: print('Concatenating by year.')
    dt_start_concat = datetime.datetime.today()
    # For each satellite we can collect the data into yearly datasets
    for int_year in lis_int_years:
        dt_start_year = datetime.datetime.today()
        str_year_before = str(int_year - 1)
        str_year = str(int_year)
        str_year_after = str(int_year + 1)
        for int_goes in range(1,16):
            dt_start_year_sat = datetime.datetime.today()

            # Get the string correlating to the GOES Satellite
            str_inst = 'GOES ' + str(int_goes)

            # Get the entries during this year for this satellite (note: I add 1 days padding each side)
            #db_batch = db_source.search(vso.attrs.Time(str_year_before+'-12-31 23:59:59', str_year_after+'-01-01 00:00:00'), vso.attrs.Instrument(str_inst))
            # Note: An alternative is to use source vso.attrs.Source(str_inst)

            # Make a list of all folders
            lis_str_path_sours = []
            if len(db_batch) > 0:
                # Get all the file paths
                for database_entry in db_batch:
                    lis_str_path_sours.append(database_entry.path)

                # Now create the TS for that dataset
                ts_year = ts.TimeSeries(list(set(lis_str_path_sours)), source='XRS', concatenate=True)

                # Crop the padding from 24h to 1 h
                ts_year.data.truncate(str_year_before+'-12-31 23:00:00', str_year_after+'-01-01 01:00:00')

                # Create an astropy table from this
                #tbl_year = Table.from_pandas(ts_year.data) # Won't include the index.
                #dic_year = {'index': np.array(ts_year.data.index),'xrsa':ts_year.data['xrsa'], 'xrsb':ts_year.data['xrsb']} # Won't save as FITS.
                dic_year = {'index': np.array(ts_year.data.index).tolist(),'xrsa':ts_year.data['xrsa'], 'xrsb':ts_year.data['xrsb']}
                tbl_year = Table(dic_year)

                # Add the useful metadata (uses the first file)
                tbl_year.meta['date-obs'] = ts_year.index[0].strftime('%Y/%m/%d')
                tbl_year.meta['time-obs'] = ts_year.index[0].strftime('%H:%M:%S.%f')
                tbl_year.meta['date-end'] = ts_year.index[-1].strftime('%Y/%m/%d')
                tbl_year.meta['time-end'] = ts_year.index[-1].strftime('%H:%M:%S.%f')
                tbl_year.meta['TELESCOP'] = ts_year.meta.metadata[0][2].get('telescop','unknown')
                tbl_year.meta['SOURCE'] = str_inst
                tbl_year.meta['INSTRUME'] = str_inst # Goes defaults to: 'X-ray Detector'
                tbl_year.meta['EXTNAME'] = 'FLUXES'
                tbl_year.meta['OBJECT'] = 'Sun'
                tbl_year.meta['TOT-DAYS'] = str(len(db_batch) - 2)
                tbl_year.meta['ORIGIN'] = ts_year.meta.metadata[0][2].get('origin','unknown')
                tbl_year.meta['COMMENT'] = 'Concatenated GOES XRS data for '+str_inst+' satellite in '+str_year+'. No other pre-processing has been done.'
                tbl_year.meta['SR'] = False
                tbl_year.meta['RS'] = False
                tbl_year.meta['BC'] = False

                # Write the table to a fits file
                if not os.path.exists(str_path_pre_proc + '//' + str_year):
                    os.makedirs(str_path_pre_proc + '//' + str_year )
                tbl_year.write(str_path_pre_proc + '//' + str_year + '//' + str_year + str_par + 'goes' + str_del + str(int_goes) + '.fits', overwrite=True)

            #
            if boo_verbose: print('        Finished '+str_year+' '+str_inst+' in ' + str(datetime.datetime.today() - dt_start_year_sat))

            # Pre-precessing the dataset
        if boo_verbose: print('    Finished '+str_year+' in ' + str(datetime.datetime.today() - dt_start_year))
    if boo_verbose: print('Finished concatenating all years in ' + str(datetime.datetime.today() - dt_start_concat))
#if boo_verbose: print('Finished the lot in ' + str(datetime.datetime.today() - dt_start_concat))

if boo_import_years_into_pre_proc_db:
    if boo_verbose: print('Importing years into pre-processed DB.')
    # Now to create the second database
    db_pre_proc = Database('sqlite:///'+str_path_db_pre_proc)

    # Import each folder into the pre-processed db
    #for str_year in next(os.walk(str_path_pre_proc))[1]:
    for int_year in lis_int_years:
        # Get the folder path
        str_folder_path = str_path_sour + '//' + str(int_year) + '//'

        # Add all the FITS files for that year
        db_source.add_from_dir(str_folder_path, ignore_already_added=True, time_string_parse_format='%d/%m/%Y')
        db_source.commit()
        print('Added ' + str_year + ' raw to source DB.')
    if boo_verbose: print('Finished importing all years into pre-processed DB.')

if boo_pre_proc_years:
    print('here 1')
    for int_year in lis_int_years:
        print('here 2')
        # The path for the year
        str_path_year = str_path_pre_proc + '//' + str(int_year) + '//'

        # For each satellite
        for int_goes in range(1,16):
            print('here 3')
            # Filepath
            str_filepath_year = str_path_year + str(int_year) + str_del + 'GOES' + str_par + str(int_goes) + '.fits'

            # Only if the file exists
            if os.path.isfile(str_filepath_year):
                print(str_filepath_year)
                # Load FITS file into astropy table
                tbl_year = Table.read(str_filepath_year)

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




#df_year_sour = pd.DataFrame({'xrsa':t['xrsa'], 'xrsb':t['xrsb']}, index=pd.to_datetime(t['index']))


"""
dt_start = datetime.datetime.today()
ts.TimeSeries(list(set(lis_str_path_sours)), source='XRS', concatenate=True)
print('Finished in ' + str(datetime.datetime.today() - dt_start))


tbl_year = Table.from_pandas(ts_year.data)
tbl_year.write(str_path_pre_proc + '//new_table.fits', overwrite=True)
tbl_year.meta['TELESCOP'] = 'GOES 15'
t = Table.read('D://work_data//goes_xrs_pre_processed//2014//2014_goes-15.fits')
t2 = Table.read('D://work_data//goes_xrs_source//2015//go1520151218.fits')
for database_entry in db_source[]:


print(Database.display_entries(db_pre_proc, ['id', 'observation_time_start', 'observation_time_end', 'instrument', 'source'], sort=True))

from sunpy.lightcurve import GOESLightCurve
lc_goes = GOESLightCurve.create('D://work_data//goes_xrs_source//2015//go1520151218.fits')


"""


"""

Some Pre-Processing metadata tags:

Spike Removal: SR: Y
Resample Width:  RSWIDTH: 12S
Resample Method: RSMETHOD: median
Resample Method: RSLOFFST: None
Box-Cart Average Width: BCWidth: 5
Box-Cart Average Width: BCCENTER: True
Box-Cart Average Width: BCMETHOD: mean
Box-Cart Average Width: BCMINPER: 1

tbl_year.meta['RSWIDTH'] =



df = pd.DataFrame({'xrsa':t['xrsa'], 'xrsb':t['xrsb']}, index=pd.to_datetime(t['index']))

center=True, min_periods=1
"""





