# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 07:16:23 2017

@author: alex_
"""

from sunpy.lightcurve import GOESLightCurve
from sunpy.time import parse_time
from sunpy.time import TimeRange
#import sunpy.timeseries as ts
import pandas as pd
import datetime

import tarfile
from ftplib import FTP
import os.path

def get_goes_xrs_year_data(years, path=""):
    """
    Downloads all the fits files for the given year.

    Note: this doesn't deal well with different satellites, it simply uses the
    SunPy Lightcurve to download a file for each day.

    Parameters
    ----------
    year : `int` or `list` or `TimeRange`
        The year (int) or years (list) or ramge of years (TimeRange) to download.

    Returns
    -------
    result : `list`
        The combined dataframe for all days.
    """
    # Sanatize inputs
    lis_years = []
    if isinstance(years, list):
        # Simply use the given list
        lis_years = years
    elif isinstance(years, int):
        # Create a list with this one year
        lis_years = [ years ]
    elif isinstance(years, TimeRange):
        # Create a list of years from the TimeRange
        lis_years = list(range(years.start.year, years.end.year+1))

    # A list for all folders
    lis_str_folders = []

    # Connect to the FTP
    str_ftp = 'umbra.nascom.nasa.gov'
    str_ftp_fol = '/goes/fits/'
    ftp = FTP(str_ftp)
    ftp.login()
    ftp.cwd(str_ftp_fol)

    for int_year in lis_years:
        # Download the year
        str_filename = str(int_year) + '.tgz'
        str_filepath = path + '//' + str_filename

        # If we dont have this file then download
        if not os.path.isfile(str_filepath):
            # If the file is avalible on the FTP
            if str_filename in ftp.nlst():
                # Download the year
                localfile = open(str_filepath, 'wb')
                print('Downloading: ' + str_filename)
                ftp.retrbinary('RETR ' + str_filename, localfile.write, 1024)
                localfile.close()

                # Untar the file
                tar = tarfile.open(str_filepath)
                print('Extracting: ' + str_filename)
                tar.extractall(path=path)
                tar.close()

                # Add folder to list
                lis_str_folders.append(path + '//' + str(int_year))
            else:
                print('Error: file ' + str_filename + ' not found on the FTP.')
        else:
            print('Note: file ' + str_filename + ' has already been downloaded.')

    # Now close the FTP
    ftp.quit()

    return lis_str_folders





def get_goes_xrs_data_as_df(start, end):
    """
    Downloads a fits file for each day in the given range and concatenate into
    a single Pandas DataFrame.

    Note: this doesn't deal well with different satelites, it simply uses the
    SunPy Lightcurve to download a file for each day.

    Parameters
    ----------
    start : `datetime.datetime`
        The start date.

    end : `datetime.datetime`
        The end date.

    Returns
    -------
    result : `pandas.DataFrame`
        The combined dataframe for all days.
    """
    dt_end = parse_time(end)
    dt_start = parse_time(start)

    # Start at the beginning of the day
    dt_day = datetime.datetime(dt_start.year, dt_start.month, dt_start.day, tzinfo=dt_end.tzinfo)

    # Add each day to the list until we get to the end day
    lis_dt_days = []
    while dt_day < dt_end:
        lis_dt_days.append(dt_day)
        dt_day = dt_day + datetime.timedelta(days=1)
    lis_dt_days.append(dt_end)

    print('lis_dt_days: '+str(lis_dt_days))

    # Download GOES XRS data for each day
    lis_lc_goes = []
    lis_df_goes = []
    for i in range(1,len(lis_dt_days)):
        try:
            lc_goes = GOESLightCurve.create(TimeRange(lis_dt_days[i-1], lis_dt_days[i]))
            #print('lc_goes: '+str(lc_goes))
            lis_lc_goes.append(lc_goes)
            lis_df_goes.append(lc_goes.data)
            print('Downloaded for: '+str(lis_dt_days[i-1]))
        except:
            print('  Unable to downloaded for: '+str(lis_dt_days[i-1]))

    """
    df_all = pd.DataFrame()
    for df_goes in lis_df_goes:
        df_all = pd.concat([df_all, ])
    """
    #print(lis_df_goes)
    df_all = pd.concat(lis_df_goes)

    return df_all

