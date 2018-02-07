# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:08:20 2017

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
import numpy as np
import copy
import flarepy.flare_detection as det

import warnings
warnings.filterwarnings("ignore")

# Parameters
str_path_sour = os.path.join('D:\\','work_data','goes_xrs_source')
str_path_pre_proc = os.path.join('D:\\','work_data','goes_xrs_pre_processed')
str_path_db_sour = os.path.join('D:\\','work_data','db','db_flarepy_source.sqlite')#db_flarepy_test.sqlite
str_path_db_data_raw = os.path.join('D:\\','work_data','db','db_flarepy_data_raw.sqlite')
str_path_db_data = os.path.join('D:\\','work_data','db','db_flarepy_data.sqlite')
str_path_flare_outputs = os.path.join('D:\\','work_data','flare_outputs')
str_path_log = os.path.join('D:\\','work_data','db','log_goes_sunpy_db_v03.log')
boo_download_data = False
boo_add_sour_to_db = True
boo_concat_data_for_years = True
boo_overwite_years = False
boo_import_years_into_pre_proc_db = True
boo_neaten_data_db = True
boo_pre_proc_years = True
boo_verbose = True
tup_str_bins = ('12s', '60s')
tup_int_boxcart_widths = (5, 11)

# Delimiters (for storing parameters in filenames)
str_par = '-'
str_del = '_'

# For timing performance
dt_start = datetime.today()

#db = Database('sqlite:///C:\\goes_data\\db_all_goes.sqlite')
# Make/open the source data database
#db_source = Database('sqlite:///'+str_path_db_sour)
db_data_raw = Database('sqlite:///'+str_path_db_data)
#db_data = Database('sqlite:///'+str_path_db_data)
print('len(db_data_raw): '+str(len(db_data_raw)))
filepath = 'D:\\work_data\\goes_xrs_pre_processed\\2016\\2016_GOES-13__ppbinwidth_12s__ppboxcwid_5.fits'
utils.add_entries_from_goes_ts_files(db_data_raw, filepath, tags=['raw'])
"""
for entry in utils.entries_from_goes_ts_files(filepath):
    db_data_raw.add(entry)
    db_data_raw.tag(entry, "raw")
"""
print('len(db_data_raw): '+str(len(db_data_raw)))

print(Database.display_entries(db_data_raw, ['id', 'observation_time_start', 'observation_time_end', 'instrument', 'tags'], sort=True))
