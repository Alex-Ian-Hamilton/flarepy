# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:50:32 2017

@author: Alex
"""

from astropy import units as u

# Select years for solar maxma
lis_int_years = [ 2011, 2012, 2013 ]

# Folders for GOES data and saving results
str_h5_goes_data_path = 'C:\\goes_h5\\'
str_fits_data_path = 'C:\\goes_fits\\'
str_h5_flare_data_path = 'C:\\flare_h5\\'

# Decide what parameter/s to use
lis_qua_bins = [ 12 * u.s ]
lis_int_rolling_mean = [5]
lis_qua_par_N = [ 4.0 * u.min ]
lis_flo_par_thr = [ 0.4 ]
lis_flo_par_drop = [ 0.5 ]

############
#
# Generate the combined/smooth GOES data files
#
############
# If there arn't already gibgle year files for the GOES data then make them from the day FITS files.
for int_year in lis_int_years:
    str_year = str(int_year)
    for qua_bin in lis_qua_bins:#for str_bin_size in lis_str_bins:
        str_bin_size = quantity_to_pandas_string(qua_bin)
        for int_rolling_mean in lis_int_rolling_mean:
            try:
                # Check for data files

                #str_bin_size = '12s'
                int_rolling_mean = 5

                # Check if the rebinned data file exists
                str_h5_year_smoothed_data_path = str_h5_goes_data_path + str_year+'_goes__rebinned_'+str_bin_size+'_median__smoothed_rolling_mean_'+str(int_rolling_mean)+'.h5'
                if not os.path.isfile(str_h5_year_smoothed_data_path):
                    # Check if the rebinned data file exists
                    str_h5_year_rebinned_data_path = str_h5_goes_data_path + str_year+'_goes__rebinned_'+str_bin_size+'_median.h5'
                    if not os.path.isfile(str_h5_year_rebinned_data_path):
                        # The data hasn't been rebinned
                        str_h5_year_data_path = str_h5_goes_data_path + str_year+'_goes.h5'

                        # Check if the raw h5 file for the whole year exists
                        if not os.path.isfile(str_h5_year_data_path):
                            # Need to create the years data using the fits files
                            str_year_datapath = str_fits_data_path + str_year + '\\'
                            save_folder_as_h5(str_year_datapath, str_h5_year_data_path)

                        # Generate the rebinned files
                        rebin_h5_files(str_h5_year_data_path, lis_qua_bins=[qua_bin])
                    # Generate the smoothed file
                    smooth_h5_files(str_h5_year_rebinned_data_path, str_h5_year_smoothed_data_path, int_rolling_mean)
            except:
                print('Error generating data files.')