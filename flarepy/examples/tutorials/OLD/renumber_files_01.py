# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:21:29 2018

@author: alex_
"""

from PIL import Image, TiffImagePlugin
import glob
import os

count = 0

str_path = 'C:\\Users\\alex_\\Documents\\Scanned Documents\\pages'
str_file_extension = 'tiff'
int_start_number = 2
int_end_number = 69
int_correction = -1
int_zero_padding = 3

lis_filenames_in = list(range(int_start_number, int_end_number + 1))
if int_correction > 0:
    lis_filenames_in.reverse()

for int_filename in lis_filenames_in:
    str_in_name = os.path.join(str_path, str(int_filename).zfill(int_zero_padding) + '.' + str_file_extension)
    str_out_name = os.path.join(str_path, str(int_filename + int_correction).zfill(int_zero_padding) + '.' + str_file_extension)
    os.rename(str_in_name, str_out_name)