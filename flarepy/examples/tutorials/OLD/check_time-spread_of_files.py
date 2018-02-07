# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:21:29 2018

@author: alex_
"""

from PIL import Image, TiffImagePlugin
import glob
import os
import numpy as np


lis_int_expected = list(range(0,10))
str_path = 'C:\\Users\\alex_\\Desktop\\GXCAPTURE images'
lis_timestamps = []


for str_filepath in glob.glob(os.path.join(str_path,'*.tif')):
    lis_timestamps.append(os.path.getmtime(str_filepath))


    #img = Image.open(str_filepath)
    #img.show()
    #count = int(input("What is the first page number? ")) - 1

lis_timestamps.sort()
arr_flo_times = np.array(lis_timestamps)
arr_flo_times = arr_flo_times - arr_flo_times[0]
print(arr_flo_times)

flo_exp = 0.182342
arr_flo_exp_comp = np.arange(10) * flo_exp
arr_flo_times = arr_flo_times - arr_flo_exp_comp

print(arr_flo_times)