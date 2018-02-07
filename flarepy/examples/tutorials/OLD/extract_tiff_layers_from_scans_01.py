# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:21:29 2018

@author: alex_
"""

from PIL import Image, TiffImagePlugin
import glob
import os

count = 0

str_path_in = 'C:\\Users\\alex_\\Documents\\Scanned Documents'
str_path_out = 'C:\\Users\\alex_\\Documents\\Scanned Documents\\temp4'

for str_filepath in glob.glob(str_path_in+'\\Image (5).tif'):
    # Open the file and ask for the first page number
    img = Image.open(str_filepath)
    img.show()
    count = int(input("What is the first page number? ")) - 1

    # try looking for upto 100 layers
    for i in range(100):
        count = count + 1
        try:
            img.seek(i)
            #print img.getpixel( (0, 0))

            #
            str_savepath = os.path.join(str_path_out, str(count).zfill(3) + '.tiff')

            #
            #img.save(str_savepath, "JPEG")
            TiffImagePlugin.WRITE_LIBTIFF = True
            img.save(str_savepath, "TIFF", compression='packbits')

            str_savepath = os.path.join(str_path_out, str(count).zfill(3) + '.jpeg')
            img.save(str_savepath, "JPEG", compression='packbits')

        except EOFError:
            # Not enough frames in img
            break