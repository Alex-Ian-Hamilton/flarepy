# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:21:24 2017

@author: alex_
"""

from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

marker_cycle = cycler("marker", list(mpl.markers.MarkerStyle.markers.keys())[:-4])

x = np.arange(10)
figure = plt.figure()
axes = figure.add_subplot(111)
for i, sty in enumerate(marker_cycle):
   axes.plot(x, x*(i+1), **sty)
#figure.show()
figure.savefig('markers', dpi=900)