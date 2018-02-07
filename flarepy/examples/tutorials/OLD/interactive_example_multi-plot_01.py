# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:25:31 2018

@author: alex_
"""

#import numpy as np
#import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import time

"""
# Making a plot for the population
fig_sig = plt.figure()
ax_sig = plt.axes([0.05, 0.05, 0.9, 0.9])
plt.show()

# Making a plot for the population
fig_pop = plt.figure()
ax_pop = plt.axes([0.05, 0.05, 0.9, 0.9])
plt.show()

time.sleep(100)
"""
# Working with multiple figure windows and subplots
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = np.sin(4*np.pi*t)

fig_sig = plt.figure(1)
ax_sig = plt.axes([0.05, 0.05, 0.9, 0.9])

#plt.subplot(211)
#plt.plot(t, s1)
#plt.subplot(212)
#plt.plot(t, 2*s1)

fig_pop = plt.figure()
ax_pop = plt.axes([0.05, 0.05, 0.9, 0.9])
ax_pop.plot(t, s2)

"""
# now switch back to figure 1 and make some changes
plt.figure(1)
plt.subplot(211)
plt.plot(t, s2, 's')
ax = plt.gca()
ax.set_xticklabels([])
"""

plt.show()