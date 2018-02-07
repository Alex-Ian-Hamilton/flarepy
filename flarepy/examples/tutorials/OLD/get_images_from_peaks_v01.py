# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:28:04 2018

@author: alex_
"""

import astropy.units as u

from sunpy.net import Fido, attrs as a

attrs_time = a.Time('2005/01/01 00:10', '2005/01/01 10:00')
result = Fido.search(attrs_time, a.Instrument('eit'))
#result = Fido.search(attrs_time, a.Instrument('eit'))
print(result)