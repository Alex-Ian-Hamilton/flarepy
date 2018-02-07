# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:14:45 2017

@author: alex_
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import datetime
import matplotlib as mpl
import pandas as pd
#from sunpy.time import parse_time
#import datetime.timedelta as td
#from sunpy.time import TimeRange
#time_range = TimeRange('2010/03/04 00:10', '2010/03/04 00:20')

# One can supply an argument to AutoMinorLocator to
# specify a fixed number of minor intervals per major interval, e.g.:
# minorLocator = AutoMinorLocator(2)
# would lead to a single minor tick between major ticks.


dt_start = datetime.datetime(2016,2,10)
#dt_end = datetime.datetime(2017,10,1)

"""
# Method to get the list of datetimes for each year between two datetimes.
def get_year_locs(from_date, to_date):
    # Holders
    lis_dt_years = list()
    lis_dt_year_mids = list()

    # Getting the first dates
    dt_year = datetime.datetime(from_date.year, 1, 1)
    dt_mid = dt_year + (datetime.datetime(from_date.year+1, 1, 1) - dt_year) * 0.5

    # For the years
    # Add the first year if the start date is at the very beginnning of that year
    if from_date == dt_year:
        lis_dt_years.append(dt_year)
        dt_year = datetime.datetime(dt_year.year+1, 1, 1)

    # Now add all subsiquent years
    while dt_year <= to_date:
        lis_dt_years.append(dt_year)
        dt_year = datetime.datetime(dt_year.year+1, 1, 1)

    # For mid years
    # Add the first year if the start date is at the very beginnning of that year
    if from_date < dt_mid:
        lis_dt_year_mids.append(dt_mid)
        dt_mid = dt_mid + (datetime.datetime(from_date.year+1, 1, 1) - dt_year) * 0.5

    # Now add all subsiquent year mids
    while dt_mid <= to_date:
        lis_dt_years.append(dt_year)
        dt_year = datetime.datetime(dt_year.year+1, 1, 1)

    return ( lis_dt_years, lis_dt_year_mids )

tup_year_locs = get_year_locs(datetime.datetime(2016,1,1), datetime.datetime(2017,10,1))
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

def suffix(dt_in):
    day_endings = {
    1: 'st',
    2: 'nd',
    3: 'rd',
    21: 'st',
    22: 'nd',
    23: 'rd',
    31: 'st'
}
    return day_endings.get(dt_in.day,'th')


# load a numpy record array from yahoo csv data with fields date,
# open, close, volume, adj_close from the mpl-data/example directory.
# The record array stores python datetime.date as an object array in
# the date column
"""
datafile = cbook.get_sample_data('goog.npy')
try:
    # Python3 cannot load python2 .npy files with datetime(object) arrays
    # unless the encoding is set to bytes. However this option was
    # not added until numpy 1.10 so this example will only work with
    # python 2 or with numpy 1.10 and later.
    r = np.load(datafile, encoding='bytes').view(np.recarray)
except TypeError:
    r = np.load(datafile).view(np.recarray)
"""

# Data
dt_start = datetime.datetime(2016,1,1)
t = np.arange(0.0, 100.0, 0.01)
arr_flo_y_vals = np.sin(2*np.pi*0.1*t)*np.exp(-t*0.01)
lis_dt_x_vals = list()
td_cadence = datetime.timedelta(days=0.2)
td_cadence = datetime.timedelta(days=0.05)
for i in range(0, len(arr_flo_y_vals)):
    lis_dt_x_vals.append(dt_start + i*td_cadence)

lis_td_ranges = [#datetime.timedelta(seconds = 30),
              #datetime.timedelta(minutes = 2),
              datetime.timedelta(hours = 1),
              datetime.timedelta(hours = 2),
              datetime.timedelta(hours = 8),
              datetime.timedelta(hours = 16),
              datetime.timedelta(hours = 34),
              datetime.timedelta(days = 2),
              datetime.timedelta(days = 6),
              datetime.timedelta(days = 12),
              datetime.timedelta(days = 31),
              datetime.timedelta(days = 365.25*0.25),
              datetime.timedelta(days = 365.25*0.4),
              datetime.timedelta(days = 365.25*0.75),
              datetime.timedelta(days = 365.25*1.25),
              datetime.timedelta(days = 365.25*3),
              datetime.timedelta(days = 365.25*5),
              datetime.timedelta(days = 365.25*10),
              datetime.timedelta(days = 365.25*20),
              datetime.timedelta(days = 365.25*30),
              datetime.timedelta(days = 365.25*60)
        ]

lis_tup_td_test_ranges = []
for i in range(1,len(lis_td_ranges)):
    td_window = lis_td_ranges[i] - lis_td_ranges[i-1]

    # Get examples for the range, example for smallest/middle/biggest window
    td_min = lis_td_ranges[i-1] + td_window * 0.005
    td_mid = lis_td_ranges[i-1] + td_window * 0.5
    td_max = lis_td_ranges[i-1] + td_window * 0.995

    # Add these to the test list
    lis_tup_td_test_ranges.append((td_min, td_mid, td_max))

int_count = -1
for tup_test_ranges in lis_tup_td_test_ranges[0:]:
    int_count = int_count + 1
    str_col = 'C' + str(int_count%10)
    int_count_sub = -1
    for test_range in tup_test_ranges:
        int_count_sub = int_count_sub + 1
        if int_count_sub == 0:
            str_win = 'min_window'
        elif int_count_sub == 1:
            str_win = 'mid_window'
        else:
            str_win = 'max_window'

        # Making the dataset
        td_cadence = test_range / t.shape[0]
        lis_dt_x_vals = []
        for i in range(0,t.shape[0]):
            lis_dt_x_vals.append(dt_start + (i * td_cadence))

        # Making the x-axis
        if test_range <= datetime.timedelta(hours=1):
            # Ticks by hour then 5 min
            str_title = 'Ticks by hour then 5 min'
            # Major ticks
            major = mdates.YearLocator()   # every year
            majorFmt = mdates.DateFormatter('%Y')
            # Minor ticks every monday
            minor = mdates.WeekdayLocator(byweekday=1, interval=1, tz=None)
            minorFmt = mdates.DateFormatter('%d')
            # x-label
            str_x_lable = ''
        elif test_range <= datetime.timedelta(hours=2):
            # Ticks by hour then 5 min
            str_title = 'Ticks by hour then 10 min'
            # Major ticks every hour
            major = mdates.HourLocator()
            majorFmt = mdates.DateFormatter('%H:00')
            # Minor ticks every 10 mins
            minor = mdates.MinuteLocator(byminute=[10,20,30,40,50])
            minorFmt = mdates.DateFormatter('%M')
            # x-label
            str_x_lable = lis_dt_x_vals[0].strftime('%D/%m/%Y')
        elif test_range <= datetime.timedelta(hours=8):
            # Ticks by hour then 30 mins
            str_title = 'Ticks by hour then 30 min'
            # Major ticks every hour
            major = mdates.HourLocator()
            majorFmt = mdates.DateFormatter('%H:00')
            # Minor ticks every 10 mins
            minor = mdates.MinuteLocator(byminute=[30])
            minorFmt = mdates.DateFormatter('')
            # x-label
            str_x_lable = lis_dt_x_vals[0].strftime('%D/%m/%Y')
        elif test_range <= datetime.timedelta(hours=16):
            # Ticks by day then hour
            str_title = 'Ticks by day then hour'
            # Major ticks every day
            major = mdates.DayLocator()
            majorFmt = mdates.DateFormatter('%a %d')
            # Minor ticks every hour
            #minor = mdates.HourLocator(byhour=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
            #minorFmt = mdates.DateFormatter('%H')
            minor = mdates.HourLocator(byhour=[3,6,9,12,15,18,21])
            minorFmt = mdates.DateFormatter('%H:00')
            # x-label
            str_x_lable = lis_dt_x_vals[0].strftime('%d/%m/%Y')
        elif test_range <= datetime.timedelta(hours=34):
            # Ticks by day then 3 hours
            str_title = 'Ticks by day then 3 hours'
            # Major ticks every day
            major = mdates.DayLocator()
            majorFmt = mdates.DateFormatter('%a %d')
            # Minor ticks every 3 hours
            minor = mdates.HourLocator(byhour=[6,12,18])
            minorFmt = mdates.DateFormatter('%H:00')
            # x-label
            str_x_lable = lis_dt_x_vals[0].strftime('%b %Y')
        elif test_range <= datetime.timedelta(days = 2):
            # Ticks by week then day
            str_title = 'Ticks by day then 3 hour'
            # Major ticks every day
            major = mdates.DayLocator()
            majorFmt = mdates.DateFormatter('%a %d')
            # Minor ticks every 3 hours
            minor = mdates.HourLocator(byhour=[3,6,9,12,15,18,21])
            minorFmt = mdates.DateFormatter('%H')#:00')
            # x-label
            str_x_lable = lis_dt_x_vals[0].strftime('%b %Y')
        elif test_range <= datetime.timedelta(days = 6):
            # Ticks by week then day
            str_title = 'Ticks by day then quarter day'
            # Major ticks every week (monday)
            major = mdates.DayLocator()
            majorFmt = mdates.DateFormatter('%d')
            # Minor ticks every quarter day (no label)
            minor = mdates.HourLocator(byhour=[6,12,18])
            minorFmt = mdates.DateFormatter('')
            # x-label
            str_x_lable = lis_dt_x_vals[0].strftime('%b %Y')
        elif test_range <= datetime.timedelta(days = 12):
            # Ticks by week then day
            str_title = 'Ticks by week then day'
            # Major ticks every week (monday)
            #major = mdates.MonthLocator()
            major = mdates.WeekdayLocator(byweekday=0, tz=None)
            majorFmt = mdates.DateFormatter('Mon\n%d/%m/%Y')
            # Minor ticks every day
            #minor = mdates.DayLocator(bymonthday=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
            minor = mdates.WeekdayLocator(byweekday=[1,2,3,4,5,6], tz=None)
            minorFmt = mdates.DateFormatter('%d')
            # x-label
            str_x_lable = ''
        elif test_range <= datetime.timedelta(days = 31):
            # Ticks by month then week (mondays)
            str_title = 'Ticks by month then week (mondays)'
            # Major ticks month
            major = mdates.MonthLocator()
            majorFmt = mdates.DateFormatter('%b')
            # Minor ticks every week (Monday)
            minor = mdates.WeekdayLocator(byweekday=1, interval=1, tz=None)
            minorFmt = mdates.DateFormatter('%d')
            # x-label
            str_x_lable = ''
        elif test_range <= datetime.timedelta(days = 365.25*0.20):
            # Ticks by month then week
            str_title = 'Ticks by month then week (1)'
            # Major ticks every month
            major = mdates.MonthLocator(bymonth=[2,3,4,5,6,7,8,9,10,11,12])
            majorFmt = mdates.DateFormatter('%b')
            # Minor ticks every week
            minor = mdates.WeekdayLocator(byweekday=1, interval=1, tz=None)
            minorFmt = mdates.DateFormatter('%d')
            # x-label
            str_x_lable = lis_dt_x_vals[0].strftime('%Y')
        elif test_range <= datetime.timedelta(days = 365.25*0.35):
            # Ticks by month then week
            str_title = 'Ticks by month then week (2)'
            # Major ticks every month
            major = mdates.MonthLocator(bymonth=[1,2,3,4,5,6,7,8,9,10,11,12])
            majorFmt = mdates.DateFormatter('%b')
            # Minor ticks every week
            minor = mdates.WeekdayLocator(byweekday=1, interval=1, tz=None)
            minorFmt = mdates.DateFormatter('%d')
            # x-label
            str_x_lable = lis_dt_x_vals[0].strftime('%Y')
        elif test_range <= datetime.timedelta(days = 365.25*0.75):
            # Ticks by month then week
            str_title = 'Ticks by month then week (3)'
            # Major ticks every month
            major = mdates.MonthLocator(bymonth=[1,2,3,4,5,6,7,8,9,10,11,12])
            majorFmt = mdates.DateFormatter('%b')
            # Minor ticks every week (no labels)
            minor = mdates.WeekdayLocator(byweekday=1, interval=1, tz=None)
            minorFmt = mdates.DateFormatter('')#'%d')
            # x-label
            str_x_lable = lis_dt_x_vals[0].strftime('%Y')
        elif test_range <= datetime.timedelta(days = 365.25*1.25):
            # Ticks by year then month
            str_title = 'Ticks by year then alternate month'
            # Major ticks year
            major = mdates.YearLocator()
            majorFmt = mdates.DateFormatter('%Y')
            # Minor ticks every other month
            minor = mdates.MonthLocator(bymonth=[3,5,7,9,11])#2,4,6,8,10,12])#interval=3)  # every month
            minorFmt = mdates.DateFormatter('%b')
            # x-label
            str_x_lable = ''
        elif test_range <= datetime.timedelta(days = 365.25*3):
            # Ticks by year then quarter
            str_title = 'Ticks by year then quarter'
            # Major ticks every year
            major = mdates.YearLocator()
            majorFmt = mdates.DateFormatter('%Y')
            # Minor ticks every quarter
            minor = mdates.MonthLocator(bymonth=[4,7,10])#interval=3)  # every month
            minorFmt = mdates.DateFormatter('%b')
            # x-label
            str_x_lable = ''
        elif test_range <= datetime.timedelta(days = 365.25*5):
            # Ticks by year then quarter
            str_title = 'Ticks by year then quarter'
            # Major ticks every year
            major = mdates.YearLocator()
            majorFmt = mdates.DateFormatter('%Y')
            # Minor ticks every quarter  (no labels)
            minor = mdates.MonthLocator(bymonth=[4,7,10])#interval=3)  # every month
            minorFmt = mdates.DateFormatter('')
            # x-label
            str_x_lable = ''
        elif test_range <= datetime.timedelta(days = 365.25*10):
            # Ticks by year then 6 months
            str_title = 'Ticks by year then 6 months'
            # Major ticks every year
            major = mdates.YearLocator()
            majorFmt = mdates.DateFormatter('%Y')
            # Minor ticks every 6 months
            minor = mdates.MonthLocator(bymonth=[7])#interval=3)  # every month
            minorFmt = mdates.DateFormatter('')
            # x-label
            str_x_lable = ''
        elif test_range <= datetime.timedelta(days = 365.25*20):
            # Ticks by year then 6 months
            str_title = 'Ticks by 2 years then year'
            # Major ticks every other year
            major = mdates.YearLocator(2)
            majorFmt = mdates.DateFormatter('%Y')
            # Minor ticks every year
            minor = mdates.YearLocator()
            minorFmt = mdates.DateFormatter('')
            # x-label
            str_x_lable = ''
        elif test_range <= datetime.timedelta(days = 365.25*30):
            # Ticks by 5 years then year
            str_title = 'Ticks by 5 years then year'
            # Major ticks every 5 years
            major = mdates.YearLocator(5)
            majorFmt = mdates.DateFormatter('%Y')
            # Minor ticks every year (no labels)
            minor = mdates.YearLocator()
            minorFmt = mdates.DateFormatter('')
            # x-label
            str_x_lable = ''
        elif test_range <= datetime.timedelta(days = 365.25*60):
            # Ticks by 10 years then year
            str_title = 'Ticks by 10 years then year'
            # Major ticks every 10 years
            major = mdates.YearLocator(10)
            majorFmt = mdates.DateFormatter('%Y')
            # Minor ticks every year (no labels)
            minor = mdates.YearLocator()
            minorFmt = mdates.DateFormatter('')
            # x-label
            str_x_lable = ''

        # Now make the plot
        fig, ax = plt.subplots()
        ax.plot(lis_dt_x_vals, arr_flo_y_vals, color=str_col)


        # format the ticks
        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_major_formatter(majorFmt)
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_minor_formatter(minorFmt)
        #ax.xaxis.set_ticks_position('top')
        #print(ax.xaxis.get_text_heights(mpl.backend_bases.RendererBase()))
        #ax.xaxis.OFFSETTEXTPAD = 200
        #plt.xlabel("...", labelpad=200)
        #ax.xaxis._autolabelpos = False
        #ax.xaxis.labelpad = 20
        """
        # Expand limits to cover whole years
        datemin = datetime.date(lis_dt_x_vals[0].year, 1, 1)
        datemax = datetime.date(lis_dt_x_vals[-1].year + 1, 1, 1)
        ax.set_xlim(datemin, datemax)
        """
        """
        # format the coords message box
        def price(x):
            return '$%1.2f' % x
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.format_ydata = price
        """
        ax.grid(True)

        ax.set_title(str_title)
        ax.xaxis.set_label_text(str_x_lable)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        #fig.autofmt_xdate()

        # use a more precise date string for the x axis locations in the
        # MatPlotLib toolbar
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        #plt.title('fig.autofmt_xdate fixes the labels')

        fig.savefig((str(int_count).zfill(2)+'-'+str(int_count_sub)+' ' +str_win+' - '+str(td_cadence)).replace('/','-').replace(':','-')+'plot.png', dpi=900, bbox_inches='tight')
        #plt.show()
