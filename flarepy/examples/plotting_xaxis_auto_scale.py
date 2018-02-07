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


dt_start = datetime.datetime(2016,1,1)
#dt_end = datetime.datetime(2017,10,1)



# Method to get the list of datetimes for each year between two datetimes.
def get_years(from_date, to_date):
    # Holders
    lis_dt_years = list()
    dt_year = datetime.datetime(from_date.year, 1, 1)

    # Add teh first year if the start date is at the very beginnning of that year
    if from_date == dt_year:
        lis_dt_years.append(dt_year)
        dt_year = datetime.datetime(dt_year.year+1, 1, 1)

    # Now add all subsiquent years
    while dt_year <= to_date:
        lis_dt_years.append(dt_year)
        dt_year = datetime.datetime(dt_year.year+1, 1, 1)

    return lis_dt_years


# Method to get the list of datetimes for each month (by the midpoints) between two datetimes.
def get_month_mids(from_date, to_date, months=None):
    dt_month = datetime.datetime(from_date.year, from_date.month, 1)
    if dt_month.month == 12:
        dt_month_plus = datetime.datetime(dt_month.year + 1, 1, 1)
    else:
        dt_month_plus = datetime.datetime(dt_month.year, dt_month.month + 1, 1)

    lis_dt_month_mids = []


    dt_month_mid = dt_month + (dt_month_plus - dt_month) * 0.5
    """
    if dt_month_mid > from_date:
        print('here0')
        lis_dt_month_mids.append(dt_month_mid)
````"""

    while dt_month_mid < to_date:
        lis_dt_month_mids.append(dt_month_mid)
        dt_month = dt_month_plus
        dt_month_plus = dt_month
        if dt_month.month == 12:
            dt_month_plus = datetime.datetime(dt_month.year + 1, 1, 1)
        else:
            dt_month_plus = datetime.datetime(dt_month.year, dt_month.month + 1, 1)
        dt_month_mid = dt_month + (dt_month_plus - dt_month) * 0.5

    if lis_dt_month_mids[0] < from_date:
        lis_dt_month_mids = lis_dt_month_mids[1:]

    return lis_dt_month_mids


# Method to get the list of datetimes for each month (by the midpoints) between two datetimes.
def get_year_mids(from_date, to_date):
    dt_year = datetime.datetime(from_date.year, 1, 1)
    dt_year_plus = datetime.datetime(dt_year.year + 1, 1, 1)

    lis_dt_year_mids = []

    dt_year_mid = dt_year + (dt_year_plus - dt_year) * 0.5

    while dt_year_mid < to_date:
        lis_dt_year_mids.append(dt_year_mid)
        dt_year = dt_year_plus
        dt_year_plus = datetime.datetime(dt_year.year + 1, 1, 1)
        dt_year_mid = dt_year + (dt_year_plus - dt_year) * 0.5

    if lis_dt_year_mids[0] < from_date:
        lis_dt_year_mids = lis_dt_year_mids[1:]

    return lis_dt_year_mids



# Method to get the list of datetimes for each month (starting on jan 1st at 00:00) between two datetimes.
def get_months(from_date, to_date, months=None):
    # If only given a single month, make it a list
    lis_int_months = months
    if not isinstance(months, list):
        lis_int_months = [lis_int_months]
    boo_all = months == None

    # Holders
    lis_dt_days = list()
    lis_dt_months = list()

    # Creates a list of all the days falling between the from_date and to_date range
    for x in range((to_date - from_date).days+1):
        lis_dt_days.append(from_date + datetime.timedelta(days=x))

    # Now transcribe only for the first day of each month
    for dt_day in lis_dt_days:
        if dt_day.day == 1:
            if boo_all:
                lis_dt_months.append(dt_day)
            else:
                if dt_day.month in lis_int_months:
                    lis_dt_months.append(dt_day)

    return lis_dt_months

lis_months = get_months(datetime.datetime(2016,1,1), datetime.datetime(2017,10,1))

# Method to get the list of datetimes for each day split into times between two datetimes.
def get_days_times(from_date, to_date, times=None):
    # If only given a single time, make it a list
    lis_tup_times = times
    if not isinstance(times, list):
        lis_tup_times = [lis_tup_times]

    # Holders
    lis_dt_day = list()
    lis_dt_datetimes = list()

    # Creates a list of all the days falling between the from_date and to_date range
    for x in range((to_date - from_date).days+1):
        lis_dt_day.append(from_date + datetime.timedelta(days=x))

    # Add times if requested
    if lis_tup_times[0] is None:
        # Not times requested, just return the start of each day
        return lis_dt_day
    else:
        for date in lis_dt_day:
            for tup_time in lis_tup_times:
                dt_time = datetime.datetime(date.year, date.month, date.day, tup_time[0], tup_time[1])
                if dt_time < to_date:
                    lis_dt_datetimes.append(dt_time)

    return lis_dt_datetimes

lis_day_times = get_days_times(datetime.datetime(2017,1,1), datetime.datetime(2017,10,1), times=[(00,00), (12,00)])

# Method to get the list of dates for given days of the week (monday = 0, ...) between two datetimes.
def get_days_of_week(from_date, to_date, week_days=[5,6], times=[(0,0)]):
    # If only given a single day, make it a list
    lis_week_days = week_days
    if not isinstance(lis_week_days, list):
        lis_week_days = [lis_week_days]

    # Holders
    lis_temp = list()
    lis_dt_day = list()

    # Creates a list of all the days falling between the from_date and to_date range
    for x in range((to_date - from_date).days+1):
        lis_temp.append(from_date + datetime.timedelta(days=x))

    # Transcribe only the given days to the output list
    for date_record in lis_temp:
        if date_record.weekday() in lis_week_days:
            for tup_time in times:
                int_year = date_record.year
                int_month = date_record.month
                int_day = date_record.day
                lis_dt_day.append(datetime.datetime(int_year, int_month, int_day, tup_time[0], tup_time[1]))

    return lis_dt_day



lis_weeks = get_days_of_week(datetime.datetime(2017,1,1), datetime.datetime(2017,10,1), week_days=[0])

t = np.arange(0.0, 100.0, 0.01)
arr_flo_y_vals = np.sin(2*np.pi*0.1*t)*np.exp(-t*0.01)

lis_td_cadence = [#datetime.timedelta(minutes=0.005),
                  datetime.timedelta(minutes=0.01),
                  datetime.timedelta(minutes=0.02),
                  datetime.timedelta(minutes=0.05),
                  datetime.timedelta(minutes=0.08),
                  datetime.timedelta(minutes=0.1),
                   datetime.timedelta(minutes=0.2),
                    datetime.timedelta(minutes=0.5),
                  datetime.timedelta(minutes=1),
                   datetime.timedelta(minutes=2),
                    datetime.timedelta(minutes=5),
                  datetime.timedelta(minutes=10),
                   datetime.timedelta(minutes=20),
                    datetime.timedelta(minutes=50),
                    datetime.timedelta(minutes=120)
                   ]


for td_cadence in lis_td_cadence[8:14]:
    # Make the data
    lis_dt_x_vals = []
    for i in range(0, len(arr_flo_y_vals)):
        lis_dt_x_vals.append(dt_start + i*td_cadence)
    x_vals = mpl.dates.date2num(lis_dt_x_vals)

    # Get the window time delta
    td_plot = lis_dt_x_vals[-1]-lis_dt_x_vals[0]

    lis_ranges = [
            [datetime.timedelta(days=31), ''],
            [datetime.timedelta(days=356.25*1.5), ''],
            [datetime.timedelta(days=356.25*4), ''],
     ]


    for i in range(1, len(lis_ranges)):
        if td_plot > lis_ranges[i-1][0] and td_plot < lis_ranges[i][0]:
            print(i-1)


    """
    str_temp = ''
    for i in range(0,24):
        for j in range(5,60,5):
            str_temp=str_temp+',('+str(i)+','+str(j)+')'
    """

    ###########
    #  Decide the x-axis parameters
    ###########
    print('\n')
    print(td_plot)
    if td_plot < datetime.timedelta(hours=1):
        print('hour then 5 min:')
        # Ticks by hour then 5 min
        # Major ticks each hour
        lis_dt_x_major_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(1,00),(2,00),(3,00),(4,00),(5,00),(6,00),(7,00),(8,00),(9,00),(10,00)
                                                ,(11,00),(12,00),(13,00),(14,00),(15,00),(16,00),(17,00),(18,00),(19,00),(20,00),(21,00),(22,00),(23,00)])
        lis_dt_x_major_tick_label_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(0,30),(1,30),(2,30),(3,30),(4,30),(5,30),(6,30),(7,30),(8,30),(9,30),(10,30)
                                                ,(11,30),(12,30),(13,30),(14,30),(15,30),(16,30),(17,30),(18,30),(19,30),(20,30),(21,30),(22,30),(23,30)])
        arr_flo_x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_tick_loc)
        arr_flo_x_major_tick_label_loc = mpl.dates.date2num(lis_dt_x_major_tick_label_loc)
        lis_str_x_major_tick_label_lab = pd.to_datetime(lis_dt_x_major_tick_loc).strftime('\n\n%H:%M')
        str_x_major_tick_rot = 'vertical'
        # Minor ticks every 5 minutes (except on the hour)
        #lis_dt_x_minor_ticks = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(1,00),(1,30),(2,00),(2,30),(3,00),(3,30),(4,00),(4,30),(5,00),(5,30),
        #                                         (6,00),(6,30),(7,00),(7,30),(8,00),(8,30),(9,00),(9,30),(10,00),(10,30),(11,00),(11,30),(12,00),(12,30),(13,00),
        #                                         (13,30),(14,00),(14,30),(15,00),(15,30),(16,00),(16,30),(17,00),(17,30),(18,00),(18,30),(19,00),(19,30),(20,00),
        #                                         (20,30),(21,00),(21,30),(22,00),(22,30),(23,00),(23,30)])
        lis_dt_x_minor_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(0,5),(0,10),(0,15),(0,20),(0,25),(0,30),(0,35),(0,40),(0,45),(0,50),(0,55),
                                                 (1,5),(1,10),(1,15),(1,20),(1,25),(1,30),(1,35),(1,40),(1,45),(1,50),(1,55),(2,5),(2,10),(2,15),(2,20),(2,25),(2,30),
                                                 (2,35),(2,40),(2,45),(2,50),(2,55),(3,5),(3,10),(3,15),(3,20),(3,25),(3,30),(3,35),(3,40),(3,45),(3,50),(3,55),(4,5),
                                                 (4,10),(4,15),(4,20),(4,25),(4,30),(4,35),(4,40),(4,45),(4,50),(4,55),(5,5),(5,10),(5,15),(5,20),(5,25),(5,30),(5,35),
                                                 (5,40),(5,45),(5,50),(5,55),(6,5),(6,10),(6,15),(6,20),(6,25),(6,30),(6,35),(6,40),(6,45),(6,50),(6,55),(7,5),(7,10),
                                                 (7,15),(7,20),(7,25),(7,30),(7,35),(7,40),(7,45),(7,50),(7,55),(8,5),(8,10),(8,15),(8,20),(8,25),(8,30),(8,35),(8,40),
                                                 (8,45),(8,50),(8,55),(9,5),(9,10),(9,15),(9,20),(9,25),(9,30),(9,35),(9,40),(9,45),(9,50),(9,55),(10,5),(10,10),(10,15),
                                                 (10,20),(10,25),(10,30),(10,35),(10,40),(10,45),(10,50),(10,55),(11,5),(11,10),(11,15),(11,20),(11,25),(11,30),(11,35),
                                                 (11,40),(11,45),(11,50),(11,55),(12,5),(12,10),(12,15),(12,20),(12,25),(12,30),(12,35),(12,40),(12,45),(12,50),(12,55),
                                                 (13,5),(13,10),(13,15),(13,20),(13,25),(13,30),(13,35),(13,40),(13,45),(13,50),(13,55),(14,5),(14,10),(14,15),(14,20),
                                                 (14,25),(14,30),(14,35),(14,40),(14,45),(14,50),(14,55),(15,5),(15,10),(15,15),(15,20),(15,25),(15,30),(15,35),(15,40),
                                                 (15,45),(15,50),(15,55),(16,5),(16,10),(16,15),(16,20),(16,25),(16,30),(16,35),(16,40),(16,45),(16,50),(16,55),(17,5),
                                                 (17,10),(17,15),(17,20),(17,25),(17,30),(17,35),(17,40),(17,45),(17,50),(17,55),(18,5),(18,10),(18,15),(18,20),(18,25),
                                                 (18,30),(18,35),(18,40),(18,45),(18,50),(18,55),(19,5),(19,10),(19,15),(19,20),(19,25),(19,30),(19,35),(19,40),(19,45),
                                                 (19,50),(19,55),(20,5),(20,10),(20,15),(20,20),(20,25),(20,30),(20,35),(20,40),(20,45),(20,50),(20,55),(21,5),(21,10),
                                                 (21,15),(21,20),(21,25),(21,30),(21,35),(21,40),(21,45),(21,50),(21,55),(22,5),(22,10),(22,15),(22,20),(22,25),(22,30),
                                                 (22,35),(22,40),(22,45),(22,50),(22,55),(23,5),(23,10),(23,15),(23,20),(23,25),(23,30),(23,35),(23,40),(23,45),(23,50),(23,55)])
        arr_flo_x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        arr_flo_x_minor_tick_label_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        lis_str_x_minor_tick_label_lab = pd.to_datetime(lis_dt_x_minor_tick_loc).strftime('\n%M')

        # String for the x-axis label, should contain all datetime details to ass to major/minor ticks to get a full datetime
        str_x_label = '\n\n' + lis_dt_x_vals[0].strftime('%d/%m/%Y')


    elif td_plot < datetime.timedelta(hours=4):
        print('hour then 10 min:')
        # Ticks by hour then 10 min
        # Major ticks each hour
        lis_dt_x_major_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(1,00),(2,00),(3,00),(4,00),(5,00),(6,00),(7,00),(8,00),(9,00),(10,00)
                                                ,(11,00),(12,00),(13,00),(14,00),(15,00),(16,00),(17,00),(18,00),(19,00),(20,00),(21,00),(22,00),(23,00)])
        lis_dt_x_major_tick_label_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(0,30),(1,30),(2,30),(3,30),(4,30),(5,30),(6,30),(7,30),(8,30),(9,30),(10,30)
                                                ,(11,30),(12,30),(13,30),(14,30),(15,30),(16,30),(17,30),(18,30),(19,30),(20,30),(21,30),(22,30),(23,30)])
        arr_flo_x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_tick_loc)
        arr_flo_x_major_tick_label_loc = mpl.dates.date2num(lis_dt_x_major_tick_label_loc)
        lis_str_x_major_tick_label_lab = pd.to_datetime(lis_dt_x_major_tick_loc).strftime('\n\n%H:%M')
        str_x_major_tick_rot = 'vertical'
        # Minor ticks every 10 minutes (except on the hour)
        #lis_dt_x_minor_ticks = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(1,00),(1,30),(2,00),(2,30),(3,00),(3,30),(4,00),(4,30),(5,00),(5,30),
        #                                         (6,00),(6,30),(7,00),(7,30),(8,00),(8,30),(9,00),(9,30),(10,00),(10,30),(11,00),(11,30),(12,00),(12,30),(13,00),
        #                                         (13,30),(14,00),(14,30),(15,00),(15,30),(16,00),(16,30),(17,00),(17,30),(18,00),(18,30),(19,00),(19,30),(20,00),
        #                                         (20,30),(21,00),(21,30),(22,00),(22,30),(23,00),(23,30)])
        lis_dt_x_minor_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(0,10),(0,20),(0,30),(0,40),(0,50),(1,10),(1,20),(1,30),(1,40),(1,50),(2,10),(2,20),(2,30),(2,40),(2,50),(3,10),(3,20),(3,30),(3,40),(3,50),(4,10),(4,20),(4,30),(4,40),(4,50),(5,10),(5,20),(5,30),(5,40),(5,50),(6,10),(6,20),(6,30),(6,40),(6,50),(7,10),(7,20),(7,30),(7,40),(7,50),(8,10),(8,20),(8,30),(8,40),(8,50),(9,10),(9,20),(9,30),(9,40),(9,50),(10,10),(10,20),(10,30),(10,40),(10,50),(11,10),(11,20),(11,30),(11,40),(11,50),(12,10),(12,20),(12,30),(12,40),(12,50),(13,10),(13,20),(13,30),(13,40),(13,50),(14,10),(14,20),(14,30),(14,40),(14,50),(15,10),(15,20),(15,30),(15,40),(15,50),(16,10),(16,20),(16,30),(16,40),(16,50),(17,10),(17,20),(17,30),(17,40),(17,50),(18,10),(18,20),(18,30),(18,40),(18,50),(19,10),(19,20),(19,30),(19,40),(19,50),(20,10),(20,20),(20,30),(20,40),(20,50),(21,10),(21,20),(21,30),(21,40),(21,50),(22,10),(22,20),(22,30),(22,40),(22,50),(23,10),(23,20),(23,30),(23,40),(23,50)])
        arr_flo_x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        arr_flo_x_minor_tick_label_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        lis_str_x_minor_tick_label_lab = pd.to_datetime(lis_dt_x_minor_tick_loc).strftime('\n%M')

        # String for the x-axis label, should contain all datetime details to ass to major/minor ticks to get a full datetime
        str_x_label = '\n\n' + lis_dt_x_vals[0].strftime('%d/%m/%Y')

    elif td_plot < datetime.timedelta(hours=12):
        print('hour then 30 min:')
        # Ticks by hour then 30 min
        # Major ticks each hour
        lis_dt_x_major_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(1,00),(2,00),(3,00),(4,00),(5,00),(6,00),(7,00),(8,00),(9,00),(10,00)
                                                ,(11,00),(12,00),(13,00),(14,00),(15,00),(16,00),(17,00),(18,00),(19,00),(20,00),(21,00),(22,00),(23,00)])
        lis_dt_x_major_tick_label_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(0,30),(1,30),(2,30),(3,30),(4,30),(5,30),(6,30),(7,30),(8,30),(9,30),(10,30)
                                                ,(11,30),(12,30),(13,30),(14,30),(15,30),(16,30),(17,30),(18,30),(19,30),(20,30),(21,30),(22,30),(23,30)])
        arr_flo_x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_tick_loc)
        arr_flo_x_major_tick_label_loc = mpl.dates.date2num(lis_dt_x_major_tick_label_loc)
        lis_str_x_major_tick_label_lab = pd.to_datetime(lis_dt_x_major_tick_loc).strftime('\n\n%H:%M')
        str_x_major_tick_rot = 'vertical'
        # Minor ticks every 5 minutes (except on the hour)
        #lis_dt_x_minor_ticks = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(1,00),(1,30),(2,00),(2,30),(3,00),(3,30),(4,00),(4,30),(5,00),(5,30),
        #                                         (6,00),(6,30),(7,00),(7,30),(8,00),(8,30),(9,00),(9,30),(10,00),(10,30),(11,00),(11,30),(12,00),(12,30),(13,00),
        #                                         (13,30),(14,00),(14,30),(15,00),(15,30),(16,00),(16,30),(17,00),(17,30),(18,00),(18,30),(19,00),(19,30),(20,00),
        #                                         (20,30),(21,00),(21,30),(22,00),(22,30),(23,00),(23,30)])
        lis_dt_x_minor_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(0,30),(1,30),(2,30),(3,30),(4,30),(5,30),(6,30),(7,30),(8,30),(9,30),(10,30)
                                                ,(11,30),(12,30),(13,30),(14,30),(15,30),(16,30),(17,30),(18,30),(19,30),(20,30),(21,30),(22,30),(23,30)])
        arr_flo_x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        arr_flo_x_minor_tick_label_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        lis_str_x_minor_tick_label_lab = pd.to_datetime(lis_dt_x_minor_tick_loc).strftime('\n%M')

        # String for the x-axis label, should contain all datetime details to ass to major/minor ticks to get a full datetime
        str_x_label = '\n\n' + lis_dt_x_vals[0].strftime('%d/%m/%Y')

    elif td_plot < datetime.timedelta(hours=16):
        print('day then hour:')
        # Ticks by day then hour
        # Major ticks each day
        lis_dt_x_major_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        lis_dt_x_major_tick_label_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(12,00)])
        arr_flo_x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_tick_loc)
        arr_flo_x_major_tick_label_loc = mpl.dates.date2num(lis_dt_x_major_tick_label_loc)
        lis_str_x_major_tick_label_lab = pd.to_datetime(lis_dt_x_major_tick_loc).strftime('\n\n%a %d')
        str_x_major_tick_rot = 'vertical'
        # Minor ticks every hour (except on midnight)
        lis_dt_x_minor_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(1,00),(2,00),(3,00),(4,00),(5,00),(6,00),(7,00),(8,00),(9,00),(10,00),(11,00),
                                                 (12,00),(13,00),(14,00),(15,00),(16,00),(17,00),(18,00),(19,00),(20,00),(21,00),(22,00),(23,00)])
        arr_flo_x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        arr_flo_x_minor_tick_label_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        lis_str_x_minor_tick_label_lab = pd.to_datetime(lis_dt_x_minor_tick_loc).strftime('\n%H')

        # String for the x-axis label, should contain all datetime details to ass to major/minor ticks to get a full datetime
        str_x_label = '\n\n' + lis_dt_x_vals[0].strftime('%m/%Y')


    elif td_plot < datetime.timedelta(days=1.5):
        print('day then 2 hours:')
        # Ticks by day then 2 hours
        # Major ticks each day
        lis_dt_x_major_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        lis_dt_x_major_tick_label_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(12,00)])
        arr_flo_x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_tick_loc)
        arr_flo_x_major_tick_label_loc = mpl.dates.date2num(lis_dt_x_major_tick_label_loc)
        lis_str_x_major_tick_label_lab = pd.to_datetime(lis_dt_x_major_tick_loc).strftime('\n\n%a %d')
        str_x_major_tick_rot = 'vertical'
        # Minor ticks every 2 hours (except on midnight)
        #lis_dt_x_minor_ticks = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(1,00),(1,30),(2,00),(2,30),(3,00),(3,30),(4,00),(4,30),(5,00),(5,30),
        #                                         (6,00),(6,30),(7,00),(7,30),(8,00),(8,30),(9,00),(9,30),(10,00),(10,30),(11,00),(11,30),(12,00),(12,30),(13,00),
        #                                         (13,30),(14,00),(14,30),(15,00),(15,30),(16,00),(16,30),(17,00),(17,30),(18,00),(18,30),(19,00),(19,30),(20,00),
        #                                         (20,30),(21,00),(21,30),(22,00),(22,30),(23,00),(23,30)])
        lis_dt_x_minor_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(2,00),(4,00),(6,00),(8,00),(10,00),(12,00),(14,00),(16,00),(18,00),(20,00),(22,00)])
        arr_flo_x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        arr_flo_x_minor_tick_label_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        lis_str_x_minor_tick_label_lab = pd.to_datetime(lis_dt_x_minor_tick_loc).strftime('\n%H')

        # String for the x-axis label, should contain all datetime details to ass to major/minor ticks to get a full datetime
        str_x_label = '\n\n' + lis_dt_x_vals[0].strftime('%m/%Y')

    elif td_plot < datetime.timedelta(days=4):
        print('day then 6h:')
        # Ticks by day then 6h
        # Major ticks each day
        lis_dt_x_major_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        lis_dt_x_major_tick_label_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(12,00)])
        arr_flo_x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_tick_loc)
        arr_flo_x_major_tick_label_loc = mpl.dates.date2num(lis_dt_x_major_tick_label_loc)
        lis_str_x_major_tick_label_lab = pd.to_datetime(lis_dt_x_major_tick_loc).strftime('\n\n%a %d')
        str_x_major_tick_rot = 'vertical'
        # Minor ticks every 6 hours (except on midnight)
        lis_dt_x_minor_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(6,00),(12,00),(18,00)])
        arr_flo_x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        arr_flo_x_minor_tick_label_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        lis_str_x_minor_tick_label_lab = pd.to_datetime(lis_dt_x_minor_tick_loc).strftime('\n%H')

        # String for the x-axis label, should contain all datetime details to ass to major/minor ticks to get a full datetime
        str_x_label = '\n\n' + lis_dt_x_vals[0].strftime('%m/%Y')

    elif td_plot < datetime.timedelta(days=10):
        print('day then 12h:')
        # Ticks by month then week
        # Major ticks each day
        lis_dt_x_major_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        lis_dt_x_major_tick_label_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(12,00)])
        arr_flo_x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_tick_loc)
        arr_flo_x_major_tick_label_loc = mpl.dates.date2num(lis_dt_x_major_tick_label_loc)
        lis_str_x_major_tick_label_lab = pd.to_datetime(lis_dt_x_major_tick_loc).strftime('\n\n%a %d')
        str_x_major_tick_rot = 'vertical'
        # Minor ticks every 12 hours (except on midnight)
        lis_dt_x_minor_tick_loc = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1], times=[(12,00)])
        arr_flo_x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        arr_flo_x_minor_tick_label_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        lis_str_x_minor_tick_label_lab = pd.to_datetime(lis_dt_x_minor_tick_loc).strftime('\n%H')

        # String for the x-axis label, should contain all datetime details to ass to major/minor ticks to get a full datetime
        str_x_label = '\n\n' + lis_dt_x_vals[0].strftime('%m/%Y')

    elif td_plot < datetime.timedelta(days=31):
        print('week then day:')
        # Ticks by week then day
        # Major ticks each week (starts Monday 00:00 and labeled by day)
        lis_dt_x_major_tick_loc = get_days_of_week(lis_dt_x_vals[0],lis_dt_x_vals[-1], week_days=[0])
        lis_dt_x_major_tick_label_loc = get_days_of_week(lis_dt_x_vals[0],lis_dt_x_vals[-1], week_days=[3], times=[(12,00)])
        arr_flo_x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_tick_loc)
        arr_flo_x_major_tick_label_loc = mpl.dates.date2num(lis_dt_x_major_tick_label_loc)
        lis_str_x_major_tick_label_lab = pd.to_datetime(lis_dt_x_major_tick_loc).strftime('\n\nweek from %d %b')
        str_x_major_tick_rot = 'vertical'
        # Minor ticks every day (except monday)
        lis_dt_x_minor_tick_loc = get_days_of_week(lis_dt_x_vals[0],lis_dt_x_vals[-1], week_days=[1,2,3,4,5,6])
        arr_flo_x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        arr_flo_x_minor_tick_label_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        lis_str_x_minor_tick_label_lab = pd.to_datetime(lis_dt_x_minor_tick_loc).strftime('\n%d')#'\n%a %d')

        """
        # Major ticks
        lis_dt_x_major_ticks = get_days_of_week(lis_dt_x_vals[0],lis_dt_x_vals[-1], week_days=[0])
        x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_ticks)
        lis_str_x_major_tick_lab = ''#pd.to_datetime(lis_dt_x_major_ticks).strftime('%b')
        x_major_tick_rot = 'vertical'
        # Minor ticks
        lis_dt_x_minor_ticks = get_days_times(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_ticks)
        lis_str_x_minor_tick_lab = pd.to_datetime(lis_dt_x_minor_ticks).strftime('%a %d')
        """

        # String for the x-axis label, should contain all datetime details to ass to major/minor ticks to get a full datetime
        str_x_label = '\n\n' + lis_dt_x_vals[0].strftime('%m/%Y')

    elif td_plot < datetime.timedelta(days=31*6):
        print('month then week:')
        # Ticks by month then week
        # Major ticks each month (starts 00:00 on first day of the month)
        lis_dt_x_major_tick_loc = get_months(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        lis_dt_x_major_tick_label_loc = get_month_mids(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        arr_flo_x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_tick_loc)
        arr_flo_x_major_tick_label_loc = mpl.dates.date2num(lis_dt_x_major_tick_label_loc)
        lis_str_x_major_tick_label_lab = pd.to_datetime(lis_dt_x_major_tick_loc).strftime('\n\n%b')
        str_x_major_tick_rot = 'vertical'
        # Minor ticks every day (except monday)
        lis_dt_x_minor_tick_loc = get_days_of_week(lis_dt_x_vals[0],lis_dt_x_vals[-1], week_days=[0])
        arr_flo_x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        arr_flo_x_minor_tick_label_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        lis_str_x_minor_tick_label_lab = pd.to_datetime(lis_dt_x_minor_tick_loc).strftime('\n%d')#'\n%a %d')

        str_x_label = '\n\n' + lis_dt_x_vals[0].strftime('%Y')


        """
        # Major ticks
        lis_dt_x_major_ticks = get_months(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_ticks)
        lis_str_x_major_tick_lab = pd.to_datetime(lis_dt_x_major_ticks).strftime('%b')
        x_major_tick_rot = 'vertical'
        # Minor ticks
        lis_dt_x_minor_ticks = get_days_of_week(lis_dt_x_vals[0],lis_dt_x_vals[-1], week_days=[0])
        x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_ticks)
        lis_str_x_minor_tick_lab = pd.to_datetime(lis_dt_x_minor_ticks).strftime('%d')
        """

    elif td_plot < datetime.timedelta(days=356.25*1.5):
        print('year then month:')
        # Ticks by year then month
        # Major ticks each year
        lis_dt_x_major_tick_loc = get_years(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        lis_dt_x_major_tick_label_loc = get_year_mids(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        arr_flo_x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_tick_loc)
        arr_flo_x_major_tick_label_loc = mpl.dates.date2num(lis_dt_x_major_tick_label_loc)
        lis_str_x_major_tick_label_lab = pd.to_datetime(lis_dt_x_major_tick_loc).strftime('\n\n%Y')
        str_x_major_tick_rot = 'vertical'
        # Minor ticks every Month
        lis_dt_x_minor_tick_loc = get_months(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        arr_flo_x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        arr_flo_x_minor_tick_label_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        lis_str_x_minor_tick_label_lab = pd.to_datetime(lis_dt_x_minor_tick_loc).strftime('\n%b')#'\n%M')

        str_x_label = '\n\n'# + lis_dt_x_vals[0].strftime('%Y')

        """

        # Major ticks
        lis_dt_x_major_ticks = get_years(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_ticks)
        lis_str_x_major_tick_lab = pd.to_datetime(lis_dt_x_major_ticks).strftime('%Y')
        x_major_tick_rot = 'vertical'
        # Minor ticks
        lis_dt_x_minor_ticks = get_months(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_ticks)
        lis_str_x_minor_tick_lab = pd.to_datetime(lis_dt_x_minor_ticks).strftime('%b')
        """

    elif td_plot < datetime.timedelta(days=356.25*3):
        print('year then quarter:')
        # Ticks by year then quarter
        # Major ticks each year
        lis_dt_x_major_tick_loc = get_years(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        lis_dt_x_major_tick_label_loc = get_year_mids(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        arr_flo_x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_tick_loc)
        arr_flo_x_major_tick_label_loc = mpl.dates.date2num(lis_dt_x_major_tick_label_loc)
        lis_str_x_major_tick_label_lab = pd.to_datetime(lis_dt_x_major_tick_loc).strftime('\n\n%Y')
        str_x_major_tick_rot = 'vertical'
        # Minor ticks every Month
        lis_dt_x_minor_tick_loc = get_months(lis_dt_x_vals[0],lis_dt_x_vals[-1], months=[4,7,10])
        arr_flo_x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        arr_flo_x_minor_tick_label_loc = mpl.dates.date2num(lis_dt_x_minor_tick_loc)
        lis_str_x_minor_tick_label_lab = pd.to_datetime(lis_dt_x_minor_tick_loc).strftime('\n%b')#'\n%M')

        str_x_label = '\n\n'# + lis_dt_x_vals[0].strftime('%Y')

        """
        # Major ticks
        lis_dt_x_major_ticks = get_years(lis_dt_x_vals[0],lis_dt_x_vals[-1])
        x_major_tick_loc = mpl.dates.date2num(lis_dt_x_major_ticks)
        lis_str_x_major_tick_lab = pd.to_datetime(lis_dt_x_major_ticks).strftime('%Y')
        x_major_tick_rot = 'vertical'
        # Minor ticks
        lis_dt_x_minor_ticks = get_months(lis_dt_x_vals[0],lis_dt_x_vals[-1], months=[1,4,7,10])
        x_minor_tick_loc = mpl.dates.date2num(lis_dt_x_minor_ticks)
        lis_str_x_minor_tick_lab = pd.to_datetime(lis_dt_x_minor_ticks).strftime('%b')
        """

    #minorLocator = AutoMinorLocator(2)



    arr_y_vals = np.sin(2*np.pi*0.1*t)*np.exp(-t*0.01)



    fig, ax = plt.subplots()
    plt.plot(x_vals, arr_flo_y_vals)
    """
    # Adding major ticks (years)
    lis_x_major_tick_loc = mpl.dates.date2num(get_years(lis_dt_x_vals[0],lis_dt_x_vals[-1]))
    lis_x_major_tick_lab = pd.to_datetime(get_years(lis_dt_x_vals[0],lis_dt_x_vals[-1])).strftime('%Y')
    x_major_tick_rot = 'vertical'
    #plt.xticks(lis_x_major_tick_loc, lis_x_major_tick_lab, rotation=x_major_tick_rot)




    # Adding minor ticks (quarters)
    lis_x_minor_tick_loc = mpl.dates.date2num(get_months(lis_dt_x_vals[0],lis_dt_x_vals[-1], months=[1,4,7,10]))
    lis_x_minor_tick_lab = pd.to_datetime(get_years(lis_dt_x_vals[0],lis_dt_x_vals[-1])).strftime('%M')
    x_minor_tick_rot = 'vertical'
    """

    ax.set_xticks(arr_flo_x_major_tick_loc)
    #ax.set_xticklabels(lis_str_x_major_tick_lab, fontdict={'verticalalignment': 'top', 'horizontalalignment': 'center'})
    ax.set_xticklabels([], visible=False)

    ax.set_xticks(arr_flo_x_minor_tick_loc, minor = True)
    ax.set_xticklabels([], visible=False, minor=True)#ax.set_xticklabels(lis_str_x_minor_tick_lab, fontdict={'verticalalignment': 'bottom', 'rotation':'horizontal'}, minor = True)

    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_markersize(30)
        tick.tick2line.set_markersize(30)
        #tick.label1.set_horizontalalignment('center')
        #print(tick.label1.get_position())
        #tick.label1.set_position((tick.label1.get_position()[0]+flo_major_tick_delta,0.0))

    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(6)
        tick.tick2line.set_markersize(6)
        #tick.label1.set_horizontalalignment('center')
        #print(tick.label1.get_position())
        #tick.label1.set_position((tick.label1.get_position()[0]+flo_major_tick_delta,0.0))

    for x_major_label in zip(arr_flo_x_major_tick_label_loc, lis_str_x_major_tick_label_lab):
        ax.text(x_major_label[0], ax.get_ybound()[0], x_major_label[1], fontsize=10, fontdict={'verticalalignment': 'top', 'rotation':'horizontal', 'horizontalalignment':'center'})

    for x_minor_label in zip(arr_flo_x_minor_tick_label_loc, lis_str_x_minor_tick_label_lab):
        ax.text(x_minor_label[0], ax.get_ybound()[0], x_minor_label[1], fontsize=7.5, fontdict={'verticalalignment': 'top', 'rotation':'horizontal', 'horizontalalignment':'center'})

    #ax.set_label('here')
    plt.xlabel(str_x_label, fontsize=12)
    """
    for tick in ax.xaxis.get_majorticklabels():
        #tick.tick1line.set_markersize(10)
        #tick.tick2line.set_markersize(10)
        #tick.label1.set_horizontalalignment('center')
        print(tick.get_position()[0])
        print(tick.get_position()[0] + flo_major_tick_delta)
        temp = tick.get_position()[0] + flo_major_tick_delta
        tick.set_position((temp,0))
    """

    """
    lis_x_major_tick_loc = [0,20,40,60,80,100]
    lis_x_major_tick_lab = ['0s','20s','40s','60s','80s','100s']
    x_major_tick_rot = 'vertical'
    plt.xticks(lis_x_major_tick_loc, lis_x_major_tick_lab, rotation=x_major_tick_rot)
    #ax.set_xticks(lis_x_major_tick_loc, lis_x_major_tick_lab)
    ax.set_xticks([5,10,25,30,35], minor = True)
    #ax.xaxis.set_minor_locator(minorLocator)

    plt.tick_params(which='both', width=2)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='r')
    """
    fig.savefig(str(td_cadence).replace('/','-').replace(':','-')+'plot.png', dpi=900, bbox_inches='tight')
    plt.show()
