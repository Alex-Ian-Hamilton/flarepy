# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 09:14:50 2017

@author: Alex
"""

import numpy as np
import pandas as pd
#from sunpy.time import TimeRange
from datetime import timedelta, datetime
#import flarepy.utils as utils
from scipy import signal

"""
    Create a simple sine bumps as a proxy for flares.
"""
def gen_sine_peak(width=100, amplitude=10**-4):
    t = np.arange(0.0, 1.0, 1.0 / width) * np.pi
    a = np.sin(t) * amplitude

    """
    fig, ax = plt.subplots()
    line1, = ax.plot(t, a, '-', linewidth=2,
                     label='Dashes set retroactively')
    plt.show()
    """
    return a

"""
    Create a simple sin baseband signal
"""
def gen_sine_baseband(int_length=100,wavelength=10, displacement_wl=0.0, amplitude=2*10**-6, displacement_y=10**-9):
    t = (np.arange(0.0, 2.0 * (int_length / wavelength), 2.0 / wavelength) + displacement_wl * 2.0 ) * np.pi
    a = np.sin(t) * amplitude + amplitude + displacement_y

    return a

"""
    Code to combine a series of 1D peak arrays with integer start positions (can
    be negative or all ow flare to run over the edge).
"""
def combine_data(arr_base=np.zeros([100]),lis_pairs=[]):
    for element in lis_pairs:
        # Consider if the flare goes over the edge of the base
        i_start = element[0]
        arr_data = element[1]
        i_max = arr_data.shape[0] + i_start
        if i_max > arr_base.shape[0]:
            i_max = arr_base.shape[0]
        i_min = i_start
        #print('('+str(i_min)+', '+str(i_max)+')')
        if i_min < 0:
            i_min = 0

        # Run through all the elements
        #print(range(i_min, i_max))
        for i in range(i_min, i_max):
            arr_base[i] = arr_base[i] + element[1][i-element[0]]

    return arr_base

"""

"""
def gen_noise(std=0.000001, int_length=100):
    arr_noise = np.random.normal(0,std,int_length)
    return arr_noise


"""
    Make an ensemble of sine flares.
"""
def gen_flare_data(int_length=100, int_flares=10):
    arr_flare_amplitudes =  10**-7 + np.random.exponential(scale=1.0,size=int_flares) * 10**-5
    #np.random.uniform(10**-7,5*10**-4,int_flares)#np.absolute(np.random.normal(0,std,int_length))
    arr_flare_displacements = np.random.randint(0,int_length,int_flares)#np.random.uniform(0,int_length,int_flares)#np.random.normal(0,std,int_length)
    arr_flare_widths = np.random.uniform(5,50,int_flares)#np.random.normal(0,std,int_length)

    lis_flares = []

    for i in range(0,int_flares):
        lis_flares.append((arr_flare_displacements[i], gen_sine_peak(width=arr_flare_widths[i], amplitude=arr_flare_amplitudes[i])))

    # Combine the plots
    arr_data = combine_data(arr_base=np.zeros([int_length]), lis_pairs=lis_flares)

    # Make a DF of the flare population
    df_flares = pd.DataFrame(data={'peak_flux': arr_flare_amplitudes, 'width': arr_flare_widths}, index=arr_flare_displacements)

    return arr_data, df_flares

def gen_synthetic_data(int_length=4000, int_flares=100):
    arr_flares, pd_flares = gen_flare_data(int_length=4000, int_flares=int_flares)
    arr_base = gen_sine_baseband(int_length=int_length,wavelength=1000, displacement_wl=0.0, amplitude=2*10**-6)
    arr_noise = gen_noise(std=0.0000002, int_length=int_length)

    arr_combined = np.add(np.add(arr_base, arr_flares), arr_noise)

    return arr_combined, arr_flares, arr_base, arr_noise, pd_flares

"""
    Making regions of np.nan values.
"""
def gen_data_gaps(arr_base=np.zeros([100]), int_gaps=4):
    arr_gap_width = np.random.randint(0,100,int_gaps)
    arr_gap_start = np.random.randint(0,arr_base.shape[0],int_gaps)

    print(arr_gap_width)
    print(arr_gap_start)
    for i in range(0, arr_gap_width.shape[0]):
        int_gap_width = arr_gap_width[i]
        i_start = arr_gap_start[i]
        i_max = int_gap_width + i_start
        if i_max > arr_base.shape[0]:
            i_max = arr_base.shape[0]
        i_min = i_start
        #print('('+str(i_min)+', '+str(i_max)+')')
        if i_min < 0:
            i_min = 0


        for j in range(i_min, i_max):
            arr_base[j] = np.nan

    return arr_base

"""
    Run and create some sythetic goes data.
"""
def gen_synthetic_goes_xrs():
    int_length = 4000
    int_flares = 1000

    #a, df_flares = gen_flare_data(int_length=int_length, int_flares=int_flares)
    x = np.arange(0.0,int_length,1.0)

    y = gen_sine_baseband(int_length=int_length,wavelength=4000, displacement_wl=0.2, amplitude=2*10**-6, displacement_y=5*10**-9)
    y_noise = gen_noise(std=0.00000002, int_length=int_length)
    y = np.add(y, y_noise)
    y_flares, arr_flares, arr_base, arr_noise, pd_flares = gen_synthetic_data(int_length=4000, int_flares=int_flares)
    y = np.add(y, y_flares)
    y = gen_data_gaps(y)

    # Make x-axis datetime
    dt_cadence = timedelta(seconds=3)
    dt_start = datetime(2017, 7, 7, 12, 00, 00)
    lis_dt_x = []
    for i in range(0, int_length):
        lis_dt_x.append(dt_start + i * dt_cadence)
    dti_index = pd.to_datetime(lis_dt_x)

    # Now make a dataframe from the data
    ser_synth = pd.Series(data=y, index=dti_index)

    fig = utils.plot_goes_2({'xrsa - raw': ser_synth}, ylim=(1e-9, 1e-0))
    fig.savefig('C:\\flare_outputs\\2017-07-07\\generated_data\\test0.png', dpi=900, bbox_inches='tight')


    #print('\n'+str(df_flares['peak_flux'].values.min())+' to '+str(df_flares['peak_flux'].values.max())+'\n')
    #print('\n'+str(a.min())+' to '+str(a.max())+'\n')

    import matplotlib.pyplot as plt


    fig, ax = plt.subplots()
    #ax.set_yscale("log")
    line1, = ax.plot(x, y, '-', linewidth=0.1,
                     label='Dashes set retroactively')
    plt.show()
    fig.savefig('C:\\flare_outputs\\2017-07-06\\generated_data\\test.png', dpi=900, bbox_inches='tight')

"""
    A basic function to create a 1D array of values for a given function with
    given parameters.
    Note that the parameters given will vary depending on teh function used.
    If you wish to combine for datasets of multiple different function types
    then use gen_synthetic_1D_dataset multiple times and combine the results using numpy.

    Note:
    When using the signal.ricker(length, a) function, the a defines the wisth of the
    function we expect, you want length (the number of points calculated) to be notably
    larger then this value, generally length >= 10x a, to avoid clipping that would
    cause discontinuities.

    Parameters
    ----------
    points : `int`
        Length or number of points in output array.

    parameters : `list` of `list`
        For each element in teh list we have a list of all teh parameter values
        to be passed to the function.
        E.G. for scipy.signal.ricker(points, a) we would have a list of lists
        with values for points and a.

    positions : ~`numpy.ndarray` of `int`
        Ann array giving the position (integer) values for each function
        contribution to be added.

    function :
        The function method that will be run for each required entry, with the
        parameters passed from the parameters array.

    pos_align : `str`
        Define where the positions relate WRT the resulting function values.
        Generally safest to use 'centre' and negative values are allowed.
        'centre': will position the function entry to centre at the given position,
                so extra point values will roll off the start and end off the output array.
        'start': will position the function entry to start at the given position,
                with extra point values rolling off the end of the output array.
        'centre': will position the function entry to end at the given position,
                with extra pont values rolling off the begining of the output array.

    verbose : `bool`
        If True then you will get feedback of values used in the console.

    Returns
    -------
    result : ~`numpy.ndarray`
        The array of the combined function entries with a length of the points parameter.

    Examples
    --------
    Make an array which has a single scipy.signal.ricker() function with given
    parameters at given (central) position:

    >>> import numpy as np
    >>> import flarepy.synthetic_data_generation as gen

    >>> length = 1000
    >>> parameters = [[700, 70]] # points and a-value for each wavelet
    >>> positions = [500] # Central starting position for each wavelet
    >>> arr_data = gen.gen_synthetic_1D_dataset(length, parameters, positions)

    To make an array which is the sum of a set of 3 scipy.signal.ricker() functions with given parameters.

    >>> length = 1000
    >>> parameters = [[500, 20],[800, 30],[1000, 40]] # points and a-value for each wavelet
    >>> positions = [50, 300, 650 ] # Central starting position for each wavelet
    >>> arr_data = gen.gen_synthetic_1D_dataset(length, parameters, positions)

    To plot this:
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(arr_data)
"""
def gen_synthetic_1D_dataset(points, parameters, positions, function=signal.ricker, pos_align='centre', verbose=False):
    # Make a list to add all contributions to
    lis_contributions = []

    # For each given value (generally arrays of randon numbers)
    for i in range(0, len(positions)):
        # Get the function datapoints
        arr_function = function(*parameters[i])

        # Align the function to start at the given position
        x_pre_space = positions[i]
        if pos_align == 'centre':
            # Align the function to centre on the given position.
            x_pre_space = x_pre_space - int(len(arr_function) / 2.0)
        elif pos_align == 'end':
            # Align the function to end at the given position
            x_pre_space = x_pre_space - len(arr_function)
        if verbose:
            print('positions[i]: '+str(positions[i]))
            print('x_pre_space: '+str(x_pre_space))

        # Add this entry to the list to be combined
        lis_contributions.append([x_pre_space, arr_function])

    # Combine into the main dataset
    arr_out = combine_data(np.zeros(points),lis_contributions)

    # Return
    return arr_out



#gen_synthetic_goes_xrs()