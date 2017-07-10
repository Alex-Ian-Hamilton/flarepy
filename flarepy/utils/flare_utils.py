# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:05:58 2017

@author: Alex
"""
import numpy as np

__all__ = ['intensity_to_flare_class', 'flare_class_to_intensity', 'arr_to_cla']

def intensity_to_flare_class(arr_intensity, str_output_type='full', int_dp=0):
    """
    Takes flux intensity in the xrsb band and returns the classification.
    Note, intensities below 10.0**-8.0 get an empty sting.

    Parameters
    ----------
    arr_intensity: arr
        The float or array of intensity values to find the classification of.

    str_output_type: str
        A string to decide the output necessary.
        'full' = the character and number
        'number' = just the number
        'character' = just the character

    int_dp: arr
        The dataset to look for maxima in.

    Returns
    -------
    result: array
        The list of indices for local maxima.
    """

    # Flare classification ranges
    dic_flare_classes = {'A': (10.0**-8.0, 10.0**-7.0),
                         'B': (10.0**-7.0, 10.0**-6.0),
                         'C': (10.0**-6.0, 10.0**-5.0),
                         'M': (10.0**-5.0, 10.0**-4.0),
                         'X': (10.0**-4.0, 10.0**-1.0)}

    # Act differently based on if we have an array or value.
    boo_singular = False
    if isinstance(arr_intensity, float):
        boo_singular = True
        arr_intensity = np.array([arr_intensity])

    # Find class for ach entry
    lis_classes = []
    for i in range(0,len(arr_intensity)):#arr_intensity:
        flo_val = arr_intensity[i]

        # Exception for flux below A-class:
        if flo_val < dic_flare_classes['A'][0]:
            lis_classes.append('')
        else:
            # Now determine the class if applicable
            for key, value in dic_flare_classes.items():
                if (flo_val > value[0]) and (flo_val < value[1]):
                    # Get the components
                    str_class = key
                    flo_class = round(flo_val / value[0], int_dp)

                    # tweak to an integer if 0 int_dp
                    if int_dp == 0:
                        flo_class = int(flo_class)

                    # Add the value to the list
                    if str_output_type == 'full':
                        lis_classes.append(str_class + str(flo_class))
                    elif str_output_type == 'number':
                        lis_classes.append(flo_class)
                    else:
                        lis_classes.append(str_class)

                    # Now break out of the dictionary loop
                    break

    # Return
    if boo_singular:
        return lis_classes[0]
    else:
        return np.array(lis_classes)


def flo_to_cla(flo_intensity, str_output_type='full', int_dp=0):
    """
    Takes flux intensity in the xrsb band and returns the classification.
    Note, intensities below 10.0**-8.0 get an empty sting.

    Parameters
    ----------
    arr_intensity: arr
        The float or array of intensity values to find the classification of.

    str_output_type: str
        A string to decide the output necessary.
        'full' = the character and number
        'number' = just the number
        'character' = just the character

    int_dp: arr
        The dataset to look for maxima in.

    Returns
    -------
    result: array
        The list of indices for local maxima.
    """

    # Flare classification ranges
    dic_flare_classes = {'A': (10.0**-8.0, 10.0**-7.0),
                         'B': (10.0**-7.0, 10.0**-6.0),
                         'C': (10.0**-6.0, 10.0**-5.0),
                         'M': (10.0**-5.0, 10.0**-4.0),
                         'X': (10.0**-4.0, 10.0**-1.0)}

    # Convert
    for key, value in dic_flare_classes.items():
        if (flo_intensity > value[0]) and (flo_intensity < value[1]):
            # Get the components
            str_class = key
            flo_class = round(flo_intensity / value[0], int_dp)

            # tweak to an integer if 0 int_dp
            if int_dp == 0:
                flo_class = int(flo_class)

            # Add the value to the list
            if str_output_type == 'full':
                return str_class + str(flo_class)
            elif str_output_type == 'number':
                return flo_class
            else:
                return str_class

            # Now break out of the dictionary loop
            break
    return np.nan

def arr_to_cla(arr_intensity, str_output_type='full', int_dp=0):
    """
    Takes flux intensity in the xrsb band and returns the classification.
    Note, intensities below 10.0**-8.0 get an empty sting.

    Parameters
    ----------
    arr_intensity: arr
        The float or array of intensity values to find the classification of.

    str_output_type: str
        A string to decide the output necessary.
        'full' = the character and number
        'number' = just the number
        'character' = just the character

    int_dp: arr
        The dataset to look for maxima in.

    Returns
    -------
    result: array
        The list of indices for local maxima.
    """
    lis_results = []

    for i in range(0,len(arr_intensity)):
        lis_results.append(flo_to_cla(arr_intensity[i], str_output_type='full', int_dp=int_dp))

    return np.array(lis_results)


def flare_class_to_intensity(arr_flare_class):
    """
    Given a flare classification string (or array) you get the estimated flux
    intensity of the flare.

    Parameters
    ----------
    arr_flare_class: array str
        #####

    Returns
    -------
    result: array float
        ####
    """
    # Flare classification ranges
    dic_flare_classes = {'A': (10.0**-8.0, 10.0**-7.0),
                         'B': (10.0**-7.0, 10.0**-6.0),
                         'C': (10.0**-6.0, 10.0**-5.0),
                         'M': (10.0**-5.0, 10.0**-4.0),
                         'X': (10.0**-4.0, 10.0**-3.0)}

    # Act differently based on if we have an array or value.
    boo_singular = False
    if isinstance(arr_flare_class, str):
        boo_singular = True
        arr_flare_class = np.array([arr_flare_class])

    # Find class for ach entry
    lis_classes = []
    for i in range(0,len(arr_flare_class)):
        # Clean the input
        str_flare_class = arr_flare_class[i]
        str_flare_class = str_flare_class.replace(' ','')

        # Account for empty input
        if str_flare_class == '':
            lis_classes.append(0.0)
        else:
            str_flare_class_char = str_flare_class[0:1]
            flo_flare_multiplier = float(str_flare_class[1:])
            flo_flare_power = dic_flare_classes.get(str_flare_class_char.upper(), 0.0)[0]
            intensity = flo_flare_multiplier * flo_flare_power
            lis_classes.append(intensity)

    # Return
    if boo_singular:
        return lis_classes[0]
    else:
        return lis_classes
