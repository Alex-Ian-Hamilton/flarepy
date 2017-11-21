# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:30:07 2017

@author: alex_
"""

from sunpy.io import fits, file_tools as sunpy_filetools
import sunpy.database
from astropy.units import Unit, nm, equivalencies
from sunpy.database.tables import *
from sunpy.time import parse_time, TimeRange, is_time_in_given_format
from sunpy.extern import six
from datetime import datetime, timedelta
import os
import sunpy.timeseries as ts
from sunpy.util import MetaDict

def entries_from_goes_file(file, default_waveunit=None, source=None):
    """Use the headers of a FITS file to generate an iterator of
    :class:`sunpy.database.tables.DatabaseEntry` instances. Gathered
    information will be saved in the attribute `fits_header_entries`. If the
    key INSTRUME, WAVELNTH or DATE-OBS / DATE_OBS is available, the attribute
    `instrument`, `wavemin` and `wavemax` or `observation_time_start` is set,
    respectively. If the wavelength unit can be read, the values of `wavemin`
    and `wavemax` are converted to nm (nanometres). The value of the `file`
    parameter is used to set the attribute `path` of each generated database
    entry.

    Parameters
    ----------
    file : str or file-like object
        Either a path pointing to a FITS file or a an opened file-like object.
        If an opened file object, its mode must be one of the following rb,
        rb+, or ab+.

    default_waveunit : str, optional
        The wavelength unit that is used for a header if it cannot be
        found.

    Raises
    ------
    sunpy.database.WaveunitNotFoundError
        If `default_waveunit` is not given and the wavelength unit cannot
        be found in one of the FITS headers

    sunpy.WaveunitNotConvertibleError
        If a wavelength unit could be found but cannot be used to create an
        instance of the type ``astropy.units.Unit``. This can be the case
        for example if a FITS header has the key `WAVEUNIT` with the value
        `nonsense`.

    Examples
    --------
    >>> from sunpy.database.tables import entries_from_file
    >>> import sunpy.data
    >>> sunpy.data.download_sample_data(overwrite=False)   # doctest: +SKIP
    >>> import sunpy.data.sample
    >>> entries = list(entries_from_file(sunpy.data.sample.SWAP_LEVEL1_IMAGE))
    >>> len(entries)
    1
    >>> entry = entries.pop()
    >>> entry.instrument
    'SWAP'
    >>> entry.observation_time_start, entry.observation_time_end
    (datetime.datetime(2012, 1, 1, 0, 16, 7, 836000), None)
    >>> entry.wavemin, entry.wavemax
    (17.400000000000002, 17.400000000000002)
    >>> len(entry.fits_header_entries)
    111

    """
    headers = fits.get_header(file)
    if isinstance(file, (str, six.text_type)):
        filename = file
    else:
        filename = getattr(file, 'name', None)
    for header in headers[0:1]:
        statinfo = os.stat(file)
        #print('a header')
        entry = DatabaseEntry(path=filename)
        entry.size = statinfo.st_size
        """
        for key, value in six.iteritems(header):
            # Yes, it is possible to have an empty key in a FITS file.
            # Example: sunpy.data.sample.EIT_195_IMAGE
            # Don't ask me why this could be a good idea.
            if key == '':
                value = str(value)
            elif key == 'KEYCOMMENTS':
                for k, v in six.iteritems(value):
                    entry.fits_key_comments.append(FitsKeyComment(k, v))
                continue
            entry.fits_header_entries.append(FitsHeaderEntry(key, value))

            #
            if key == 'TELESCOP':        # Not 'INSTRUME'
                entry.instrument = value # So E.G. 'GOES 6' instead 'X-ray Detector'

            # NOTE: the key DATE-END or DATE_END is not part of the official
            # FITS standard, but many FITS files use it in their header
            elif key in ('DATE-END', 'DATE_END'):

                entry.observation_time_end = parse_time(value)
            elif key in ('DATE-OBS', 'DATE_OBS'):
                entry.observation_time_start = parse_time(value)
        """



        # Add/tweak start/end entries for GOES
        if header.get('TELESCOP','') != '':
            #header['INSTRUME'] = header['TELESCOP']# So E.G. 'GOES 6' instead 'X-ray Detector'
            entry.instrument = header['TELESCOP']
        if (header.get('DATE-OBS','') != '') and (header.get('DATE-END','') != ''):
            if is_time_in_given_format(header['DATE-OBS'], '%d/%m/%Y'):
                start_time = datetime.strptime(header['DATE-OBS'], '%d/%m/%Y')
            elif is_time_in_given_format(header['DATE-OBS'], '%d/%m/%y'):
                start_time = datetime.strptime(header['DATE-OBS'], '%d/%m/%y')
            end_time = start_time + timedelta(days=1,seconds=-1)
            #header['DATE-OBS'] = start_time.strftime('%Y/%m/%d')#'%d/%m/%Y')
            #header['TIME-OBS'] = start_time.strftime('%H:%M:%S')
            #header['DATE-END'] = end_time.strftime('%Y/%m/%d')#'%d/%m/%Y')
            #header['TIME-END'] = end_time.strftime('%H:%M:%S')
            # Add these to the entry
            entry.observation_time_start = start_time
            entry.observation_time_end = end_time


        #print('')
        #print(dir(entry))
        #print('')
        #entry.wavemax = 0.8 * nm  # XRSB '1.0--8.0 $\AA$'
        #entry.wavemin = 0.05 * nm # XRSA '0.5--4.0 $\AA$'
        entry.wavemax = 0.8  # XRSB '1.0--8.0 $\AA$'
        entry.wavemin = 0.05 # XRSA '0.5--4.0 $\AA$'
        """
        waveunit = fits.extract_waveunit(header)
        if waveunit is None:
            waveunit = default_waveunit
        unit = None
        if waveunit is not None:
            try:
                unit = Unit(waveunit)
            except ValueError:
                raise WaveunitNotConvertibleError(waveunit)
        """
        """
        for header_entry in entry.fits_header_entries:
            key, value = header_entry.key, header_entry.value
            if key == 'INSTRUME':
                entry.instrument = value
            elif key == 'WAVELNTH':
                if unit is None:
                    raise WaveunitNotFoundError(file)
                # use the value of `unit` to convert the wavelength to nm
                entry.wavemin = entry.wavemax = unit.to(
                    nm, value, equivalencies.spectral())
        """
        yield entry

def entries_from_goes_ts_files(*files, default_waveunit=None, source=None):
    """Use the headers of a FITS file to generate an iterator of
    :class:`sunpy.database.tables.DatabaseEntry` instances. Gathered
    information will be saved in the attribute `fits_header_entries`. If the
    key INSTRUME, WAVELNTH or DATE-OBS / DATE_OBS is available, the attribute
    `instrument`, `wavemin` and `wavemax` or `observation_time_start` is set,
    respectively. If the wavelength unit can be read, the values of `wavemin`
    and `wavemax` are converted to nm (nanometres). The value of the `file`
    parameter is used to set the attribute `path` of each generated database
    entry.

    Parameters
    ----------
    file : str or file-like object
        Either a path pointing to a FITS file or a an opened file-like object.
        If an opened file object, its mode must be one of the following rb,
        rb+, or ab+.

    default_waveunit : str, optional
        The wavelength unit that is used for a header if it cannot be
        found.

    Raises
    ------
    sunpy.database.WaveunitNotFoundError
        If `default_waveunit` is not given and the wavelength unit cannot
        be found in one of the FITS headers

    sunpy.WaveunitNotConvertibleError
        If a wavelength unit could be found but cannot be used to create an
        instance of the type ``astropy.units.Unit``. This can be the case
        for example if a FITS header has the key `WAVEUNIT` with the value
        `nonsense`.

    Examples
    --------
    >>> from sunpy.database.tables import entries_from_file
    >>> import sunpy.data
    >>> sunpy.data.download_sample_data(overwrite=False)   # doctest: +SKIP
    >>> import sunpy.data.sample
    >>> entries = list(entries_from_file(sunpy.data.sample.SWAP_LEVEL1_IMAGE))
    >>> len(entries)
    1
    >>> entry = entries.pop()
    >>> entry.instrument
    'SWAP'
    >>> entry.observation_time_start, entry.observation_time_end
    (datetime.datetime(2012, 1, 1, 0, 16, 7, 836000), None)
    >>> entry.wavemin, entry.wavemax
    (17.400000000000002, 17.400000000000002)
    >>> len(entry.fits_header_entries)
    111

    """


    """
    ts_goes = ts.TimeSeries(file)
    statinfo = os.stat(file)
    entry = DatabaseEntry(path=file)
    entry.size = statinfo.st_size

    #header['INSTRUME'] = header['TELESCOP']# So E.G. 'GOES 6' instead 'X-ray Detector'
    entry.instrument = ts_goes.meta.get('TELESCOP').values()
    entry.instrument = ts_goes.meta.get('TELESCOP').values()

    entry.wavemax = 0.8  # XRSB '1.0--8.0 $\AA$'
    entry.wavemin = 0.05 # XRSA '0.5--4.0 $\AA$'

    #
    entry.observation_time_start = ts_goes.meta.get('date-beg').values()[0]
    entry.observation_time_end = ts_goes.meta.get('date-end').values()[0]

    entry.metadata = ts_goes.meta.metadata[0][2]

    #entry.tags = [ sunpy.database.attrs.Tag('raw') ]
    """


    for file in files:
        headers = fits.get_header(file)
        if isinstance(file, (str, six.text_type)):
            filename = file
        else:
            filename = getattr(file, 'name', None)
        statinfo = os.stat(file)
        #print('a header')
        entry = DatabaseEntry(path=filename)
        entry.size = statinfo.st_size

        # Add/tweak start/end entries for GOES
        if headers[0].get('TELESCOP','') != '':
            #header['INSTRUME'] = header['TELESCOP']# So E.G. 'GOES 6' instead 'X-ray Detector'
            entry.instrument = headers[0]['TELESCOP']
        elif headers[1].get('TELESCOP','') != '':
            entry.instrument = headers[1]['TELESCOP']
        if (headers[0].get('DATE-OBS','') != ''):
            if is_time_in_given_format(headers[0]['DATE-OBS'], '%d/%m/%Y'):
                start_time = datetime.strptime(headers[0]['DATE-OBS'], '%d/%m/%Y')
            elif is_time_in_given_format(headers[0]['DATE-OBS'], '%d/%m/%y'):
                start_time = datetime.strptime(headers[0]['DATE-OBS'], '%d/%m/%y')
            else:
                start_time = parse_time(headers[0]['DATE-OBS'])
        elif (headers[1].get('DATE-OBS','') != ''):
            if is_time_in_given_format(headers[1]['DATE-OBS'], '%d/%m/%Y'):
                start_time = datetime.strptime(headers[1]['DATE-OBS'], '%d/%m/%Y')
            elif is_time_in_given_format(headers[1]['DATE-OBS'], '%d/%m/%y'):
                start_time = datetime.strptime(headers[1]['DATE-OBS'], '%d/%m/%y')
            else:
                start_time = parse_time(headers[1]['DATE-OBS'])

        if (headers[0].get('DATE-END','') != ''):
            if is_time_in_given_format(headers[0]['DATE-END'], '%d/%m/%Y'):
                end_time = datetime.strptime(headers[0]['DATE-END'], '%d/%m/%Y')
            elif is_time_in_given_format(headers[0]['DATE-END'], '%d/%m/%y'):
                end_time = datetime.strptime(headers[0]['DATE-END'], '%d/%m/%y')
            else:
                end_time = parse_time(headers[0]['DATE-END'])
        elif (headers[1].get('DATE-END','') != ''):
            if is_time_in_given_format(headers[1]['DATE-END'], '%d/%m/%Y'):
                end_time = datetime.strptime(headers[1]['DATE-END'], '%d/%m/%Y')
            elif is_time_in_given_format(headers[1]['DATE-END'], '%d/%m/%y'):
                end_time = datetime.strptime(headers[1]['DATE-END'], '%d/%m/%y')
            else:
                end_time = parse_time(headers[1]['DATE-END'])
        else:
            end_time = start_time + timedelta(days=1,seconds=-1)

        # Add these to the entry
        entry.observation_time_start = start_time
        entry.observation_time_end = end_time

        entry.wavemax = 0.8  # XRSB '1.0--8.0 $\AA$'
        entry.wavemin = 0.05 # XRSA '0.5--4.0 $\AA$'

        if source:
            entry.source = source

        entry.metadata = MetaDict(headers[1])
        #entry.tags = sunpy.database.attrs.Tag('raw')

        #entry = DatabaseEntry(instrument='EIT', wavemin=25.0)

    #return entry
    yield entry


def add_entries_from_goes_ts_files(database, *files, tags=[], default_waveunit=None, source=None):
    for entry in entries_from_goes_ts_files(*files, default_waveunit=default_waveunit, source=source):
        database.add(entry)
        for tag in tags:
            database.tag(entry, tag)

















def entries_from_goes_ts_file2(file, default_waveunit=None):
    """Use the headers of a FITS file to generate an iterator of
    :class:`sunpy.database.tables.DatabaseEntry` instances. Gathered
    information will be saved in the attribute `fits_header_entries`. If the
    key INSTRUME, WAVELNTH or DATE-OBS / DATE_OBS is available, the attribute
    `instrument`, `wavemin` and `wavemax` or `observation_time_start` is set,
    respectively. If the wavelength unit can be read, the values of `wavemin`
    and `wavemax` are converted to nm (nanometres). The value of the `file`
    parameter is used to set the attribute `path` of each generated database
    entry.

    Parameters
    ----------
    file : str or file-like object
        Either a path pointing to a FITS file or a an opened file-like object.
        If an opened file object, its mode must be one of the following rb,
        rb+, or ab+.

    default_waveunit : str, optional
        The wavelength unit that is used for a header if it cannot be
        found.

    Raises
    ------
    sunpy.database.WaveunitNotFoundError
        If `default_waveunit` is not given and the wavelength unit cannot
        be found in one of the FITS headers

    sunpy.WaveunitNotConvertibleError
        If a wavelength unit could be found but cannot be used to create an
        instance of the type ``astropy.units.Unit``. This can be the case
        for example if a FITS header has the key `WAVEUNIT` with the value
        `nonsense`.
    """

    headers = fits.get_header(file)
    if isinstance(file, (str, six.text_type)):
        filename = file
    else:
        filename = getattr(file, 'name', None)

    statinfo = os.stat(file)
    #print('a header')
    entry = DatabaseEntry(path=filename)
    size = statinfo.st_size

    # Add/tweak start/end entries for GOES
    if headers[0].get('TELESCOP','') != '':
        #header['INSTRUME'] = header['TELESCOP']# So E.G. 'GOES 6' instead 'X-ray Detector'
        entry.instrument = headers[0]['TELESCOP']
    if (headers[0].get('DATE-OBS','') != ''):
        if is_time_in_given_format(headers[0]['DATE-OBS'], '%d/%m/%Y'):
            start_time = datetime.strptime(headers[0]['DATE-OBS'], '%d/%m/%Y')
        elif is_time_in_given_format(headers[0]['DATE-OBS'], '%d/%m/%y'):
            start_time = datetime.strptime(headers[0]['DATE-OBS'], '%d/%m/%y')
        else:
            start_time = parse_time(headers[0]['DATE-OBS'])
    elif (headers[1].get('DATE-OBS','') != ''):
        if is_time_in_given_format(headers[1]['DATE-OBS'], '%d/%m/%Y'):
            start_time = datetime.strptime(headers[1]['DATE-OBS'], '%d/%m/%Y')
        elif is_time_in_given_format(headers[1]['DATE-OBS'], '%d/%m/%y'):
            start_time = datetime.strptime(headers[1]['DATE-OBS'], '%d/%m/%y')
        else:
            start_time = parse_time(headers[1]['DATE-OBS'])

    if (headers[0].get('DATE-END','') != ''):
        if is_time_in_given_format(headers[0]['DATE-END'], '%d/%m/%Y'):
            end_time = datetime.strptime(headers[0]['DATE-END'], '%d/%m/%Y')
        elif is_time_in_given_format(headers[0]['DATE-END'], '%d/%m/%y'):
            end_time = datetime.strptime(headers[0]['DATE-END'], '%d/%m/%y')
        else:
            end_time = parse_time(headers[0]['DATE-END'])
    elif (headers[1].get('DATE-END','') != ''):
        if is_time_in_given_format(headers[1]['DATE-END'], '%d/%m/%Y'):
            end_time = datetime.strptime(headers[1]['DATE-END'], '%d/%m/%Y')
        elif is_time_in_given_format(headers[1]['DATE-END'], '%d/%m/%y'):
            end_time = datetime.strptime(headers[1]['DATE-END'], '%d/%m/%y')
        else:
            end_time = parse_time(headers[1]['DATE-END'])
    else:
        end_time = start_time + timedelta(days=1,seconds=-1)

    # Add these to the entry
    observation_time_start = start_time
    observation_time_end = end_time

    wavemax = 0.8  # XRSB '1.0--8.0 $\AA$'
    wavemin = 0.05 # XRSA '0.5--4.0 $\AA$'

    metadata = MetaDict(headers[1])
    #entry.tags = sunpy.database.attrs.Tag('raw')

    entry = DatabaseEntry(observation_time_start=start_time,
                          observation_time_end = end_time,
                          instrument='EIT',
                          wavemin=wavemin,
                          wavemax=wavemax,
                          metadata=metadata,
                          size=size)

    return entry



