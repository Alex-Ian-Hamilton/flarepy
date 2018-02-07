# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:51:11 2018

@author: alex_
"""


from scipy.signal.wavelets import ricker
from scipy.stats import scoreatpercentile
from scipy import signal
import scipy.fftpack
from scipy._lib._version import NumpyVersion
import matplotlib.pyplot as plt

import numpy as np
import threading
import timeit
import sys
import math

_rfft_mt_safe = (NumpyVersion(np.__version__) >= '1.9.0.dev-e24486e')
_rfft_lock = threading.Lock()

# Some of the intermediate values I want
cwt_dat = False
ridge_lines = False
filtered = False
max_locs = False

def modified_cwt(data, wavelet, widths, alex={}):
    """
    Continuous wavelet transform.
    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter.
    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(width,length)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.
    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).
    Notes
    -----
    >>> length = min(10 * width[ii], len(data))
    >>> cwt[ii,:] = scipy.signal.convolve(data, wavelet(length,
    ...                                       width[ii]), mode='same')
    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(-1, 1, 200, endpoint=False)
    >>> sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
    >>> widths = np.arange(1, 31)
    >>> cwtmatr = signal.cwt(sig, signal.ricker, widths)
    >>> plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
    ...            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    >>> plt.show()
    """
    # Creat an empty image array
    output = np.zeros([len(widths), len(data)])

    # For each width value (of the wavelet)
    for index, width in enumerate(widths):
        # Generate the wavelet of the given length
        wavelet_data = wavelet(min(10 * width, len(data)), width) # Will make the wavelet upto "width" wide, or the width of the data, truncating if the data is shorter then the width
        # Add the line to CWT image
        output[index, :] = convolve(data, wavelet_data,
                                  mode='same', alex=alex)
    return output

def convolve(in1, in2, mode='full', method='auto', alex={}):
    """
    Convolve two N-dimensional arrays.
    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.

    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.

    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.

    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).  See Notes for more detail.
           .. version added:: 0.19.0

        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.

    Returns
    -------
    convolve : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.
    See Also
    --------
    numpy.polymul : performs polynomial multiplication (same operation, but
                    also accepts poly1d objects)
    choose_conv_method : chooses the fastest appropriate convolution method
    fftconvolve
    Notes
    -----
    By default, `convolve` and `correlate` use ``method='auto'``, which calls
    `choose_conv_method` to choose the fastest method using pre-computed
    values (`choose_conv_method` can also measure real-world timing with a
    keyword argument). Because `fftconvolve` relies on floating point numbers,
    there are certain constraints that may force `method=direct` (more detail
    in `choose_conv_method` docstring).
    Examples
    --------
    Smooth a square pulse using a Hann window:
    >>> from scipy import signal
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> win = signal.hann(50)
    >>> filtered = signal.convolve(sig, win, mode='same') / sum(win)
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('Original pulse')
    >>> ax_orig.margins(0, 0.1)
    >>> ax_win.plot(win)
    >>> ax_win.set_title('Filter impulse response')
    >>> ax_win.margins(0, 0.1)
    >>> ax_filt.plot(filtered)
    >>> ax_filt.set_title('Filtered signal')
    >>> ax_filt.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> fig.show()
    """
    # Making sure you have numpy arrays as inputs
    volume = np.asarray(in1)
    kernel = np.asarray(in2)

    if volume.ndim == kernel.ndim == 0:
        # Check if this is activated
        str_temp = 'volume * kernel'
        dic_convolution_methods[str_temp] = dic_convolution_methods.get(str_temp,0) + 1
        str_temp = alex.get('signal_name', '') + ' ' + str_temp
        dic_convolution_methods_breakdown[str_temp] = dic_convolution_methods_breakdown.get(str_temp,0) + 1
        return volume * kernel

    #
    if _inputs_swap_needed(mode, volume.shape, kernel.shape):
        # Convolution is commutative; order doesn't have any effect on output
        volume, kernel = kernel, volume

    #
    if method == 'auto':
        method = choose_conv_method(volume, kernel, mode=mode)

    #
    if method == 'fft':
        # Check if this is activated
        str_temp = 'fftconvolve()'
        dic_convolution_methods[str_temp] = dic_convolution_methods.get(str_temp,0) + 1
        str_temp = alex.get('signal_name', '') + ' ' + str_temp
        dic_convolution_methods_breakdown[str_temp] = dic_convolution_methods_breakdown.get(str_temp,0) + 1

        out = fftconvolve(volume, kernel, mode=mode)
        result_type = np.result_type(volume, kernel)
        if result_type.kind in {'u', 'i'}:
            out = np.around(out)
        return out.astype(result_type)

    # fastpath to faster numpy.convolve for 1d inputs when possible
    if _np_conv_ok(volume, kernel, mode):
        # Check if this is activated
        str_temp = 'np.convolve()'
        dic_convolution_methods[str_temp] = dic_convolution_methods.get(str_temp,0) + 1
        str_temp = alex.get('signal_name', '') + ' ' + str_temp
        dic_convolution_methods_breakdown[str_temp] = dic_convolution_methods_breakdown.get(str_temp,0) + 1

        return np.convolve(volume, kernel, mode)

    # Otherwise
    # Check if this is activated
    str_temp = 'np.correlate()'
    dic_convolution_methods[str_temp] = dic_convolution_methods.get(str_temp,0) + 1
    str_temp = alex.get('signal_name', '') + ' ' + str_temp
    dic_convolution_methods_breakdown[str_temp] = dic_convolution_methods_breakdown.get(str_temp,0) + 1

    return np.correlate(volume, _reverse_and_conj(kernel), mode, 'direct')


def fftconvolve(in1, in2, mode="full"):
    """Convolve two N-dimensional arrays using FFT.
    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.
    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).
    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`;
        if sizes of `in1` and `in2` are not equal then `in1` has to be the
        larger array.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.
    Examples
    --------
    Autocorrelation of white noise is an impulse.  (This is at least 100 times
    as fast as `convolve`.)
    >>> from scipy import signal
    >>> sig = np.random.randn(1000)
    >>> autocorr = signal.fftconvolve(sig, sig[::-1], mode='full')
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('White noise')
    >>> ax_mag.plot(np.arange(-len(sig)+1,len(sig)), autocorr)
    >>> ax_mag.set_title('Autocorrelation')
    >>> fig.tight_layout()
    >>> fig.show()
    Gaussian blur implemented using FFT convolution.  Notice the dark borders
    around the image, due to the zero-padding beyond its boundaries.
    The `convolve2d` function allows for other types of image boundaries,
    but is far slower.
    >>> from scipy import misc
    >>> lena = misc.lena()
    >>> kernel = np.outer(signal.gaussian(70, 8), signal.gaussian(70, 8))
    >>> blurred = signal.fftconvolve(lena, kernel, mode='same')
    >>> fig, (ax_orig, ax_kernel, ax_blurred) = plt.subplots(1, 3)
    >>> ax_orig.imshow(lena, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_kernel.imshow(kernel, cmap='gray')
    >>> ax_kernel.set_title('Gaussian kernel')
    >>> ax_kernel.set_axis_off()
    >>> ax_blurred.imshow(blurred, cmap='gray')
    >>> ax_blurred.set_title('Blurred')
    >>> ax_blurred.set_axis_off()
    >>> fig.show()
    """
    # Check if this function is activated
    #print('\nfftconvolve method used.\n')

    #
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))
    shape = s1 + s2 - 1

    if mode == "valid":
        _check_valid_mode_shapes(s1, s2)

    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [_next_regular(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
    # sure we only call np.fft.rfftn/np.fft.irfftn from one thread at a time.
    if not complex_result and (_rfft_mt_safe or _rfft_lock.acquire(False)):
        try:
            ret = np.fft.irfftn(np.fft.rfftn(in1, fshape) *
                         np.fft.rfftn(in2, fshape), fshape)[fslice].copy()
        finally:
            if not _rfft_mt_safe:
                _rfft_lock.release()
    else:
        # If we're here, it's either because we need a complex result, or we
        # failed to acquire _rfft_lock (meaning np.fft.rfftn isn't threadsafe and
        # is already in use by another thread).  In either case, use the
        # (threadsafe but slower) SciPy complex-FFT routines instead.
        ret = scipy.fftpack.ifftn(scipy.fftpack.fftn(in1, fshape) * scipy.fftpack.fftn(in2, fshape))[fslice].copy()
        if not complex_result:
            ret = ret.real

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def _check_valid_mode_shapes(shape1, shape2):
    for d1, d2 in zip(shape1, shape2):
        if not d1 >= d2:
            raise ValueError(
                "in1 should have at least as many items as in2 in "
                "every dimension for 'valid' mode.")

def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
    """
    reverse = [slice(None, None, -1)] * x.ndim
    return x[reverse].conj()

def _np_conv_ok(volume, kernel, mode):
    """
    See if numpy supports convolution of `volume` and `kernel` (i.e. both are
    1D ndarrays and of the appropriate shape).  Numpy's 'same' mode uses the
    size of the larger input, while Scipy's uses the size of the first input.
    """
    np_conv_ok = volume.ndim == kernel.ndim == 1
    return np_conv_ok and (volume.size >= kernel.size or mode != 'same')

def _inputs_swap_needed(mode, shape1, shape2):
    """
    If in 'valid' mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every dimension.
    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.
    Note that if the mode provided is not 'valid', False is immediately
    returned.
    """
    if mode == 'valid':
        ok1, ok2 = True, True

        for d1, d2 in zip(shape1, shape2):
            if not d1 >= d2:
                ok1 = False
            if not d2 >= d1:
                ok2 = False

        if not (ok1 or ok2):
            raise ValueError("For 'valid' mode, one must be at least "
                             "as large as the other in every dimension")

        return not ok1

    return False

def choose_conv_method(in1, in2, mode='full', measure=False):
    """
    Find the fastest convolution/correlation method.
    This primarily exists to be called during the ``method='auto'`` option in
    `convolve` and `correlate`, but can also be used when performing many
    convolutions of the same input shapes and dtypes, determining
    which method to use for all of them, either to avoid the overhead of the
    'auto' option or to use accurate real-world measurements.
    Parameters
    ----------
    in1 : array_like
        The first argument passed into the convolution function.
    in2 : array_like
        The second argument passed into the convolution function.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    measure : bool, optional
        If True, run and time the convolution of `in1` and `in2` with both
        methods and return the fastest. If False (default), predict the fastest
        method using precomputed values.
    Returns
    -------
    method : str
        A string indicating which convolution method is fastest, either
        'direct' or 'fft'
    times : dict, optional
        A dictionary containing the times (in seconds) needed for each method.
        This value is only returned if ``measure=True``.
    See Also
    --------
    convolve
    correlate
    Notes
    -----
    For large n, ``measure=False`` is accurate and can quickly determine the
    fastest method to perform the convolution.  However, this is not as
    accurate for small n (when any dimension in the input or output is small).
    In practice, we found that this function estimates the faster method up to
    a multiplicative factor of 5 (i.e., the estimated method is *at most* 5
    times slower than the fastest method). The estimation values were tuned on
    an early 2015 MacBook Pro with 8GB RAM but we found that the prediction
    held *fairly* accurately across different machines.
    If ``measure=True``, time the convolutions. Because this function uses
    `fftconvolve`, an error will be thrown if it does not support the inputs.
    There are cases when `fftconvolve` supports the inputs but this function
    returns `direct` (e.g., to protect against floating point integer
    precision).
    .. versionadded:: 0.19
    Examples
    --------
    Estimate the fastest method for a given input:
    >>> from scipy import signal
    >>> a = np.random.randn(1000)
    >>> b = np.random.randn(1000000)
    >>> method = signal.choose_conv_method(a, b, mode='same')
    >>> method
    'fft'
    This can then be applied to other arrays of the same dtype and shape:
    >>> c = np.random.randn(1000)
    >>> d = np.random.randn(1000000)
    >>> # `method` works with correlate and convolve
    >>> corr1 = signal.correlate(a, b, mode='same', method=method)
    >>> corr2 = signal.correlate(c, d, mode='same', method=method)
    >>> conv1 = signal.convolve(a, b, mode='same', method=method)
    >>> conv2 = signal.convolve(c, d, mode='same', method=method)
    """
    volume = np.asarray(in1)
    kernel = np.asarray(in2)

    if measure:
        times = {}
        for method in ['fft', 'direct']:
            times[method] = _timeit_fast(lambda: convolve(volume, kernel,
                                         mode=mode, method=method))

        chosen_method = 'fft' if times['fft'] < times['direct'] else 'direct'
        return chosen_method, times

    # fftconvolve doesn't support complex256
    fftconv_unsup = "complex256" if sys.maxsize > 2**32 else "complex192"
    if hasattr(np, fftconv_unsup):
        if volume.dtype == fftconv_unsup or kernel.dtype == fftconv_unsup:
            return 'direct'

    # for integer input,
    # catch when more precision required than float provides (representing an
    # integer as float can lose precision in fftconvolve if larger than 2**52)
    if any([_numeric_arrays([x], kinds='ui') for x in [volume, kernel]]):
        max_value = int(np.abs(volume).max()) * int(np.abs(kernel).max())
        max_value *= int(min(volume.size, kernel.size))
        if max_value > 2**np.finfo('float').nmant - 1:
            return 'direct'

    if _numeric_arrays([volume, kernel], kinds='b'):
        return 'direct'

    if _numeric_arrays([volume, kernel]):
        if _fftconv_faster(volume, kernel, mode):
            return 'fft'
    return 'direct'

def _fftconv_faster(x, h, mode):
    """
    See if using `fftconvolve` or `_correlateND` is faster. The boolean value
    returned depends on the sizes and shapes of the input values.
    The big O ratios were found to hold across different machines, which makes
    sense as it's the ratio that matters (the effective speed of the computer
    is found in both big O constants). Regardless, this had been tuned on an
    early 2015 MacBook Pro with 8GB RAM and an Intel i5 processor.
    """
    if mode == 'full':
        out_shape = [n + k - 1 for n, k in zip(x.shape, h.shape)]
        big_O_constant = 10963.92823819 if x.ndim == 1 else 8899.1104874
    elif mode == 'same':
        out_shape = x.shape
        if x.ndim == 1:
            if h.size <= x.size:
                big_O_constant = 7183.41306773
            else:
                big_O_constant = 856.78174111
        else:
            big_O_constant = 34519.21021589
    elif mode == 'valid':
        out_shape = [n - k + 1 for n, k in zip(x.shape, h.shape)]
        big_O_constant = 41954.28006344 if x.ndim == 1 else 66453.24316434
    else:
        raise ValueError('mode is invalid')

    # see whether the Fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    direct_time = (x.size * h.size * _prod(out_shape))
    fft_time = sum(n * math.log(n) for n in (x.shape + h.shape +
                                             tuple(out_shape)))
    return big_O_constant * fft_time < direct_time

def _prod(iterable):
    """
    Product of a list of numbers.
    Faster than np.prod for short lists like array shapes.
    """
    product = 1
    for x in iterable:
        product *= x
    return product

def _numeric_arrays(arrays, kinds='buifc'):
    """
    See if a list of arrays are all numeric.
    Parameters
    ----------
    ndarrays : array or list of arrays
        arrays to check if numeric.
    numeric_kinds : string-like
        The dtypes of the arrays to be checked. If the dtype.kind of
        the ndarrays are not in this string the function returns False and
        otherwise returns True.
    """
    if type(arrays) == np.ndarray:
        return arrays.dtype.kind in kinds
    for array_ in arrays:
        if array_.dtype.kind not in kinds:
            return False
    return True

def _timeit_fast(stmt="pass", setup="pass", repeat=3):
    """
    Returns the time the statement/function took, in seconds.
    Faster, less precise version of IPython's timeit. `stmt` can be a statement
    written as a string or a callable.
    Will do only 1 loop (like IPython's timeit) with no repetitions
    (unlike IPython) for very slow functions.  For fast functions, only does
    enough loops to take 5 ms, which seems to produce similar results (on
    Windows at least), and avoids doing an extraneous cycle that isn't
    measured.
    """
    timer = timeit.Timer(stmt, setup)

    # determine number of calls per rep so total time for 1 rep >= 5 ms
    x = 0
    for p in range(0, 10):
        number = 10**p
        x = timer.timeit(number)  # seconds
        if x >= 5e-3 / 10:  # 5 ms for final test, 1/10th that for this one
            break
    if x > 1:  # second
        # If it's macroscopic, don't bother with repetitions
        best = x
    else:
        number *= 10
        r = timer.repeat(repeat, number)
        best = min(r)

    sec = best / number
    return sec

def _boolrelextrema(data, comparator,
                  axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.
    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.
    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take 2 numbers as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.  'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default 'clip'.  See numpy.take
    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.
    See also
    --------
    argrelmax, argrelmin
    Examples
    --------
    >>> testdata = np.array([1,2,3,2,1])
    >>> _boolrelextrema(testdata, np.greater, axis=0)
    array([False, False,  True, False, False], dtype=bool)
    """
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    #for shift in xrange(1, order + 1):
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if(~results.any()):
            return results
    return results

def _identify_ridge_lines(matr, max_distances, gap_thresh):
    """
    Identify ridges in the 2-D matrix.
    Expect that the width of the wavelet feature increases with increasing row
    number.
    Parameters
    ----------
    matr : 2-D ndarray
        Matrix in which to identify ridge lines.
    max_distances : 1-D sequence
        At each row, a ridge line is only connected
        if the relative max at row[n] is within
        `max_distances`[n] from the relative max at row[n+1].
    gap_thresh : int
        If a relative maximum is not found within `max_distances`,
        there will be a gap. A ridge line is discontinued if
        there are more than `gap_thresh` points without connecting
        a new relative maximum.
    Returns
    -------
    ridge_lines : tuple
        Tuple of 2 1-D sequences. `ridge_lines`[ii][0] are the rows of the ii-th
        ridge-line, `ridge_lines`[ii][1] are the columns. Empty if none found.
        Each ridge-line will be sorted by row (increasing), but the order
        of the ridge lines is not specified.
    References
    ----------
    Bioinformatics (2006) 22 (17): 2059-2065.
    doi: 10.1093/bioinformatics/btl355
    http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
    Examples
    --------
    >>> data = np.random.rand(5,5)
    >>> ridge_lines = _identify_ridge_lines(data, 1, 1)
    Notes
    -----
    This function is intended to be used in conjunction with `cwt`
    as part of `find_peaks_cwt`.
    """
    if(len(max_distances) < matr.shape[0]):
        raise ValueError('Max_distances must have at least as many rows as matr')

    all_max_cols = _boolrelextrema(matr, np.greater, axis=1, order=1)
    #Highest row for which there are any relative maxima
    has_relmax = np.where(all_max_cols.any(axis=1))[0]
    if(len(has_relmax) == 0):
        return []
    start_row = has_relmax[-1]
    #Each ridge line is a 3-tuple:
    #rows, cols,Gap number
    ridge_lines = [[[start_row],
                   [col],
                   0] for col in np.where(all_max_cols[start_row])[0]]
    final_lines = []
    rows = np.arange(start_row - 1, -1, -1)
    cols = np.arange(0, matr.shape[1])
    for row in rows:
        this_max_cols = cols[all_max_cols[row]]

        #Increment gap number of each line,
        #set it to zero later if appropriate
        for line in ridge_lines:
            line[2] += 1

        #XXX These should always be all_max_cols[row]
        #But the order might be different. Might be an efficiency gain
        #to make sure the order is the same and avoid this iteration
        prev_ridge_cols = np.array([line[1][-1] for line in ridge_lines])
        #Look through every relative maximum found at current row
        #Attempt to connect them with existing ridge lines.
        for ind, col in enumerate(this_max_cols):
            """
            If there is a previous ridge line within
            the max_distance to connect to, do so.
            Otherwise start a new one.
            """
            line = None
            if(len(prev_ridge_cols) > 0):
                diffs = np.abs(col - prev_ridge_cols)
                closest = np.argmin(diffs)
                if diffs[closest] <= max_distances[row]:
                    line = ridge_lines[closest]
            if(line is not None):
                #Found a point close enough, extend current ridge line
                line[1].append(col)
                line[0].append(row)
                line[2] = 0
            else:
                new_line = [[row],
                            [col],
                            0]
                ridge_lines.append(new_line)

        #Remove the ridge lines with gap_number too high
        #XXX Modifying a list while iterating over it.
        #Should be safe, since we iterate backwards, but
        #still tacky.
        #for ind in xrange(len(ridge_lines) - 1, -1, -1):
        for ind in range(len(ridge_lines) - 1, -1, -1):
            line = ridge_lines[ind]
            if line[2] > gap_thresh:
                final_lines.append(line)
                del ridge_lines[ind]

    out_lines = []
    for line in (final_lines + ridge_lines):
        sortargs = np.array(np.argsort(line[0]))
        rows, cols = np.zeros_like(sortargs), np.zeros_like(sortargs)
        rows[sortargs] = line[0]
        cols[sortargs] = line[1]
        out_lines.append([rows, cols])

    return out_lines

def _filter_ridge_lines(cwt, ridge_lines, window_size=None, min_length=None,
                       min_snr=1, noise_perc=10):
    """
    Filter ridge lines according to prescribed criteria. Intended
    to be used for finding relative maxima.

    Stored as a list of ridges lists, each of these lists has 2 arrayes
    corrisponding to the y and x axes coordinates of the points making the ridge
    line.
    The coordinates for each ridgeline are in ascending value of y, so the first
    entry for a given ridge-line will be the foot of that ridge-line.

    filtered =
    [
      [  # First ridge
         array_y_coords, array_x_coords
      ],
      [  # Second ridge
         array_y_coords, array_x_coords
      ],
      [  # Third ridge
         array_y_coords, array_x_coords
      ]
    ]

    So to access the i-th rdige line:
    line_y = filtered[i][0]
    line_x = filtered[i][1]

    Note that the ridge-lines arn't listed in x-location order, so you may get
    ridge-lines starting at similar locations far appart in this list.

    Parameters
    ----------
    cwt : 2-D ndarray
        Continuous wavelet transform from which the `ridge_lines` were defined.
    ridge_lines : 1-D sequence
        Each element should contain 2 sequences, the rows and columns
        of the ridge line (respectively).
    window_size : int, optional
        Size of window to use to calculate noise floor.
        Default is ``cwt.shape[1] / 20``.
    min_length : int, optional
        Minimum length a ridge line needs to be acceptable.
        Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.
    min_snr : float, optional
        Minimum SNR ratio. Default 1. The signal is the value of
        the cwt matrix at the shortest length scale (``cwt[0, loc]``), the
        noise is the `noise_perc`th percentile of datapoints contained within a
        window of `window_size` around ``cwt[0, loc]``.
    noise_perc : float, optional
        When calculating the noise floor, percentile of data points
        examined below which to consider noise. Calculated using
        scipy.stats.scoreatpercentile.
    References
    ----------
    Bioinformatics (2006) 22 (17): 2059-2065. doi: 10.1093/bioinformatics/btl355
    http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
    """
    # Sanitise input parameters
    num_points = cwt.shape[1]
    if min_length is None:
        # Min length a quarter of the height of the CWT, rounded up to an integer.
        min_length = np.ceil(cwt.shape[0] / 4)
    if window_size is None:
        # Window is a quarter of the width (in datapoints), rounded up to an integer.
        window_size = np.ceil(num_points / 20)
    # The falf window size
    hf_window = window_size / 2

    #Filter based on SNR
    row_one = cwt[0, :]
    noises = np.zeros_like(row_one)
    for index, val in enumerate(row_one):
        window = np.arange(max([index - hf_window, 0]), min([index + hf_window, num_points]))
        window = window.astype(int)
        noises[index] = scoreatpercentile(row_one[window], per=noise_perc)

    #
    def filt_func(line):
        # Check if the line has enough entries
        if len(line[0]) < min_length:
            return False
        #
        snr = abs(cwt[line[0][0], line[1][0]] / noises[line[1][0]])
        if snr < min_snr:
            return False
        return True

    return list(filter(filt_func, ridge_lines))



def modified_find_peaks_cwt(vector, widths, wavelet=None, max_distances=None, gap_thresh=None,
                   min_length=None, min_snr=1, noise_perc=10, alex={}):
    """
    Attempt to find the peaks in a 1-D array.
    The general approach is to smooth `vector` by convolving it with
    `wavelet(width)` for each width in `widths`. Relative maxima which
    appear at enough length scales, and with sufficiently high SNR, are
    accepted.
    .. versionadded:: 0.11.0
    Parameters
    ----------
    vector : ndarray
        1-D array in which to find the peaks.
    widths : sequence
        1-D array of widths to use for calculating the CWT matrix. In general,
        this range should cover the expected width of peaks of interest.
    wavelet : callable, optional
        Should take a single variable and return a 1-D array to convolve
        with `vector`.  Should be normalized to unit area.
        Default is the ricker wavelet.
    max_distances : ndarray, optional
        At each row, a ridge line is only connected if the relative max at
        row[n] is within ``max_distances[n]`` from the relative max at
        ``row[n+1]``.  Default value is ``widths/4``.
    gap_thresh : float, optional
        If a relative maximum is not found within `max_distances`,
        there will be a gap. A ridge line is discontinued if there are more
        than `gap_thresh` points without connecting a new relative maximum.
        Default is 2.
    min_length : int, optional
        Minimum length a ridge line needs to be acceptable.
        Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.
    min_snr : float, optional
        Minimum SNR ratio. Default 1. The signal is the value of
        the cwt matrix at the shortest length scale (``cwt[0, loc]``), the
        noise is the `noise_perc`th percentile of datapoints contained within a
        window of `window_size` around ``cwt[0, loc]``.
    noise_perc : float, optional
        When calculating the noise floor, percentile of data points
        examined below which to consider noise. Calculated using
        `stats.scoreatpercentile`.  Default is 10.

    See Also
    --------
    cwt
    Notes
    -----
    This approach was designed for finding sharp peaks among noisy data,
    however with proper parameter selection it should function well for
    different peak shapes.
    The algorithm is as follows:
     1. Perform a continuous wavelet transform on `vector`, for the supplied
        `widths`. This is a convolution of `vector` with `wavelet(width)` for
        each width in `widths`. See `cwt`
     2. Identify "ridge lines" in the cwt matrix. These are relative maxima
        at each row, connected across adjacent rows. See identify_ridge_lines
     3. Filter the ridge_lines using filter_ridge_lines.
    References
    ----------
    .. [1] Bioinformatics (2006) 22 (17): 2059-2065.
        doi: 10.1093/bioinformatics/btl355
        http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
    Examples
    --------
    >>> from scipy import signal
    >>> xs = np.arange(0, np.pi, 0.05)
    >>> data = np.sin(xs)
    >>> peakind = signal.find_peaks_cwt(data, np.arange(1,10))
    >>> peakind, xs[peakind], data[peakind]
    ([32], array([ 1.6]), array([ 0.9995736]))
    """
    # If no gap_thresh is defined then use the maximum width
    if gap_thresh is None:
        gap_thresh = np.ceil(widths[0])
    # If no max_distance is defined then...
    if max_distances is None:
        max_distances = widths / 4.0
    # Set the default wavelet to the ricker wavelet
    if wavelet is None:
        wavelet = ricker

    #
    cwt_dat = modified_cwt(vector, wavelet, widths, alex=alex)

    #
    ridge_lines = _identify_ridge_lines(cwt_dat, max_distances, gap_thresh)

    # The ridge lines above a given length/height (min_length)
    filtered = _filter_ridge_lines(cwt_dat, ridge_lines, min_length=min_length,
                                   min_snr=min_snr, noise_perc=noise_perc)

    # The first/foot x-locations for each of the ridge-lines
    # This is the list of detected peaks, not in order.
    max_locs = [x[1][0] for x in filtered] #
    return sorted(max_locs), cwt_dat, ridge_lines, filtered


def modified_get_flare_peaks_cwt(ser_data, widths=np.arange(1,100), raw_data=None, ser_minima=None, get_duration=True, get_energies=True, alex={}):
    """
    Implment SciPy CWT to find peaks in the given data.
    Note: input data is expected to be pre-processed (generally resampled and averaged).

    Parameters
    ----------
    ser_data: ~`pandas.Series`
        The dataset to look for flare peaks in.
        Generally GOES XRS B-channel X-Ray data.
        Should be pre-processed to exclude data spikes, data gaps (interpolate),
        rebinned to a common bin width and ideally averaged.

    raw_data: ~`pandas.Series`
        The raw dataset, used for getting the intensity at the time of each peak
        (and thus the flare classification).
        This is generally the un-averaged data, because that'll tend to give
        results closer to the HEK listings.
        Note: defaults to using the ser_data if no raw_data is given.

    widths : (M,) sequence
        The widths to check within the CWT routine.
        See `scipy.signal.cwt` for mor details.

    get_duration : `bool`
        When True the start and end times will be found and then the duraction
        calculated.
        The current implmentation finds the local minima before and after the peak
        and deems these the start and end.
        This must be true if you calculate the energy via numerical integration.

    get_energies : `bool`
        When True use the start/end times and data to use numerical integration
        to interpret the energy detected at the detector.

    Returns
    -------
    result: ~`pandas.DataFrame`
        The table of results, ordered/indexed by peak time.
    """

    # Make the data a pandas.Series if it isn't already
    ser_raw_data = raw_data
    if not isinstance(raw_data, pd.Series):
        ser_raw_data = ser_data

    # Get the peaks
    ###arr_peak_time_indices = signal.find_peaks_cwt(ser_data.values, widths)
    arr_peak_time_indices, cwt_dat, ridge_lines, filtered = modified_find_peaks_cwt(ser_data.values, widths, alex=alex)
    ser_cwt_peaks = ser_raw_data[arr_peak_time_indices]

    # As a dataframe
    pd_peaks_cwt = pd.DataFrame(data={'fl_peakflux': ser_cwt_peaks})
    pd_peaks_cwt['fl_goescls'] = utils.arr_to_cla(pd_peaks_cwt['fl_peakflux'].values, int_dp=1)
    pd_peaks_cwt['event_peaktime'] = pd_peaks_cwt.index
    pd_peaks_cwt['i_index'] = arr_peak_time_indices

    # Assuming we want the time/energy details
    if get_duration:
        # Get local minima if not given.
        if ser_minima == None:
            ser_minima = ser_data[utils.find_minima_fast(ser_data.interpolate().values)]

        # Now get the star/end time details and add to the DataFrame
        pd_durations = utils.get_flare_start_end_using_min_min(ser_data=ser_data, ser_minima=ser_minima, ser_peaks=ser_cwt_peaks)
        # Add to the original DataFrame
        pd_peaks_cwt = pd.concat([pd_peaks_cwt, pd_durations], axis=1)
        """
        print('\n')
        print(pd_peaks_cwt)
        print('\n')
        """
        # Now get the energies if requested.
        if get_energies:
            pd_energies = utils.get_flare_energy_trap_inte(ser_data, pd_durations['event_starttime'], pd_durations['event_endtime'], pd_peaks_cwt.index)
            # Add to the original DataFrame
            pd_peaks_cwt = pd.concat([pd_peaks_cwt, pd_energies], axis=1)

    # Return the results
    return pd_peaks_cwt, cwt_dat, ridge_lines, filtered


# Basic imports
import pandas as pd
import numpy as np
from datetime import timedelta
import os.path
import datetime

# Advanced imports
import flarepy.plotting as plotting
import flarepy.utils as utils
import flarepy.flare_detection as det
from sunpy.lightcurve import GOESLightCurve
from sunpy.time import TimeRange

# Parameters
# Specify the start/end times
str_start = '2014-03-29 00:00:00' # Only necessary because only DL GOES for single days
str_end = '2014-03-30 00:00:00'

lc_goes_29th = GOESLightCurve.create(TimeRange(str_start, str_end))
df_goes_XRS = lc_goes_29th.data #pd.concat([lc_goes_5th.data, lc_goes_6th.data])




# Parameters
int_max_width = 50
arr_cwt_widths = np.arange(1,int_max_width)
int_datapoints = 1000
str_function = 'arbitrary function'

dic_datasets = {}

# Dataset - Several Examples given, comment out unwanted
dic_datasets['sinusoid'] = np.sin(np.arange(int_datapoints) * np.pi / 30. )
dic_datasets['ricker, a=50'] = signal.ricker(int_datapoints, 50)
dic_datasets['morlet, a=10'] = signal.morlet(int_datapoints, 10)
dic_datasets['exponential'] = signal.exponential(int_datapoints)
dic_datasets['gaussian, std=50'] = signal.gaussian(int_datapoints, std=50)

dic_convolution_methods = {}
dic_convolution_methods_breakdown = {}

for str_function, arr_data in dic_datasets.items():
    dic_details = { 'signal_name': str_function}

    # Get the CWT peaks
    arr_peaks, cwt_dat, ridge_lines, filtered = modified_find_peaks_cwt(arr_data, arr_cwt_widths, alex=dic_details)


    """
        ridge_lines : tuple
            Tuple of 2 1-D sequences. `ridge_lines`[ii][0] are the rows of the ii-th
            ridge-line, `ridge_lines`[ii][1] are the columns. Empty if none found.
            Each ridge-line will be sorted by row (increasing), but the order
            of the ridge lines is not specified.
    """
    # Four subplots, the axes array is 1-d
    fig, axarr = plt.subplots(4, sharex=True)

    # Linear data plots
    # Plot the data line in the top:
    x_line = np.arange(len(arr_data))
    y_line = arr_data
    plt_line = axarr[0].plot(x_line, y_line, color='blue', marker='None', linestyle='-')

    # Plot the peaks:
    x_peaks = arr_peaks
    y_peaks = arr_data [arr_peaks]
    plt_peaks = axarr[0].plot(x_peaks, y_peaks, color='green', marker='x', linestyle='None', markersize=5)

    # Add a title to the whole figure
    axarr[0].set_title('CWT Peak Detection Steps - '+str_function)

    # Log plots
    # Plot the logged data line:
    plt_line_log = axarr[1].plot(x_line, y_line, color='blue', marker='None', linestyle='-')
    # Plot the logged peaks:
    plt_peaks_log = axarr[1].plot(x_peaks, y_peaks, color='green', marker='x', linestyle='None', markersize=5)
    # Set the scale to logged:
    axarr[1].set_yscale("log")

    # Plot the CWT image
    plt_img = axarr[2].imshow(cwt_dat, origin='lower')#, extent=[x_line[0],x_line[-1],0,100])
    # Trying to strecth vertically, but I can't figure it out:
    #axarr[2].set_ylim((0,100))
    #axarr[2].set_ymargin(0)

    # Plotting the ridge plot
    # Adding all ridge points
    x_all = []
    y_all = []
    for i, ridge_line in enumerate(ridge_lines):
        #print('i: '+str(i))
        for j in range(len(ridge_line[0])):
            #print('    j: '+str(j))
            y_all.append(ridge_lines[i][0][j])
            x_all.append(ridge_lines[i][1][j])

    # Adding the filtered ridge points, those associated with a peak detection
    x_filtered = []
    y_filtered = []
    for i, ridge_line in enumerate(filtered):
        #print('i: '+str(i))
        for j in range(len(ridge_line[0])):
            #print('    j: '+str(j))
            y_filtered.append(filtered[i][0][j])
            x_filtered.append(filtered[i][1][j])

    # Adding these values to the lowest plot:
    axarr[3].plot(x_all, y_all, color='k', marker='.', linestyle='None', markersize=1)
    axarr[3].plot(x_filtered, y_filtered, color='blue', marker='.', linestyle='None', markersize=1)

    # Save teh figure for viewing
    fig.savefig('CWT Peak Detection Steps - '+str_function+'.png', dpi=900, bbox_inches='tight')


