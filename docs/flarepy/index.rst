*********************
flarepy Documentation
*********************

This is the documentation for flarepy.
This is where I store all my code for the detection, evaluation and plotting for flare detections.
Primarily utilises GOES XRS X-ray timeseries data ATM.

Installation
=============

.. toctree::
    :maxdepth: 1

    installation/index.rst

Reference/API
=============

Flare Detection Routines
-----------------

For flare detection we have the following functions:

.. automodapi:: flarepy.flare_detection


Flare Plotting Tools
-----------------

Plotting routines for visually representing the data.

.. automodapi:: flarepy.plotting


Synthetic Data Generation Tools
-----------------

Tools for generating synthetic data for use in evaluating detection routines.

.. automodapi:: flarepy.synthetic_data_generation


Utils
-----------------

Right now this is where I store all my useful periferal code, from the ability to download HEK/GOES data (using SunPy) to the plotting algorithms.

Pre-Processing Utils
~~~~~~~~~~~~~~~~~~~~~~
Tools for simplifying the pre-processing of input data.

.. automodapi:: flarepy.utils.pre_processing


HEK Utils
~~~~~~~~~~~~~~~~~~~~~~
Tools to download and sanitise HEK/GOES XRS data (using SunPy).

.. automodapi:: flarepy.utils.hek_utils

Flare Utils
~~~~~~~~~~~~~~~~~~~~~~

Tools for manipulation of flare data, for example conversion from/to GOES flare classification.

.. automodapi:: flarepy.utils.flare_utils

Flare Start/End Utils
~~~~~~~~~~~~~~~~~~~~~~

Tools for finding the start and end times of flares.
Note: most detection routines only find the peaks and so methods to find the start/end can be used interchangably.

.. automodapi:: flarepy.utils.flare_start_end_utils


Flare Energy Utils
~~~~~~~~~~~~~~~~~~~~~~

Tools for finding the energy of of flares.

.. automodapi:: flarepy.utils.flare_energy_utils

Comparison Utils
~~~~~~~~~~~~~~~~~~~~~~

Tools for comparing flare results.

.. automodapi:: flarepy.utils.comparison_utils

Peak Picking Utils
~~~~~~~~~~~~~~~~~~~~~~

Tools for finding local maxima and minima in datasets.

.. automodapi:: flarepy.utils.peak_picking_utils


Pandas Utils
~~~~~~~~~~~~~~~~~~~~~~

Tools for use with pandas objects.

.. automodapi:: flarepy.utils.pandas_utils

