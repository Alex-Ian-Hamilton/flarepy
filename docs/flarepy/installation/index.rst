============
Installation
============

FlarePy is a Python package for solar flare detection and analysis, it 
leverages the tools provided by a number of other open source Python
libraries, most notably SunPy which is based on AstroPy.
These and other packages will also need to be installed in order to run
FlarePy.

Installing Scientific Python and SunPy
--------------------------------------

It is highly recommended that you run FlarePy on Continuum Anaconda Python,
this is a scientific distribution of Python with it's own package and
environment manager called Conda.
You can get Anacond `here <https://www.continuum.io/downloads>`_.
, 
You can read how to install SunPy in their documentation `here <http://docs.sunpy.org/en/stable/guide/installation/>`_.

Git Clone and Install FlarePy
-----------------------------

FlarePy is a GitHub repository, you can download it by cloning into a local folder.

For example::

    git clone https://github.com/Alex-Ian-Hamilton/flarepy.git flarepy

This will create a 'flarepy' folder in the current folder with the FlarePy
code.
Note: You will be asked for you GitHub login credentials, if you haven't already
registered then you will need to do so `here <https://github.com/>`_.

To setup you then move to the 'flarpy' folder and setup, using::

    cd flarepy
    python setup.py develop


Updating FlarePy to a New Version
#################################

FlarePy is in constant development, I add new code on a daily basis, to get the
latest version you can use git to pull the latest version by opening the FlarePy
folder and running::

    git pull
