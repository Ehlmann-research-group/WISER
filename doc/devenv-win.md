## Windows 10 Dev Environment Setup

[Anaconda](https://www.anaconda.com/) seems to be a widely used Python package
and library manager for Windows and scientific computing.

Unfortunately, the full Anaconda3 installation doesn't seem to work with the
required tools and libraries, probably because of library/DLL versioning issues.
(As of 2020-06-04, Donnie has not spent the time to track down the causes of
these issues.)

The [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html) seems
to work though, and simply requires a bit more setup.

1.  [Install Miniconda3](https://docs.conda.io/en/latest/miniconda.html) for
    all users.

2.  Start an Anaconda Prompt in admin mode, to install packages

    Start -> Anaconda3 -> Anaconda Prompt [Right-Click] -> More -> Run as Administrator

3.  `conda config --add channels conda-forge`

    (Note:  Says that conda-forge is already present, investigate.)

4.  `conda install pyside2`

5.  `conda install gdal`

6.  `conda install matplotlib`

    (Note:  causes pyqt to be installed - how to prevent this?)

7.  `conda install astropy`

This should be sufficient to get WISER running on Windows.
