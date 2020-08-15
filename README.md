# WISER:  The Workbench for Imaging Spectroscopy Exploration and Research

Tools for visualizing and working with imaging spectroscopy data.

## Platform Requirements

This software currently requires Python 3.7.  There are some compatibility
issues between Qt5/PySide2 and Python 3.8 that have not yet been sorted out.

This software requires a few external tools and libraries be installed on the
local environment.  These include:

*   [GDAL](https://gdal.org)
*   Qt 5

If you are developing on a Mac, these are easily installed via MacPorts:

```
sudo port install GDAL
sudo port install qt5
```

## Python Setup

If you are working on a Mac, you are strongly encouraged to set up a virtual
environment using the `virtualenv` tool:

```
# From inside the WISER directory
virtualenv venv
source venv/bin/activate
```

Then, you can easily install all required dependencies **in this order**:

```
pip install numpy
pip install astropy
pip install matplotlib
pip install GDAL
pip install PySide2
```

>   Note:  If you get an error like "no module \_gdal_array" after setting up
>   WISER, a common cause is that the GDAL Python library was installed _before_
>   NumPy was installed.  To further complicate the matter, `pip` may have
>   cached the GDAL library without the `_gdal_array` module.  Thus, to fix the
>   issue, you can do this:
>
>       pip uninstall GDAL
>       pip install numpy                  (if not already installed)
>       pip install --no-cache-dir GDAL

These are additional development tools used by the project:

```
pip install pylint
pip install mypy
```

Finally, to build the Mac installer, you need the py2app library:

```
pip install py2app
```

## Generated Sources

Some of the Qt user interfaces are built with the Qt Designer.  To generate
the corresponding Python code, use these commands:

```
cd src/gui
make
```

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
