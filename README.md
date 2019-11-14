# Imaging Spectroscopy Workbench

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
# From inside the ISWorkbench directory
virtualenv venv
source venv/bin/activate
```

Then, you can easily install all required dependencies:

```
pip install numpy
pip install astropy
pip install matplotlib
pip install GDAL
pip install PySide2
```

