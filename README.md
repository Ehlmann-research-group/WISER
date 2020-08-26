# WISER:  The Workbench for Imaging Spectroscopy Exploration and Research

WISER is an open-source, extensible tool for visualizing and analyzing spectral
imaging data that may include many different frequency bands.  It is written in
Python, and leverages the [Qt 5](https://www.qt.io/) platform with the
[PySide2](https://wiki.qt.io/Qt_for_Python) Python bindings to provide a
cross-platform GUI.  [GDAL](https://gdal.org/) is used to load and save spectral
data, so that many different data formats may be supported.  Internally, the
software uses [NumPy](https://numpy.org) for representing and manipulating
spectral data.

Currently, WISER is supported on MacOSX and Windows platforms.  Distributable
packages or installers are provided for these platforms.

Linux support will be added in the near future.

## Development Environment

If you are interested in setting up a development environment on Windows or
MacOSX, please see these documents:

*   [MacOSX Environment Setup](./doc/devenv-mac.md)
*   [Windows Environment Setup](./doc/devenv-win.md)
