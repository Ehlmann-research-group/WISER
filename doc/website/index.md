# WISER:  The Workbench for Imaging Spectroscopy Exploration and Research

WISER is a tool for visualizing and analyzing imaging spectroscopy data.  It is
implemented in Python 3, and makes use of a number of libraries used widely in
scientific computing.

## Distributables

This software is currently pre-release, and should probably only be installed
by the intrepid.

The software is distributed as a frozen Python application (using
[PyInstaller](https://www.pyinstaller.org/)), and is currently available in
these forms:

*   A MacOSX disk image (DMG) is available here:
    [WISER-1.0a3.dmg](./WISER-1.0a3.dmg)

    The application can be run directly out of the disk image if you just want
    to try it out, or you can drag the application into your Applications folder
    for faster startup times.

    **Known Issues:**  On MacOS Catalina, the Gatekeeper security system really
    interferes with running software that is not from the App Store, or that is
    not signed with a key from Apple ($99/yr).  [This documentation about
    Gatekeeper](https://blog.macsales.com/57866-how-to-work-with-and-around-gatekeeper/)
    may be of some help in working around these issues.

*   A Windows installer is available here:
    [Install-WISER-1.0a3.exe](./Install-WISER-1.0a3.exe)

    The Windows installer will install the software into your system's
    applications, along with an uninstaller for removing the software.  You can
    run the application (or the uninstaller) through the Start Menu.  You can
    also uninstall the software through the "apps" functionality in the
    System Console.

    **Known Issues:**  The Windows version of the software has a few known
    issues that simply haven't been addressed yet.  They are listed here:

    *   The frozen version of the software on Windows is more than 800MiB.  This
        seems excessive, and subsequent efforts will try to bring this down to a
        much more reasonable value.

    *   When the application runs on Windows, an empty console window pops up.
        This is because the PyInstaller-frozen app currently only works when
        built as a console application, but not as a windowed application.
        This issue will hopefully be addressed in the near future.

## Development

When we get around to it, information will be added here for how to set up a
development environment.

## Thanks

The Workbench is built upon these libraries:

*   [GDAL](https://gdal.org/) for loading and saving raster data
*   [NumPy](https://numpy.org/) and [AstroPy](https://www.astropy.org/) for
    internal calculations and data representation
*   [Qt 5](https://www.qt.io/) and the
    [PySide2 Python bindings](https://wiki.qt.io/Qt_for_Python) for the
    Graphical User Interface
*   [matplotlib](https://matplotlib.org/) for drawing spectral plots and other
    graphs

Distributable packages are built with
[PyInstaller](https://www.pyinstaller.org/).
The Windows version is built on top of
[Miniconda](https://docs.conda.io/en/latest/miniconda.html), and the installer
is generated with [NSIS](https://nsis.sourceforge.io/).

[<img src="./bugsnag.png"> BugSnag](https://www.bugsnag.com/) is used for online
reporting of crashes and other severe errors, if users choose to opt in.
