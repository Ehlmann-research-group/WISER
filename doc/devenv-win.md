# Developing and Building WISER on Windows 10

WISER was developed primarily on MacOS X, so the Windows 10 instructions are
not nearly as well developed.  Currently, Windows 10 is used mainly for
packaging up the application and building an installer for Windows users.
**If anyone wants to refine these instructions for those who wish to develop
on Windows, this would be greatly appreciated.**

## Python Environment - Miniconda 3

[Anaconda](https://www.anaconda.com/) is a widely used Python package
and library manager for Windows and scientific computing.  Unfortunately, the
full Anaconda3 installation doesn't seem to work with the required tools and
libraries, probably because of library/DLL versioning issues.  (As of
2020-06-04, Donnie has not spent the time to track down the causes of these
issues.)

However, the [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html) seems
to work though, and simply requires a bit more setup.

1.  [Install 64-bit Miniconda3](https://docs.conda.io/en/latest/miniconda.html) for
    all users.  The lastest Python 3 version is fine.

2.  Start an Anaconda Prompt in admin mode, to install packages.  From the
    Windows Start menu in the bottom corner:

    Start -> Anaconda3 (64-bit) -> Anaconda Prompt [Right-Click] -> More ->
    Run as Administrator

3.  Just in case the `conda-forge` channel is not already included in the
    Miniconda config, you should do this:
    
    `conda config --add channels conda-forge`

4.  The following dependencies need to be installed via `conda`:

    ```
    conda install pyside2
    conda install gdal
    conda install matplotlib
    conda install astropy
    conda install pyinstaller
    ```

5.  The following dependencies need to be installed via `pip`:

    ```
    pip install pillow
    ```

6.  The `make` utility is used to generate supporting files for Qt 5.

    TODO - WRITE.

## Python IDE - PyCharm

TODO - WRITE.

## Installer - NSIS

The [Nullsoft Scriptable Install System (NSIS)](https://nsis.sourceforge.io/Main_Page)
is used to build the WISER installer.
