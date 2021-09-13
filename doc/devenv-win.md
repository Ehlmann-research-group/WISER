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
    # Install NumPy first, making sure OpenBLAS is available so that the HUGE
    # MKL libraries are not present (they cause the binary to be HUGE.)
    conda install conda-forge::blas=*=openblas
    conda install -c conda-forge numpy

    conda install pyside2
    conda install gdal

    # This verison of matplotlib is required due to a pyinstaller incompatibility
    conda install matplotlib=3.2.2
    
    conda install astropy
    conda install pyinstaller
    ```

    >   NOTE:  PyInstaller 4.0 has a bug in its support of matplotlib 3.3.0.  This
    >   is why we must install matplotlib 3.2.2 for the time being.  The bug
    >   manifests as an inability to start the packaged Windows distributable.

5.  The following dependencies need to be installed via `pip`:

    ```
    pip install pillow
    pip install bugsnag
    pip install lark
    ```

6.  The `make` utility is used to generate supporting files for Qt 5.

    TODO:  GNU Make for Windows

## Python IDE - PyCharm

TODO - WRITE.

## Installer - NSIS

The [Nullsoft Scriptable Install System (NSIS)](https://nsis.sourceforge.io/Main_Page)
is used to build the WISER installer.

## Code-Signing

To code-sign the WISER installer, the
[Windows 10 SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/)
needs to be downloaded and installed so that the `SignTool` utility is available.

# Building the Project

1.  Open an Anaconda terminal window.

    Start -> Anaconda3 (64-bit) -> Anaconda Prompt (Miniconda3)

2.  Figure out how to run `make` from the Anaconda terminal.

    I use GNU Make, so I run it like this:

    ```c:\Program Files (x86)\GnuWin32\bin\make.exe```

3.  Go to the PyCharm project directory for WISER.

    On my computer this is:  `C:\Users\donnie\PycharmProjects\WISER`

4.  Clean up any existing build artifacts.

    **This is currently a manual process, because the `clean` target doesn't
    work on Windows yet.**

    Delete these files:

    ```
    build directory
    dist directory
    Install-WISER-*.exe
    src\gui\generated\*.py
    ```

5.  Build the project:

    ```
    "c:\Program Files (x86)\GnuWin32\bin\make.exe" dist-win
    ```

    This should result in the creation of an NSIS installer in the local
    directory.

## Code Signing

"c:\Program Files (x86)\Windows Kits\10\App Certification Kit\signtool.exe" sign /v /debug /f C:\Users\donnie\Downloads\WISER-Codesign-cs /tr http://timestamp.sectigo.com Install-WISER-1.0a4-dev0.exe
