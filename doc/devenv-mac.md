# Developing and Building WISER on MacOS X

WISER was developed primarily on MacOS X, so it is pretty straightforward to set
up a development environment on a Mac.  **Currently these instructions require
the use of [MacPorts](https://www.macports.org).  If anyone wants to figure out
how to set up a development environment under [Homebrew](https://brew.sh) and
write up instructions, this would be greatly appreciated.**

WISER requires the Qt5 and GDAL libraries to be installed on the system.  Under
MacPorts this is straightforward:

```
sudo port install qt5
sudo port install gdal
```

Once these operations are completed, one can set up a Python environment to
develop and run the application.  

```
# NOTE:  Python 3.7 is recommended for WISER development, to avoid
#        compatibility issues with Qt5 and PySide2.
sudo port install python37

# Install your favorite version of virtualenv, at least Python 3.7, but
# Python 3.8+ will also work.
# EXAMPLE:  sudo port install py37-virtualenv

# Set up a virtual environment that uses the Python 3.7 interpreter.
virtualenv --python=/opt/local/bin/python3.7 venv

# Activate the new Python 3.7 virtual environment.
source venv/bin/activate
```

At this point, you can install all the requirements:

```
pip install -r requirements.txt
```

If you are feeling more ambitious, you can install the minimal set of
requirements manually, so that you get the most recent versions of everything.

```
pip install numpy
pip install astropy
pip install matplotlib==3.2.2
pip install GDAL
pip install PySide2
```

>   NOTE 1:  PyInstaller 4.0 has a bug in its support of matplotlib 3.3.0.  This
>   is why we must install matplotlib 3.2.2 for the time being.  The bug
>   manifests as an inability to start the packaged Mac distributable.

>   NOTE 2:  If you get an error like "no module \_gdal_array" after setting up
>   WISER, a common cause is that the GDAL Python library was installed _before_
>   NumPy was installed.  To further complicate the matter, `pip` may have
>   cached the GDAL library without the `_gdal_array` module.  Thus, to fix the
>   issue, you can do this:
>
>       pip uninstall GDAL
>       pip install numpy                  (if not already installed)
>       pip install --no-cache-dir GDAL

For full support of all build steps, including packaging a Mac distributable,
you also need to install these tools:

```
pip install pylint
pip install mypy
pip install pyinstaller
```

## Important Build Targets

Before you can run WISER, you must generate various files required by the Qt
runtime.  These include a `resources.py` file containing all icons used in the
project, and also a number of `*_ui.py` files which contain UI view code
generated from the corresponding `*.ui` files created with
[Qt Designer](https://doc.qt.io/qt-5/qtdesigner-manual.html).  These files are
generated into the `src/gui/generated` directory of the code-base.

The WISER project uses `make` to automate various build steps, since it is a
simple and reliable tool for this kind of work.

To generate these files, go to the top directory of the WISER project, and run:

```
# Just typing 'make' will also run the 'generated' target.
make generated
```

At this point, you should be able to run WISER like this:

```
cd src
./wiser [file1 file2 ...]
```

To build a distributable `.dmg` file, go back to the top level directory of the
WISER project, and build this `make` target:

```
make dist-mac
```

To clean up all generated files and distributable directories, do this from the
top level directory of the WISER project:

```
make clean
```
