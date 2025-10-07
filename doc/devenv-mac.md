# Developing and Building WISER on MacOS X

In order to have reproducible environments across all users and developers, WISER has lockfiles for creating conda environments. Most of this information is detailed in _WISER/doc/sphinx-general-wiser-docs/source/developer-content/environment-setup.md_, but I will go over it quickly here.

You will need conda and python installed. You will need to do `pip install conda-lock`. I suggest you do this inside of a conda-environment. You will also need to have `make` installed. Once you have all of this installed you are ready to go into the /etc folder and run the command `make install-dev-env`. If you are on an ARM mac and you want your dev environment to be for Intel macs, do `make install-dev-env ENV=intel`. And that's it!

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

You can easily test the Mac application like this:

```
open dist/WISER.app
```

To clean up all generated files and distributable directories, do this from the
top level directory of the WISER project:

```
make clean
```
