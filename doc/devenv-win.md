# Developing and Building WISER on Windows 10

You will need conda and python installed. You will need to do `pip install conda-lock` and `pip install pyyaml`. I suggest you do this inside of a conda-environment. You will also need to have `make` installed. Once you have all of this installed you are ready to go into the `/etc` folder and run the command `make install-dev-env`.

## How to Install Conda

[Anaconda](https://www.anaconda.com/) is a widely used Python package
and library manager for Windows and scientific computing.  In previous versions of WISER, the full Anaconda3 installation didn't seem to work with the required tools and
libraries, probably because of library/DLL versioning issues. However, between WISER versions 1.1b1 and 1.4b1 it has worked, so I assume it will work for you. If anaconda does not work for you, you can use the [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html).

## IDE

The IDE that WISER is currently (04/14/2026) developed on is Visual Studio Code. Although, I don't see why other IDEs would not work.

## Installer - NSIS

The [Nullsoft Scriptable Install System (NSIS)](https://nsis.sourceforge.io/Main_Page)
is used to build the WISER installer. The installer installs on the user level.

## Installing on Windows

As of release 2.0b1, WISER effectively assumed that only one version of WISER would be installed on a
computer at a time. The installer and uninstaller logic used a single uninstall registry key for WISER,
and the install flow would look up that key and run the previous uninstaller before proceeding with a
new install. In practice, this meant that WISER behaved like a single-install application rather than
a side-by-side application.

As of release 2.1b1, WISER is meant to be installed into any location that the installer considers safe.
This includes places like Program Files, AppData, or another writable folder chosen by the user. The
installer still uses registry keys, but those keys are now versioned using the WISER version number. The
install directory, Start Menu folder, and uninstaller name are also versioned.

This versioned approach means that multiple different versions of WISER can coexist on the same computer without
overwriting each other's uninstall registration. If one version of WISER is uninstalled, that uninstall
removes only the files, shortcuts, and uninstall registry key associated with that specific version.
Other installed versions should remain visible to the system because they keep their own separate uninstall
registry keys.

One subtle point is that the uninstaller does not decide what to delete by scanning the registry for all
WISER installs. Instead, it primarily removes the install directory it is associated with, and then deletes
that version's own uninstall registry key. So if the same version of WISER is installed twice and one version
is uninstalled, the registry key for that version (which there is only one of even though there are two WISERs
installed for that version) will be deleted. The registry is used to register and discover a particular
installed version, not as the master source for removing every WISER installation on the machine. Additionally,
WISER writes app data like config files, log files, and numba cache to a location that is independent of WISER's
install location and is dependent on the WISER version. For example on windows this is %LOCALAPPDATA%/WISER-{version}.
This means that if there are two of the same version of WISER installed onto a computer, they will be writing to
the same directory.

## Code-Signing

To code-sign the WISER installer, the
[Windows 10 SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/)
needs to be downloaded and installed so that the `SignTool` utility is available.

## Building the Project

1.  Open an Anaconda terminal window.

    Start -> "Anaconda3 (64-bit)" -> "Anaconda Prompt (Miniconda3)"

2.  Ensure you are in the _wiser-prod_ conda environment. You first make this environment
    my going to the `/etc` folder and typing `make install-prod-env`. Then do
    `conda activate wiser-prod`.


3.  Figure out how to run `make` from the Anaconda terminal.

    I use GNU Make, so I run it like this:

    ```c:\Program Files (x86)\GnuWin32\bin\make.exe```

4.  Go to the project directory for WISER.

    On my computer this is:  `C:\Users\jgarc\OneDrive\Documents\Schmidt-Code\WISER`

5.  Clean up any existing build artifacts.

    **This is currently a manual process, because the `clean` target doesn't
    work on Windows yet.**

    Delete these files:

    ```
    build directory
    dist directory
    Install-WISER-*.exe
    src\gui\generated\*.py
    ```

6.  Build the project:

    ```
    "c:\Program Files (x86)\GnuWin32\bin\make.exe" dist-win
    ```

    This should result in the creation of an NSIS installer in the local
    directory.

**NOTE:**  On Python 3.9 and PySide2 5.13.2, there is an issue with a Python
file that is part of `pyside2-uic`.  Specifically, the `uiparser.py` file (at
path `C:\ProgramData\Miniconda3\Lib\site-packages\pyside2uic\uiparser.py`)
has a call to `elem.getiterator()` on line 797 that needs to be changed to
`elem.iter()` instead.  Because of where this file lives, it needs to be
edited with Administrator permissions, or else the edits cannot be saved.
WISER currently does not use PySide2 5.13.2, but if for some reason you're
environment has this, be warned.

## Code Signing Revisited

Code signing currently occurs in the file _/install-win/win-install.nsi_. We codesign both WISER's installer and uninstaller. Here is the codesign command we use in that script:

```
'"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool" sign /sha1 "${SHA1_THUMBPRINT}" /fd SHA256 /t http://timestamp.sectigo.com "%1"'
```

Where %1 is replaced by `Install-WISER-1.4b1.exe` in the install script.
