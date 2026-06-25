# Developer Guide

This guide walks you through setting up a WISER development environment and
covers the full contributor workflow from first clone to merged PR.

**Reading order for new contributors:**

1. **This page** — install conda, set up the dev environment, run WISER from source
2. [Contributing & Code Quality](contributing-and-quality.md) — code style, PR process, review standards
3. [Testing & QA](testing-and-qa.md) — running tests, writing tests, CI
4. [CI/CD and Releases](ci-cd-and-releases.md) — how releases are built and shipped (see
   also: [macOS Code Signing](codesign-mac.md) and [Windows Code Signing](codesign-win.md))
5. [System Design](system-design.md) — architecture overview for deeper changes
6. [Design Documents](design-documents.md) — longer-form design rationale

For a deeper look at how individual subsystems are implemented, see the
[Code Documentation](code-documentation.md) section.

```{toctree}
:hidden:
:maxdepth: 1

contributing-and-quality.md
testing-and-qa.md
ci-cd-and-releases.md
codesign-mac.md
codesign-win.md
system-design.md
plugin-system.md
plugin-dependencies.md
design-documents.md
```

## Developer Environment Setup

This section covers environment setup for all platforms —
lockfiles, conda environments, dependency management, and
platform-specific build and distribution steps.

For macOS code signing, see [macOS Code Signing](codesign-mac.md). For Windows
code signing, see [Windows Code Signing](codesign-win.md).

## Prerequisites

You will need conda and Python installed. You will need to do
`pip install conda-lock` and `pip install pyyaml`. It is recommended to do
this inside a conda environment. You will also need `make` available in your
terminal. Based on your operating system, look up how to install `make`.

## Lockfiles

In WISER under `/etc` there are two very important files for environment setup:
_dev-conda-lock.yml_ and _prod-conda-lock.yml_. These are the lockfiles for
development and production respectively. You should be very hesitant to change
these files. You should also never directly edit them!

If you want to change the dependencies that these lockfiles have, you will first
have to decide if the dependency should go into the prod lockfile or the dev
lockfile. Dependencies in the prod lockfile are needed for WISER to run in
production. Dependencies in the dev lockfile are needed to run WISER from
source on GitHub and do all of the tasks a developer would do — like run the
tests or generate Sphinx docs. Then you will have to follow the instructions
below for each type of dependency you want to change.

## Changing Production (prod) Dependencies

To change dependencies for production, go into the file _etc/wiser-prod.yml_ and
add the dependency. The dependency will either have to be a dependency on conda
that we can get from conda-forge, or a dependency that we can get from pip. Add
your dependency and version to the .yml file in the appropriate location (under
`dependencies` if it is from conda, and under `pip` if it is from PyPI).

It is encouraged to use as accurate a version number as you can (try to use
major.minor.patch). We do not want our dependency versions to jump a lot when
we add a new dependency. However, having big jumps in dependency versions may
be unavoidable for important/large dependencies like Python, but we would still
want to do this infrequently. This is important because plugin creators must
partially base their plugin dependencies off of WISER's dependencies.

## Changing Development (dev) Dependencies

To change dependencies for development, first go to the file
`etc/wiser-dev-additions.txt` and add the dependency either under the
`dependencies:` line (for conda-forge dependencies) or under the `pip:` line
for pip dependencies. Follow the example at the top of the file. Note that you
only use one equal sign to specify versions in this file. You would then go to
the root directory and run:

```
python src/devtools/make_dev_env.py etc/wiser-prod.yml etc/wiser-dev-additions.txt etc/wiser-dev.yml
```

This command will generate `wiser-dev.yml`, combining the packages in
`wiser-dev-additions` with those in `wiser-prod.yml`. You can also go into the
`etc/` folder and run `make dev-yaml`.

## Regenerating the Lock File

To regenerate the dev lockfile, run `conda-lock lock -f wiser-dev.yml` and then
rename the lock file to _dev-conda-lock.yml_. To regenerate the prod lockfile,
run `conda-lock lock -f wiser-prod.yml` and rename the lockfile to
_prod-conda-lock.yml_.

You can also go into the `etc/` folder and run `make create-lockfiles` to
create both the dev and the prod lockfiles. To create just the dev lockfile,
run `make dev-lockfile`; to create just the prod lockfile, run
`make prod-lockfile`. If you run `make dev-lockfile` or `make create-lockfiles`
and the dev .yml file has not been created yet, these commands will create it
first.

(installing-environment-from-lockfile)=
## Installing Environment from Lockfile

Once you have the lock files, run:

```
conda-lock install -n <conda-env-name> dev-conda-lock.yml
```

where `conda-env-name` is the name of the conda environment you want to install
into. You can also add `--force-platform <OS-name>` to specify a target OS. For
example, to build an Intel conda environment on an ARM Mac:

```
conda-lock install -n wiser-intel --force-platform osx-64 dev-conda-lock.yml
```

The text `osx-64` targets Intel Mac.

If you are on macOS, to fully ensure your conda environment is the right
platform, you may also need to run `conda config --env --set subdir osx-64`
(for Intel) or `conda config --env --set subdir osx-arm64` (for ARM) while
inside the conda environment. These commands only need to be run once.

You can also run `make install-dev-env` to install the dev conda environment,
or `make install-prod-env` to install the prod conda environment. On macOS, use
`make install-dev-env ENV=intel` or `make install-dev-env ENV=arm` to
explicitly target Intel or ARM. If the dev .yml and lockfiles have not been
created, these install commands will create them first.

## Running WISER

Once you have the above steps done and are in the appropriate conda environment,
go into the `src` folder and run:

```
python -m wiser
```

WISER should start running. If it is your first time running it after your
computer has started, it may take a little longer.

## Platform-Specific Build & Distribution

### macOS

You will need conda and Python installed, along with `pip install conda-lock`
and `pip install pyyaml`. You will also need `make` installed. Once you have
all of this, go into the `/etc` folder and run `make install-dev-env`. If you
are on an ARM Mac and want your dev environment to target Intel Macs, run
`make install-dev-env ENV=intel`.

#### Important Build Targets

Before you can run WISER, you must generate various files required by the Qt
runtime. These include a `resources.py` file containing all icons used in the
project, and a number of `*_ui.py` files which contain UI view code generated
from the corresponding `*.ui` files created with
[Qt Designer](https://doc.qt.io/qt-5/qtdesigner-manual.html). These files are
generated into the `src/wiser/gui/generated/` directory.

WISER uses `make` to automate various build steps. To generate these files, go
to the top directory of the WISER project and run:

```
# Just typing 'make' will also run the 'generated' target.
make generated
```

At this point, you should be able to run WISER:

```
cd src
python -m wiser
```

#### Building a Distributable .dmg

First make sure you are in the `wiser-prod` conda environment. Create this
environment by going into the `/etc` folder and running either
`make install-prod-env ENV=intel` or `make install-prod-env ENV=arm` depending
on your CPU architecture. Then activate the environment:

```
conda activate wiser-prod-arm    # macOS ARM
conda activate wiser-prod-intel  # macOS Intel
```

Then go back to the top-level directory of the WISER project and build:

```
make dist-mac
```

You can test the Mac application like this:

```
open dist/WISER.app
```

To clean up all generated files and distributable directories from the
top-level directory:

```
make clean
```

### Windows

You will need conda and Python installed, along with `pip install conda-lock`
and `pip install pyyaml`. You will also need `make` installed. Once you have
all of this, go into the `/etc` folder and run `make install-dev-env`.

#### How to Install Conda

[Anaconda](https://www.anaconda.com/) is a widely used Python package and
library manager for Windows and scientific computing. In previous versions of
WISER, the full Anaconda3 installation had some issues, but between WISER
versions 1.1b1 and 1.4b1 it has worked. If Anaconda does not work for you, use
the [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html).

#### IDE

The IDE that WISER is currently developed on is Visual Studio Code, although
other IDEs should work as well.

#### Installer — NSIS

The [Nullsoft Scriptable Install System (NSIS)](https://nsis.sourceforge.io/Main_Page)
is used to build the WISER installer. The installer can install at both the user and admin level.

#### Installing on Windows

As of release 2.1b1, WISER is installed into a versioned location chosen by the
installer (Program Files, AppData, or another writable folder). The installer
uses versioned registry keys so that multiple different versions of WISER can
coexist on the same machine without overwriting each other's uninstall
registration. Uninstalling one version removes only that version's files,
shortcuts, and registry key; other installed versions are unaffected.

WISER writes app data (config files, log files, numba cache) to a location
independent of the install directory and dependent on the WISER version — for
example `%LOCALAPPDATA%/WISER-{version}` on Windows. If two copies of the same
version are installed, they share that app-data directory.

#### Code Signing

To code-sign the WISER installer, the
[Windows 10 SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/)
needs to be installed so that the `SignTool` utility is available. Code signing
occurs in _/install-win/win-install.nsi_. Both the WISER installer and
uninstaller are code-signed. The sign command used in that script is:

```
"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool" sign /sha1 "${SHA1_THUMBPRINT}" /fd SHA256 /t http://timestamp.sectigo.com "%1"
```

where `%1` is replaced by the installer filename (e.g. `Install-WISER-1.4b1.exe`).

For a full walkthrough of Windows code signing — the tools you need (NSIS and
the Windows SDK), how to configure the `signtool` path, how to find your
certificate thumbprint, and a step-by-step explanation of how the installer
script signs both the installer and uninstaller — see
[Windows Code Signing](codesign-win.md).

#### Building the Project

1. Open an Anaconda terminal window:

   Start → "Anaconda3 (64-bit)" → "Anaconda Prompt (Miniconda3)"

2. Ensure you are in the _wiser-prod_ conda environment. Create it by going to
   the `/etc` folder and running `make install-prod-env`, then:

   ```
   conda activate wiser-prod
   ```

3. Figure out how to run `make` from the Anaconda terminal. Using GNU Make:

   ```
   c:\Program Files (x86)\GnuWin32\bin\make.exe
   ```

4. Go to the project directory for WISER, for example:

   ```
   C:\Users\<username>\WISER
   ```

5. Clean up any existing build artifacts.

   **This is currently a manual process because the `clean` target does not
   work on Windows yet.**

   Delete these items:
   ```
   build directory
   dist directory
   Install-WISER-*.exe
   src\wiser\gui\generated\*.py
   ```

6. Build the project:

   ```
   "c:\Program Files (x86)\GnuWin32\bin\make.exe" dist-win
   ```

   This should result in the creation of an NSIS installer in the local
   directory.

**NOTE:** On Python 3.9 and PySide2 5.13.2, `uiparser.py` (at
`C:\ProgramData\Miniconda3\Lib\site-packages\pyside2uic\uiparser.py`) has a
call to `elem.getiterator()` on line 797 that must be changed to `elem.iter()`.
That file requires Administrator permissions to edit. WISER does not currently
use PySide2 5.13.2, but this is noted for reference.

### Linux

You will need conda and Python installed, along with `pip install conda-lock`
and `pip install pyyaml`. You will also need `git` and `make` installed, and
`build-essential` is recommended. Once you have all of this, go into the `/etc`
folder and run `make install-dev-env`.

#### Supported Distributions and Architectures

WISER builds target Ubuntu 20.04, Debian 11, and Fedora 39. Builds should
generally work on future versions of these distributions due to their backwards
compatibility goals. Supported Instruction Set Architectures (ISAs) are amd64
(x86_64) and aarch64 (arm64).

Development has been tested on Ubuntu 22.04.5 amd64. The instructions below
should work for other Linux distributions.

#### How to Install Conda on Linux

Follow the instructions at
https://www.anaconda.com/docs/getting-started/miniconda/main to install
Miniconda. Then follow the [Installing Environment from Lockfile](#installing-environment-from-lockfile)
section above to set up your conda environments.

#### How to Build WISER

Once you have your conda environments set up, activate the `wiser-prod`
environment and run `make build-linux` from the root directory of the
repository. This produces a Linux build targeting your current distribution
and ISA.

It is recommended to use the GitHub Action to create your Linux build, as it
handles dynamic library dependency issues that you will likely encounter when
building locally with `make build-linux`.

The build output is placed under `/dist`. To test it, go to `/dist/WISER` and
run:

```
./WISER_Bin
```

#### Continuous Deployment

Official WISER releases for Linux are produced by the GitHub Action
`Build and Smoke WISER`. The process:

1. A matrix strategy runs jobs in parallel targeting Ubuntu 20.04 + amd64,
   Debian 11 + amd64, Fedora 39 + amd64, Ubuntu 20.04 + arm64, Debian 11 +
   arm64, and Fedora 39 + arm64.
2. For each target, `./build_linux_multistage.sh` is run with the OS and ISA,
   which runs the corresponding Docker bake target in `docker_bake.hcl`. The
   Dockerfile produces an output artifact at `/app/WISER.tar.gz` in the
   container.
3. `./build_linux_multistage.sh` copies `/app/WISER.tar.gz` to the host. It
   contains:
   - `install_wiser_appimage.sh` — installs the .AppImage on the user's machine
   - `RUN.txt` — directions for running WISER
   - `SHA256SUMS` — SHA256 for verifying `WISER-x86_64.AppImage`
   - `WISER-x86_64.AppImage` — the actual application
4. The GitHub Action uploads `WISER.tar.gz` to GitHub. Note that the upload
   always re-zips the file, which is slightly redundant since it is already
   compressed.

## Other Useful Tools

### GitHub CLI Tool

The GitHub CLI Tool provides the `gh` command, which is used to download and
sign artifacts from the deployment pipeline. Installation instructions can be
found at https://github.com/cli/cli#installation.

## Listing Out Dependencies from Lockfile

You may need to create a .yml file to show end users what production
dependencies WISER has. Run:

```
conda-lock render --kind env -p osx-64 -p osx-arm64 -p win-64 prod-conda-lock.yml
```

This prints the dependencies for the specified platforms. Excluding all
`-p <OS-version>` arguments will automatically create .yml files for all
operating systems that _prod-conda-lock.yml_ supports.

**NOTE:** The .yml files produced by `conda-lock render` **cannot** be
substituted for the conda-lock.yml files. Installing a conda environment based
on those rendered .yml files is not always reproducible.
