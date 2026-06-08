WISER Documentation
===================

**WISER** (Workbench for Imaging Spectroscopy Exploration and Research) is an
open-source, cross-platform GUI for visualizing and analyzing hyperspectral
imagery. Built in Python on Qt/PySide2, it runs on **macOS, Windows, and
Linux** with no commercial license required.

Developed and maintained by the `Ehlmann Research Group <https://github.com/Ehlmann-research-group>`_
at Caltech & CU Boulder. For any questions, contact wiser_AT_caltech.edu.

**Key capabilities:**

- Load, display, and navigate hyperspectral and multispectral raster data
- Click any pixel to view and collect its spectrum in a live spectral plot
- Regions of Interest (ROIs) for area-averaged spectra and pixel exports
- Contrast stretch and color mapping for display control
- Band math with custom expressions and plugin-defined functions
- Analysis tools: PCA, Spectral Angle Mapper, Continuum Removal, Scatter Plot,
 Minimum Noise Fraction, K-means, and more
- Georeferencing and coordinate system management
- Extensible plugin system (Tools Menu, Context Menu, Band Math)

----

Sample Datasets
---------------

WISER natively opens several hyperspectral and spectral-library formats, and
can fall back to any format supported by GDAL for additional coverage.

**Natively supported formats**

- **ENVI raster** (``*.img``, ``*.hdr``)
- **TIFF / GeoTIFF** (``*.tiff``, ``*.tif``, ``*.tfw``)
- **NetCDF** (``*.nc``)
- **JPEG 2000** (``*.JP2``)
- **PDS raster** (``*.PDS``, ``*.img``, ``*.lbl``, ``*.xml``)
- **ENVI spectral libraries** (``*.sli``, ``*.hdr``)
- **GDAL-readable formats** --- any format that GDAL can open (e.g. HDF4/5,
  GRIdded Binary, COG, and more)

**Example datasets to try**

The following publicly available datasets are good starting points for
exploring WISER:

- `Sample AVIRIS-NG image of Caltech <https://avng.jpl.nasa.gov/pub/DThompson/istutor/ang20171108t184227_corr_v2p13_subset_bil>`_
  (also download the matching
  `header file <https://avng.jpl.nasa.gov/pub/DThompson/istutor/ang20171108t184227_corr_v2p13_subset_bil.hdr>`_)
- `AVIRIS Data Portal <https://aviris.jpl.nasa.gov/dataportal/>`_ --- archive of airborne imaging-spectrometer scenes
- `PDS Geosciences Node <https://pds-geosciences.wustl.edu/>`_ --- planetary hyperspectral datasets (CRISM, OMEGA, and more)
- `Ehlmann Lab datasets <https://ehlmann.caltech.edu/publications/datasets.html>`_ --- laboratory and field imaging-spectroscopy data

----

Subscribe to Email Updates
--------------------------

To receive notifications about new WISER releases, send an email to
``wiser-announce-request@caltech.edu`` with the subject line **Subscribe**.
Follow the instructions in the reply to confirm your subscription.


Installation
------------

Download WISER
~~~~~~~~~~~~~~

Pre-built installers for macOS, Windows, and Linux are available at:
`ehlmann.caltech.edu/wiser <https://ehlmann.caltech.edu/wiser/index.html>`_

Download the installer for your platform and follow the on-screen instructions.
Users can also download and install WISER from
`GitHub Releases <https://github.com/Ehlmann-research-group/WISER/releases>`_.

.. note::

   The download location will change in a future release as WISER transitions
   to CU Boulder. Links on this page will be updated when that happens.

Running from Source
~~~~~~~~~~~~~~~~~~~

If you want to run WISER from source or contribute to development, see the
:doc:`Developer Environment Setup <developer-content/environment-setup>` guide.

In brief:

.. code-block:: bash

   cd etc
   make install-dev-env          # macOS/Linux
   conda activate wiser-dev
   cd ../src
   python -m wiser

Supported Platforms
~~~~~~~~~~~~~~~~~~~

WISER builds currently target:

- **macOS 15** --- ARM (Apple Silicon) and Intel
- **Windows 10/11**
- **Linux** --- Ubuntu 20.04+, Debian 11+, Fedora 39+ (amd64 and aarch64)

System Requirements
^^^^^^^^^^^^^^^^^^^

The minimum and recommended specifications for running WISER are:

**Operating system**

- **Windows:** Windows 10 or 11 (64-bit)
- **macOS:** macOS 15 or newer (Intel and Apple Silicon)
- **Linux:** Ubuntu 20.04+, Debian 11+, or Fedora 39+ (amd64 and aarch64)

**Hardware**

- **CPU architecture:** x86_64 (Intel/AMD) or arm64 (Apple Silicon / aarch64)
- **Memory:** 8 GB minimum; 16--32 GB recommended for large datasets
- **Storage:** ~1 GB for installation; SSD strongly recommended
- **GPU:** Not required

If you encounter issues building or running WISER, please
`open a GitHub Issue <https://github.com/Ehlmann-research-group/WISER/issues>`_.

----

Where to Go Next
----------------

.. rubric:: User Manual

:doc:`Interface Overview <user-content/interface-overview>` ·
:doc:`Working with Data <user-content/working-with-data>` ·
:doc:`Spatial Tools <user-content/spatial-tools>`

.. rubric:: Extend WISER

:doc:`Build plugins <extending-wiser/index>` to add custom workflows,
context-menu operations, and band-math functions — no rebuild required.

.. rubric:: Developer Guide

:doc:`Environment Setup <developer-content/environment-setup>` ·
:doc:`Contributing & Code Quality <developer-content/contributing-and-quality>` ·
:doc:`Testing & QA <developer-content/testing-and-qa>`

----

.. rubric:: Get Help

- **Questions, bugs, feature requests:** `Open a GitHub Issue <https://github.com/Ehlmann-research-group/WISER/issues>`_
- **Community plugins:** `WISER Plugin Repository <https://github.com/Ehlmann-research-group/WISER-Plugin-Repository>`_

.. rubric:: License

Copyright 2019–2026, California Institute of Technology (Caltech) and
Regents of the University of Colorado. All rights reserved.
See the `LICENSE <https://github.com/Ehlmann-research-group/WISER/blob/main/LICENSE>`_ for the full text.

----

.. toctree::
   :hidden:

   user-content/user-manual
   extending-wiser/index
   developer-content/environment-setup

   contributing
