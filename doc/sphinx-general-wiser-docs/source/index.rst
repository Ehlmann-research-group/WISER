WISER Documentation
===================

**WISER** — the Workbench for Imaging Spectroscopy Exploration and Research —
is an open-source, cross-platform application for visualising and analysing
hyperspectral imagery. It is written in Python on Qt/PySide, runs on
**macOS, Windows and Linux**, and needs no commercial licence.

Developed and maintained by the
`Ehlmann Research Group <https://github.com/Ehlmann-research-group>`_ at Caltech
and CU Boulder. Questions: wiser_AT_lists.lasp.colorado.edu.

.. rubric:: Start here

.. list-table::
   :widths: 30 70

   * - :doc:`Install WISER <installation>`
     - Download an installer, or run from source
   * - :doc:`Tutorials <tutorials/index>`
     - Seven short walkthroughs on data that ships with WISER, then six applied
       labs on full public datasets
   * - :doc:`User Manual <user-content/user-manual>`
     - The reference for every pane, dialog and option
   * - :doc:`Extend WISER <extending-wiser/index>`
     - Add tools, context-menu actions and band-math functions as plugins
   * - :doc:`Developer Guide <developer-content/index>`
     - Build it, test it, and understand how it works inside

----

What WISER does
---------------

**Look at data**

- Load, display and navigate hyperspectral and multispectral rasters
- Context, main and zoom panes that stay in step, plus a grid view for
  comparing datasets side by side
- Contrast stretches, conditioners, colormaps and a decorrelation stretch

**Read spectra**

- Click any pixel for its spectrum; collect, colour, average and export them
- Import ENVI spectral libraries and ASCII spectra
- Continuum removal on a spectrum, a collection, or a whole cube

**Compute**

- Band math with a full expression language, saved expressions and batch
  processing over a folder
- Regions of Interest for class signatures, masks and pixel exports
- Savitzky–Golay, mean, median and Gaussian filters

**Analyse**

- **Transforms** — Principal Component Analysis, Minimum Noise Fraction
- **Classification** — K-means clustering
- **Detection** — Spectral Angle Mapper, Spectral Feature Fitting,
  Mixture-Tuned Matched Filter
- **Unmixing** — Linear Unmixing with per-pixel residuals
- **Visualisation** — Interactive Scatter Plot linked back to the image

**Handle geometry**

- Georeferencing from ground control points, custom coordinate reference
  systems, similarity transforms
- Mosaicking overlapping scenes onto one output grid

**Keep and share your work**

- Projects that save the whole session to a single file, referenced or
  self-contained
- A plugin API for adding your own analyses without rebuilding

----

Formats WISER reads
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Format
     - Typical files
   * - ENVI raster
     - ``*.img``, ``*.hdr``, ``*.dat``, or no extension
   * - TIFF / GeoTIFF
     - ``*.tif``, ``*.tiff``, ``*.tfw``
   * - NetCDF
     - ``*.nc``
   * - JPEG 2000
     - ``*.JP2``
   * - PDS3
     - ``*.PDS``, ``*.img``, ``*.lbl``
   * - PDS4
     - ``*.xml``
   * - FITS
     - ``*.fits``, ``*.fit``, ``*.fts``
   * - ASCII Grid
     - ``*.asc``
   * - ENVI spectral library
     - ``*.sli``, ``*.hdr``
   * - Anything else GDAL reads
     - HDF4/5, GRIB, COG and many more

See :doc:`Opening Data Files <user-content/opening-data-files>` for
multi-file datasets, sub-datasets and troubleshooting.

----

Data to try
-----------

The :doc:`Getting Started tutorials <tutorials/index>` run on fixtures in the
WISER source tree, so there is nothing to download. For full scenes, the
:doc:`Applied Labs <tutorials/labs/index>` walk through these end to end:

- `AVIRIS-NG Caltech subset <https://avng.jpl.nasa.gov/pub/DThompson/istutor/ang20171108t184227_corr_v2p13_subset_bil>`_
  (plus its `header <https://avng.jpl.nasa.gov/pub/DThompson/istutor/ang20171108t184227_corr_v2p13_subset_bil.hdr>`__)
  --- 425 bands over Pasadena, and the basis of
  :doc:`Lab A <tutorials/labs/lab-aviris-ng-urban>`
- `AVIRIS-Classic reflectance over Cuprite, Nevada
  <https://popo.jpl.nasa.gov/pub/RKokaly/f230918t01p00r11_rfl>`_
  (plus its `header <https://popo.jpl.nasa.gov/pub/RKokaly/f230918t01p00r11_rfl.hdr>`__)
  --- 224 bands, the basis of :doc:`Lab B <tutorials/labs/lab-cuprite-minerals>`
- `AVIRIS Data Portal <https://aviris.jpl.nasa.gov/dataportal/>`_ and
  `AVIRIS free data <https://aviris.jpl.nasa.gov/data/free_data.html>`_ ---
  more airborne scenes
- `EMIT L2A Reflectance <https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl2arfl-001>`_ ---
  spaceborne imaging spectroscopy of arid land surfaces
- `PACE/OCI <https://pace.oceansciences.org/access_pace_data.htm>`_ ---
  hyperspectral ocean colour
- `CRISM MTRDR over Jezero Crater
  <https://pds-geosciences.wustl.edu/mro/mro-m-crism-5-rdr-mptargeted-v1/mrocr_4001/mtrdr/2007/2007_029/hrl000040ff/>`_
  --- 489 bands, the basis of :doc:`Lab C <tutorials/labs/lab-mars-crism>`
- `PDS Geosciences Node <https://pds-geosciences.wustl.edu/>`_ and the
  `Mars Orbital Data Explorer <https://ode.rsl.wustl.edu/mars/>`_ ---
  CRISM, OMEGA, M3 and more
- `Ehlmann Lab datasets <https://lasp.colorado.edu/ehlmann-lab/datasets/>`_ ---
  laboratory and field imaging spectroscopy
- `USGS Spectral Library Version 7 <https://dx.doi.org/10.5066/F7RR1WDJ>`_ and
  the `ECOSTRESS Spectral Library <https://speclib.jpl.nasa.gov/>`_ ---
  reference spectra

----

.. rubric:: Get help

- **Questions, bugs, feature requests:**
  `open a GitHub Issue <https://github.com/Ehlmann-research-group/WISER/issues/new/choose>`_
- **Community plugins:**
  `WISER Plugin Repository <https://github.com/Ehlmann-research-group/WISER-Plugin-Repository>`_
- **Release announcements:** email ``sympa@lists.lasp.colorado.edu`` with the
  subject **subscribe wiser-announcements**

.. rubric:: License

Copyright 2019–2026, California Institute of Technology (Caltech) and
Regents of the University of Colorado. All rights reserved.
See the `LICENSE <https://github.com/Ehlmann-research-group/WISER/blob/main/LICENSE>`_
for the full text.

.. toctree::
   :hidden:

   installation
   tutorials/index
   user-content/user-manual
   extending-wiser/index
   developer-content/index
   contributing
