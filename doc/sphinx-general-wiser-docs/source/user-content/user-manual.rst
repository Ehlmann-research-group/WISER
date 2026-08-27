User Manual
===========

The reference for WISER's interface: every pane, dialog and option, and what
each one is for.

If you are new to WISER, start with the :doc:`Tutorials <../tutorials/index>`
instead — they walk through the same tools in the order you would actually use
them, on data that ships with the source. Come back here for the detail on a
particular control.

Getting around
--------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Page
     - Covers
   * - :doc:`Interface Overview <interface-overview>`
     - The main window, the panes, the toolbars, the status bar, the Activity
       Monitor, and WISER's preferences
   * - :doc:`Opening Data Files <opening-data-files>`
     - Supported formats, multi-file datasets, sub-datasets, and what to do
       when a file will not open
   * - :doc:`Display and Contrast Stretch <display-and-stretch>`
     - Band selection, colormaps, grid view and linking, and every contrast
       stretch and conditioner
   * - :doc:`Spectra and Spectral Libraries <spectra-and-libraries>`
     - The spectrum plot, collecting and averaging spectra, importing
       libraries, continuum removal
   * - :doc:`Regions of Interest <regions-of-interest>`
     - Defining regions, extracting class signatures, masks, exporting
       geometry and pixel spectra
   * - :doc:`Band Math <band-math>`
     - The expression language, variable binding, saved expressions, batch
       processing
   * - :doc:`Filters and Smoothing <filters>`
     - Savitzky--Golay, mean, median and Gaussian filters
   * - :doc:`Data Analysis Tools <data-analysis-tools/data-analysis-tools>`
     - PCA, MNF, K-means, SAM, SFF, MTMF, linear unmixing, continuum removal,
       interactive scatter plot
   * - :doc:`Spatial Tools <spatial-tools>`
     - Georeferencing, coordinate reference systems, the Reference System
       Creator, similarity transforms
   * - :doc:`Mosaic <mosaic>`
     - Combining overlapping georeferenced scenes and exporting the result
   * - :doc:`Saving and Exporting <saving-and-exporting>`
     - Which "save" does what: datasets, images, spectra, ROIs
   * - :doc:`Saving and Opening Projects <projects>`
     - Saving a whole session to a ``.wiserproj``, choosing what it holds,
       sharing it

Common tasks
------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - I want to...
     - Go to
   * - Open a file that will not load
     - :doc:`Opening Data Files <opening-data-files>`
   * - Make a dark image readable
     - :doc:`Display and Contrast Stretch <display-and-stretch>`
   * - Fix a computed index that displays as one flat colour
     - :doc:`Display and Contrast Stretch <display-and-stretch>`
   * - Get numbers out of a region
     - :doc:`Regions of Interest <regions-of-interest>`
   * - Compute an index such as NDVI
     - :doc:`Band Math <band-math>`
   * - Find out what a mineral is
     - :doc:`Spectral Angle Mapper <data-analysis-tools/spectral-angle-mapper>`
   * - Reduce a 400-band cube to something manageable
     - :doc:`PCA <data-analysis-tools/pca>` or :doc:`MNF <data-analysis-tools/mnf>`
   * - Classify a scene without training data
     - :doc:`K-Means <data-analysis-tools/kmeans>`
   * - Smooth a noisy spectrum without wrecking its bands
     - :doc:`Filters and Smoothing <filters>`
   * - Stitch flight lines together
     - :doc:`Mosaic <mosaic>`
   * - Send a colleague my whole session
     - :doc:`Saving and Opening Projects <projects>`
   * - Add my own algorithm
     - :doc:`Extending WISER <../extending-wiser/index>`

.. toctree::
   :hidden:

   interface-overview
   opening-data-files
   display-and-stretch
   spectra-and-libraries
   regions-of-interest
   band-math
   filters
   data-analysis-tools/data-analysis-tools
   spatial-tools
   mosaic
   saving-and-exporting
   projects
