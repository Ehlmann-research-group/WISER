Supporting Types
================

WISER uses a number of types internally to wrap various kinds of information.
These types often hold NumPy ``ndarray`` objects or other such details for the
actual data, and also provide metadata about the information being managed.
These types are important for most kinds of plugins, and are documented here.

Raster Data Sets
----------------

Imaging spectroscopy cubes are represented with the ``RasterDataSet`` class.
This class provides access to both the data and metadata of a spectral image
cube.

.. autodoc2-object:: wiser.raster.dataset.RasterDataSet

Making a Data Set from a NumPy Array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A data-set may be constructed from a NumPy array using the WISER data loader.
See the section :ref:`Loading Raster Data into WISER` for more details.

Finding a Data Set's Files on Disk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A plugin that hands a data set to an external tool needs the file it was read
from rather than its arrays.  Use ``RasterDataSet.get_source_path()``: it
returns the plain path of the file on disk, or ``None`` for a data set that
exists only in memory.  The path is as it was given when the data set was
opened, so call ``os.path.abspath()`` on it if the tool runs in another
directory.

``RasterDataSet.get_filepaths()`` returns every file associated with the data
set, and what it contains depends on how the data set was loaded:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Data set
     - ``get_filepaths()``
   * - File formats read through GDAL (ENVI, GeoTIFF, JPEG 2000, FITS, PDS3,
       PDS4, ASCII grid, and any other format GDAL opens)
     - All files GDAL associates with the data set, primary data file first;
       an ENVI data set lists both its data file and its ``.hdr`` header.
   * - NetCDF subdataset (EMIT, PACE, and other NetCDF products)
     - A single-element list holding the GDAL subdataset descriptor, such as
       ``NETCDF:"/data/EMIT_L2A_RFL.nc":reflectance``, which is not a path.
       ``get_source_path()`` returns ``/data/EMIT_L2A_RFL.nc`` and
       ``get_subdataset_name()`` returns the full descriptor.
   * - In memory (built from a NumPy array, or a result not yet saved)
     - An empty list.

``get_source_path()`` raises ``ValueError`` if a NetCDF subdataset's
descriptor is not of the ``NETCDF:"<path>":<variable>`` form.

Raster Data Bands
-----------------

A single band of a raster data set may be represented by the ``RasterDataBand``
class.  This class is a simple wrapper of a
:class:`wiser.raster.RasterDataSet` object, that also includes the index of the
referenced band.

.. autodoc2-object:: wiser.raster.dataset.RasterDataBand

Spectra
-------

WISER has a rather complex class hierarchy to represent spectra, as they may
come from pixels in a spectral image cube, they may be from a spectral library,
they may be calculated from the area in a Region of Interest or the area around
a pixel in an image, and so forth.  The base-type of all these different kinds
of spectra is the ``Spectrum`` class:

.. autodoc2-object:: wiser.raster.spectrum.Spectrum

Making a Spectrum from a NumPy Array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A spectrum may be constructed from a NumPy array using the
``NumPyArraySpectrum`` subclass:

.. autodoc2-object:: wiser.raster.spectrum.NumPyArraySpectrum


Region Of Interest
------------------

A region of interest that is draw on the screen is represented by the ``Region Of Interest``
class.

.. autodoc2-object:: wiser.raster.roi.RegionOfInterest

DynamicInputDialog
------------------

A utility for plugin developers to use to easily collect user input through a GUI.

.. autodoc2-object:: wiser.gui.ui_library.DynamicInputDialog

DynamicInputType
----------------
.. autodoc2-object:: wiser.gui.ui_library.DynamicInputType

TableDisplayWidget
------------------

A utility for plugin developers to easily display information in a table.

.. autodoc2-object:: wiser.gui.ui_library.TableDisplayWidget

MatplotlibDisplayWidget
-----------------------

A utility for plugin developers to easily display matplotlib plots.

.. autodoc2-object:: wiser.gui.ui_library.MatplotlibDisplayWidget
