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

.. autoclass:: wiser.raster.RasterDataSet
    :members:

Making a Data Set from a NumPy Array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A data-set may be constructed from a NumPy array using the WISER data loader.
See the section :ref:`Loading Raster Data into WISER` for more details.

Raster Data Bands
-----------------

A single band of a raster data set may be represented by the ``RasterDataBand``
class.  This class is a simple wrapper of a
:class:`wiser.raster.RasterDataSet` object, that also includes the index of the
referenced band.

.. autoclass:: wiser.raster.RasterDataBand
    :members:

Spectra
-------

WISER has a rather complex class hierarchy to represent spectra, as they may
come from pixels in a spectral image cube, they may be from a spectral library,
they may be calculated from the area in a Region of Interest or the area around
a pixel in an image, and so forth.  The base-type of all these different kinds
of spectra is the ``Spectrum`` class:

.. autoclass:: wiser.raster.Spectrum
    :members:

Making a Spectrum from a NumPy Array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A spectrum may be constructed from a NumPy array using the
``NumPyArraySpectrum`` subclass:

.. autoclass:: wiser.raster.NumPyArraySpectrum
    :members:


Region Of Interest
------------------

A region of interest that is draw on the screen is represented by the ``Region Of Interest``
class.

.. autoclass:: wiser.raster.RegionOfInterest
    :members:
    :undoc-members:

DynamicInputDialog
------------------

A utility for plugin developers to use to easily collect user input through a GUI.

.. autoclass:: wiser.gui.ui_library.DynamicInputDialog
    :members:

TableDisplayWidget
------------------

A utility for plugin developers to easily display information in a table.

.. autoclass:: wiser.gui.ui_library.TableDisplayWidget
    :members:

MatplotlibDisplayWidget
------------------

A utility for plugin developers to easily display matplotlib plots.

.. autoclass:: wiser.gui.ui_library.MatplotlibDisplayWidget
    :members:
