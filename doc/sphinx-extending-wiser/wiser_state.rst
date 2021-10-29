Accessing and Modifying WISER State
===================================

Tool plugins and context-menu plugins can access and manipulate WISER
application state through the ``ApplicationState`` class.  The plugin can
access an instance of this class via the ``"wiser"`` value in the context
passed to the plugin.

.. autoclass:: wiser.gui.app_state.ApplicationState
    :members:

Operations involving loading and storing raster data sets will require the use
of a loader object, also accessible off of the ``ApplicationState`` object.
See the ``get_loader()`` function above.  The loader offers these operations:

.. autoclass:: wiser.raster.RasterDataLoader
    :members:

Loading Raster Data into WISER
------------------------------

The above classes expose two ways to load raster data into WISER.  The first
option is to use the ``ApplicationState.open_file()`` function, which simply
loads a file into WISER.  Several kinds of files (raster images, spectral
libraries, and a few other kinds of files) are supported by this method.  No
value is returned, so this function is not useful for getting access to the
contents of a specific raster file.

To open a raster data file and get back an object to access its contents
requires the use of the ``RasterDataLoader``.  The loader is accessed from the
``ApplicationState.get_loader()`` method, as described above.  Note that any
raster image loaded this way is not automatically displayed; after it has been
loaded and possibly manipulated, it must be added to the ``ApplicationState``
object with the ``add_dataset()`` method, before it will be displayed.

Similarly, a NumPy array can be converted into a ``RasterDataSet`` object using
the ``RasterDataLoader.dataset_from_numpy_array()`` method.  It will likely be
desirable to add metadata to the resulting ``RasterDataSet`` object (it is
convenient to copy it from an existing data-set, if the new object was generated
from an existing data-set), but this is not required.

Finally, it is not advisable to change a ``RasterDataSet`` object after handing
it to WISER for display, as WISER currently expects that the object will not
change once it has been given to WISER.
