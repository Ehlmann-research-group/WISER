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
