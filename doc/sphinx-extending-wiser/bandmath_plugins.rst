Band-Math Plugins
=================

The WISER band-math functionality may be extended by implementing a
:class:`wiser.plugins.BandMathPlugin` instance that provides implementations of
the :class:`wiser.bandmath.BandMathFunction` type.  The plugin itself is very
straightforward, but the implementation of functions is somewhat involved, as
WISER must manage various details in the band-math implementation.

The ``BandMathPlugin`` Class
----------------------------

Band-math plugins must derive from the ``BandMathPlugin`` class:

.. autoclass:: wiser.plugins.BandMathPlugin
    :members:

The plugin implementation is very straightforward; it must simply return a
Python ``dict`` that associates string function-names and corresponding
``BandMathFunction`` instances.

The implementation of the band-math functions themselves is more involved;
read on for details.

The ``BandMathFunction`` Class
------------------------------

Band-math functions must derive from the ``BandMathFunction`` class:

.. autoclass:: wiser.bandmath.BandMathFunction
    :members:

For maximum flexibility, band-math functions may accept any number and type of
arguments.  For example, a band-math function ``dotprod(a, b)`` may accept two
spectra (returning a number), a spectrum and a data-set (for a pixel-wise
dot-product of the spectrum with the data-set, returning a 1-band data-set),
or a data-set and a spectrum (same).  Therefore, functions are responsible for
reporting errors if they are given the wrong number of arguments, or if the
arguments are of the wrong types.  Additional checks may also be necessary, if
e.g. there is a mismatch in the number of bands between arguments, or a mismatch
in the spatial dimensions of images or bands.

The ``BandMathValue`` Class
---------------------------

Band-math function arguments and return-values are wrapped in the
``BandMathValue`` class:

.. autoclass:: wiser.bandmath.BandMathValue
    :members:

For arguments, non-scalar band-math values may be fetched from a
``BandMathValue`` object via the :meth:`BandMathValue.as_numpy_array()` method.
Scalar values may of course be retrieved directly from the
:attr:`BandMathValue.value` member.

For function return-values, results must also be wrapped in a ``BandMathValue``
object.  The result may be a NumPy ``ndarray`` instance, or a specific kind of
object (e.g. ``RasterDataSet``, ``Spectrum``, ...), or a scalar.  In all these
cases, the type of the result must be reported with a value from the
:class:`wiser.bandmath.VariableType` enumeration:

.. autoenum:: wiser.bandmath.VariableType

Reporting Other Function Details
--------------------------------

:class:`BandMathFunction` implementations should also implement the other
specified operations, to fully integrate with the functionality exposed in
WISER.  For example, the :meth:`wiser.bandmath.BandMathFunction.get_result_type`
method reports the type of the result based on the number and types of the
arguments, which is then reported to the user in the GUI.

.. warning::
    Additional abstract methods will be added to the ``BandMathFunction`` type
    in the near future.  For example, WISER needs the ability to estimate the
    memory requirements for evaluating a given band-math expression, as
    operations may be very resource-intensive.  Thus, functions will need to
    expose this kind of information in the future.

    The goal of the band-math implementation will be to not *require* these new
    operations, so that existing band-math plugins are rendered unusable when
    new implementation details are added.  However, for full integration into
    WISER's capabilities, it would be valuable to implement missing
    functionality as quickly as possible.

Implementation Suggestions
--------------------------

The present form of the band-math functionality requires the use of a number of
wrapper objects for functions and values.  Thus, plugin writers are encouraged
to separate the *operations themselves* from the *band-math integration code*.
This will ensure that useful operations you may create are still widely useful
in your own code, and only the integration-points with the WISER band-math
functionality will require the additional overhead of the above classes.

Additionally, it is suggested that computational operations should be
implemented to operate on NumPy ``ndarray`` objects for maximum generality,
although it may also be useful to implement certain operations against
:class:`wiser.raster.RasterDataSet` or :class:`wiser.raster.Spectrum` objects,
where metadata is available for use.

Example BandMath Plugin
-----------------------

Okay, lets get into an example plugin that lets you do a spectral angle calculation
between a spectrum and a dataset.

.. literalinclude:: ../../src/example_plugins/bandmath_plugin.py
   :language: python
