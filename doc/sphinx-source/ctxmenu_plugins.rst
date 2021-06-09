Context-Menu Plugins
====================

Context-menu plugins can provide additional tools and capabilities in specific
scenarios within WISER.  For example, operations can be provided on datasets or
or other objects picked by a user (a point in a dataset, a spectrum, a region of
interest, etc.).  To do this, the plugin must subclass the
:class:`wiser.plugins.ContextMenuPlugin` type, filling in the various operations
that WISER will call.

Implementing a context-menu plugin requires some familiarity with Qt 5, since
the plugin must, at a minimum, add ``QMenu`` actions for specific operations
that are exposed.  If a plugin intends to expose its own GUI for configuration
or other user interactions, please see :ref:`GUI Plugins in WISER` for more
details on how this may be done.  You will also need to understand everything
in this section so that you can interface effectively with WISER's internals.

The ``ContextMenuPlugin`` Class
-------------------------------

Context-menu plugins must derive from the ``ContextMenuPlugin`` class.  The
documentation for this class spells out the essential details for interfacing
with WISER.

.. autoclass:: wiser.plugins.ContextMenuPlugin
    :members:

The ``ContextMenuType`` enumeration is as follows:

.. autoenum:: wiser.plugins.ContextMenuType
