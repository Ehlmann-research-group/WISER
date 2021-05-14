WISER Plugins Overview
======================

WISER has three places where user-implemented plugins can be incorporated into
the application:

*   **"Tools" menu plugins** are shown in the WISER application menubar, under
    the "Tools" menu.  These plugins can implement their own workflows, and can
    access and manipulate application state.

*   **Context-menu plugins** are shown in context menus displayed in various
    parts of WISER, when the user right-clicks in an area, or uses other OS
    support to show a context menu.  Besides the capabilities of "Tools" menu
    plugins, context-menu plugins are also able to respond to the picking of
    specific objects in WISER, such as datasets, spectra, and Regions of
    Interest (ROIs).

*   **Custom band-math function plugins** extend the WISER band-math
    functionality with additional custom functions.  They are only used in
    this context.

The process of creating a WISER plugin is straightforward.  One must create a
subclass of the plugin base-class corresponding to the functionality to be
extended:

*   :class:`wiser.plugins.ToolsMenuPlugin` - "Tools" menu plugins
*   :class:`wiser.plugins.ContextMenuPlugin` - context-menu plugins
*   :class:`wiser.plugins.BandMathPlugin` - band-math plugins

Each of these classes has slightly different functionality to implement.

**Note:**  To implement a WISER plugin, you will need to be comfortable with
these libraries:

*   Python 3
*   Qt5 / PySide2 (for graphical UI interactions)
*   NumPy / AstroPy (for operations involving imaging spectroscopy data)

Development Environment
-----------------------

TODO:  Document.
