
Extending WISER
===============

WISER's functionality can be extended through a **plugin system** that exposes
three distinct integration points:

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`Tools-Menu Plugins <toolsmenu_plugins>`
     - Add named actions to the WISER **Tools** menu. Useful for launching
       custom workflows, dialogs, or analyses that operate on the currently
       loaded data.
   * - :doc:`Context-Menu Plugins <ctxmenu_plugins>`
     - Appear when the user right-clicks on specific objects (datasets,
       spectra, ROIs). Receive the selected object as context, so they can
       act on exactly what the user picked.
   * - :doc:`Band-Math Plugins <bandmath_plugins>`
     - Register custom functions in the WISER **Band Math** dialog, extending
       the built-in set of spectral operations with your own algorithms.

All plugins are plain Python classes that subclass one of the three base types
defined in ``wiser.plugins``. No internal build step is required — point WISER
at a directory containing your plugin's source (or its virtualenv
``site-packages``) via **Settings → Plugins**, and WISER will load it at
startup.

A complete worked example of each plugin type is available in
``src/example_plugins/`` in the WISER repository.
Community-contributed plugins are hosted in the
`WISER Plugin Repository <https://github.com/Ehlmann-research-group/WISER-Plugin-Repository>`_.

.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    plugins
    plugin_dependencies

.. toctree::
    :maxdepth: 1
    :caption: Plugin Types

    toolsmenu_plugins
    ctxmenu_plugins
    bandmath_plugins
    ui_plugins

.. toctree::
    :maxdepth: 1
    :caption: API Reference

    wiser_state
    supporting_types
    plugin_utilities

.. toctree::
    :maxdepth: 1
    :caption: Examples & Community

    more_example_plugins
    plugin_repository

