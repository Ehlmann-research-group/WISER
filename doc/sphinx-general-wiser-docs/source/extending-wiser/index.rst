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
defined in ``wiser.plugins``:

- :class:`wiser.plugins.ToolsMenuPlugin` — Tools menu plugins
- :class:`wiser.plugins.ContextMenuPlugin` — context-menu plugins
- :class:`wiser.plugins.BandMathPlugin` — band-math plugins

No internal build step is required — point WISER at a directory containing
your plugin's source (or its virtualenv ``site-packages``) via
**Settings → Plugins**, and WISER will load it at startup.

A complete worked example of each plugin type is available in
``src/example_plugins/`` in the WISER repository.

**Prerequisites:** plugin development requires comfort with Python 3,
Qt5/PySide2 (for any graphical UI), and NumPy/AstroPy (for spectral data
operations).

Setting Up Plugin Paths
-----------------------

WISER ships only the libraries needed for visualization. If your plugin
requires additional third-party dependencies, configure plugin paths in
**Settings → Plugins**:

.. image:: images/plugin_config.png

In the **top half** of the config window enter the paths that WISER should
search. This will either be the path to the plugin's source directory (for
example, ``~/Projects/WISER-Plugins``) or the path to its installed
dependencies (for example,
``~/Projects/WISER-Plugins/venv/lib/python3.9/site-packages``).

A typical virtualenv-based setup looks like this:

.. code-block:: console

    # From the ~/Projects/WISER-Plugins directory
    virtualenv venv
    source venv/bin/activate
    pip install opencv-python

Point WISER at the ``site-packages`` directory of that environment so it can
find the installed packages.

In the **bottom half** of the config window enter the fully qualified class
names of your plugin classes. The **Verify All** button attempts to load and
initialize every registered plugin and reports any errors — check the WISER
logs for details if a plugin fails to load.

Plugin Dependencies
-------------------

WISER ships pinned conda environments; it is good practice to match your
plugin's dependency versions against the version of WISER you are targeting.
Mismatched versions usually work but can produce unexpected behaviour.

The dependency lists for each release are available as downloadable text
files below.

.. raw:: html

    <style>
      .release {
        border: 3px solid #000;
        border-radius: 2px;
        margin: 12px 0;
        background: #fff;
      }
      .release > summary {
        list-style: none;
        cursor: pointer;
        padding: 14px 16px;
        font-weight: 700;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
      }
      .release > summary::before {
        content: '\25B6';
        display: inline-block;
        margin-right: 10px;
        transform: translateY(1px);
      }
      .release[open] > summary::before {
        content: '\25BC';
      }
      .release + .release {
        border-top: 3px solid #000;
      }
      .release > div {
        border-top: 3px solid #000;
        padding: 12px 16px;
        background: #fafafa;
      }
      .release p { margin: 0.3em 0; }
      .release a { text-decoration: underline; }
    </style>

.. raw:: html

    <details class="release">
      <summary>Release 1.3b1 Dependencies</summary>
      <div>

Windows dependencies: :download:`downloadable text file <resources/rel-1.3b1/windows-dependencies.txt>`

ARM Mac dependencies: :download:`downloadable text file <resources/rel-1.3b1/arm-mac-dependencies.txt>`

Intel Mac dependencies: :download:`downloadable text file <resources/rel-1.3b1/intel-mac-dependencies.txt>`

.. raw:: html

      </div>
    </details>

Logging
-------

Plugins should use Python's standard
`logging <https://docs.python.org/3/howto/logging.html>`_ module to write to
the WISER runtime logs. Log locations by platform:

- **Linux:** ``~/.wiser/wiser.log``
- **macOS:** ``~/Library/WISER/wiser.log``
- **Windows:** ``%USERPROFILE%\\AppData\\Local\\WISER\\wiser.log``

WISER provides a ``@log_exceptions`` decorator that automatically logs any
exception raised inside a decorated plugin method:

.. autodecorator:: wiser.plugins.log_exceptions

.. toctree::
   :hidden:
   :maxdepth: 1

   toolsmenu_plugins
   ctxmenu_plugins
   bandmath_plugins
   ui_plugins
   wiser_state
   supporting_types
   plugin_utilities
   examples
