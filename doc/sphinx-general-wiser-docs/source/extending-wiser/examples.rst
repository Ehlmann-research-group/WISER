Examples & Community
====================

.. _example-ndi-plugin:

Normalized Difference Index Plugin
-----------------------------------

This combined example uses both a Tools Menu plugin and a Context Menu plugin
to implement a normalized difference index workflow. The Tools Menu plugin
creates a new index definition; the Context Menu plugin applies it to a
displayed image in one click.

The example also demonstrates how to read raster data from WISER's
application state and write a new derived dataset back into it.

.. literalinclude:: ./resources/example_tools_ctx_plugin.py
   :language: python

Plugin Repository
-----------------

The `WISER Plugin Repository <https://github.com/Ehlmann-research-group/WISER-Plugin-Repository>`_
hosts community-contributed plugins that extend WISER with additional
remote sensing workflows and analysis tools.

**For users** — the repository README covers:

- Browsing available plugins
- Installing a plugin and its conda environment
- Enabling a plugin in WISER via **Settings → Plugins**

**For contributors** — to submit a plugin, see the repository's
`CONTRIBUTING.md <https://github.com/Ehlmann-research-group/WISER-Plugin-Repository/blob/main/CONTRIBUTING.md>`_
and `plugin specification <https://github.com/Ehlmann-research-group/WISER-Plugin-Repository/blob/main/plugin-spec.md>`_.

