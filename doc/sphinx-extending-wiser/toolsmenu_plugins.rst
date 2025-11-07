Tools-Menu Plugins
==================

Tools-menu plugins allow you to add custom actions to the Tool Menu located in the top navigation bar.
Each plugin appears using the name you provide, and selecting it triggers the function you have linked
to that action.

If you have actions that are used less frequently, consider organizing them into a submenu to keep the
main Tool Menu concise and easy to navigate.

.. autoclass:: wiser.plugins.ToolsMenuPlugin
    :members:

Example Tools-Menu Plugin
-------------------------

Okay, lets get into an example plugin.

.. literalinclude:: ../../src/example_plugins/tool_plugin.py
   :language: python
