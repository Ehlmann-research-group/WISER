More Example Plugins
====================

This section provides additional example plugins to demonstrate various capabilities of the WISER
plugin system. These examples build upon the basic concepts introduced in each plugin section.

Normalized Difference Index Example Plugin
------------------------------------------

This plugin uses both a Tools Menu plugin and a Context Menu plugin to create the desired functionality.
You use the Tool Menu plugin to create a new normalized difference index and the context menu plugin
to use it on a displayed image quickly.

This plugin also teaches you how to get data from WISER and add data back into WISER.

.. literalinclude:: ./resources/example_tools_ctx_plugin.py
   :language: python
