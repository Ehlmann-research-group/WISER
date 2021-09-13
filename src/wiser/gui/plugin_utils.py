import logging

from typing import Any, Dict
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from .app_state import ApplicationState


from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .app_state import ApplicationState
from wiser import plugins


logger = logging.getLogger(__name__)


def add_plugin_context_menu_items(app_state: ApplicationState,
        context_type: plugins.ContextMenuType, menu: QMenu, **kwargs):
    '''
    Helper function to add plugin-provided context menu items to a menu.  The
    function will handle and log any exceptions raised by plugins during the
    operation.

    Additional context values can be specified with keyword arguments to the
    function.  These will be packaged into a context and passed to the plugin.
    Each plugin is handed a copy of the context, so that if one plugin mutates
    its context, no other plugins will see this.
    '''

    for (plugin_name, plugin) in app_state.get_plugins().items():
        if isinstance(plugin, plugins.ContextMenuPlugin):
            # Make a copy of the context, so that plugins can misbehave and
            # mutate the context without affecting each other.
            context = kwargs.copy()
            context['wiser'] = app_state

            try:
                # Call the plugin!
                plugin.add_context_menu_items(context_type, menu, context)
            except:
                logger.exception(f'Plugin {plugin_name} raised an ' +
                    f'exception while adding {context_type} context-menu items.')
