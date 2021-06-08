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
        context_type: plugins.ContextMenuType, menu: QMenu, context: Dict[str, Any]):
    '''
    Helper function to add plugin-provided context menu items to a menu.  The
    function will handle and log any exceptions raised by plugins during the
    operation.
    '''
    for (plugin_name, plugin) in app_state.get_plugins().items():
        if isinstance(plugin, plugins.ContextMenuPlugin):
            try:
                plugin.add_context_menu_items(context_type, menu, context)
            except:
                logger.exception(f'Plugin {plugin_name} raised an ' +
                    f'exception while adding {context_type} context-menu items.')
