import enum
import importlib

from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide2.QtWidgets import QMenu

from bandmath import BandMathValue
from bandmath.functions import BandMathFunction # Type annotation


class ContextMenuType(enum.Enum):
    '''
    This enumeration specifies the kind of context-menu event that occurred,
    so that plugins know what items to add to the menu.
    '''

    # Context-menu display in a raster-view, which probably is showing a
    # dataset.  The current dataset is passed to the plugin.
    RASTER_VIEW = 1

    # Context-menu display in the spectrum-plot window.
    SPECTRUM_PLOT = 2

    # A specific dataset was picked.  This may not be in the context of a
    # raster-view window, e.g. if the user right-clicks on a dataset in the info
    # viewer.
    DATASET_PICK = 10

    # A specific spectrum was picked.  The spectrum is passed to the plugin.
    SPECTRUM_PICK = 11

    # A specific ROI was picked.  The ROI is passed, along with the current
    # dataset (if available).
    ROI_PICK = 12


class Plugin:
    ''' The base type for all WISER plugins. '''
    pass


class ToolsMenuPlugin(Plugin):
    '''
    This is the base type for plugins that integrate into the WISER "Tools"
    application-menu.
    '''

    def __init__(self):
        super().__init__()


    def add_tool_menu_items(self, tool_menu: QMenu) -> None:
        '''
        This method is called by WISER to allow plugins to add menu actions or
        submenus into the Tools application menu.

        If a plugin provides multiple actions, the developer has several
        choices.  If all actions are useful and expected to be invoked
        regularly, the actions can be added directly to the Tools menu.  If
        some actions are used much less frequently, it is recommended that
        these actions be put into a submenu, to keep the Tools menu from
        becoming too cluttered.

        Use QMenu.addAction() to add individual actions, or QMenu.addMenu() to
        add sub-menus to the Tools menu.
        '''
        pass


class ContextMenuPlugin(Plugin):
    '''
    This is the base type for plugins that integrate into WISER pop-up context
    menus.
    '''

    def __init__(self):
        super().__init__()

    def add_context_menu_items(self, context_type: ContextMenuType,
            context_menu: QMenu, context: Dict[str, Any]) -> None:
        pass


class BandMathPlugin(Plugin):
    '''
    This is the base type for plugins that provide custom band-math functions.
    '''

    def __init__(self):
        super().__init__()

    def get_bandmath_functions(self) -> Dict[str, BandMathFunction]:
        '''
        This method returns a dictionary of all band-math functions defined by
        the plugin.  The
        '''
        pass


def is_plugin(obj: Any):
    return (isinstance(obj, ToolsMenuPlugin) or
            isinstance(obj, ContextMenuPlugin) or
            isinstance(obj, BandMathPlugin))


def instantiate(fully_qualified_class_name: str) -> Plugin:
    '''
    Given the fully qualified name of a class, attempt to instantiate an object
    of that type.
    '''
    parts = fully_qualified_class_name.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]

    module_obj = importlib.import_module(module_name)
    class_obj = getattr(module_obj, class_name)
    return class_obj()
