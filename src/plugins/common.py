import enum
import importlib

from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide2.QtWidgets import QMenu

from bandmath import BandMathValue


class ItemPickType(enum.IntFlag):
    DATASET_PICK = 1
    SPECTRUM_PICK = 2
    ROI_PICK = 4


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

    def get_item_pick_type(self) -> ItemPickType:
        '''
        If the plugin is a context-menu plugin, this method reports the kind(s)
        of items that the plugin wants to have context-menu actions on.
        '''
        pass

    def add_context_menu_items(self, context_menu: QMenu) -> None:
        pass


class BandMathPlugin(Plugin):

    def __init__(self):
        super().__init__()

    def get_bandmath_functions(self) -> Dict[str, Callable[[List[BandMathValue]], BandMathValue]]:
        '''
        If the plugin is a band-math plugin, this method returns all band-math
        functions defined by the plugin.
        '''
        pass


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
