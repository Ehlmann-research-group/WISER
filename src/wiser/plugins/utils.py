import importlib
import logging

from typing import Any

from .types import Plugin, ToolsMenuPlugin, ContextMenuPlugin, BandMathPlugin


logger = logging.getLogger(__name__)


def is_plugin(obj: Any):
    '''
    Returns True if the specified argument is a recognized plugin type; that is,
    ToolsMenuPlugin, ContextMenuPlugin, or BandMathPlugin.
    '''
    return (isinstance(obj, ToolsMenuPlugin) or
            isinstance(obj, ContextMenuPlugin) or
            isinstance(obj, BandMathPlugin))


def instantiate(fully_qualified_class_name: str) -> Plugin:
    '''
    Given the fully qualified name of a class, attempt to instantiate an object
    of that type.
    '''
    logger.debug(f'Instantiating plugin class "{fully_qualified_class_name}"')

    parts = fully_qualified_class_name.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]

    module_obj = importlib.import_module(module_name)
    class_obj = getattr(module_obj, class_name)
    return class_obj()
