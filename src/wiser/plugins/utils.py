import importlib
import logging

from typing import Any, Optional

from .types import Plugin, ToolsMenuPlugin, ContextMenuPlugin, BandMathPlugin, BatchProcessingPlugin

from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QWidget


logger = logging.getLogger(__name__)


def is_plugin(obj: Any):
    '''
    Returns True if the specified argument is a recognized plugin type; that is,
    ToolsMenuPlugin, ContextMenuPlugin, or BandMathPlugin.
    '''
    return (isinstance(obj, ToolsMenuPlugin) or
            isinstance(obj, ContextMenuPlugin) or
            isinstance(obj, BandMathPlugin) or
            isinstance(obj, BatchProcessingPlugin))


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


def load_ui_file(ui_file_path: str, parent: Optional[QWidget] = None) -> QWidget:
    '''
    Given the path and filename of a QtDesigner .ui file, this helper function
    uses a QUiLoader to instantiate the widget for the .ui file.

    To load a .ui file relative to the currently-executing Python module, do
    something like this:

        path = os.path.join(os.path.dirname(__file__), 'some_widget.ui')
        widget = load_ui_file(path)
    '''
    # https://doc.qt.io/archives/qtforpython-5.12/PySide2/QtUiTools/QUiLoader.html
    logger.info(f'Loading Qt .ui file:  "{ui_file_path}"')
    f = QFile(ui_file_path)
    if not f.open(QFile.ReadOnly):
        raise IOError(f'Cannot open {ui_file_path}: {f.errorString()}')

    loader = QUiLoader()
    widget = loader.load(f, parentWidget=parent)
    f.close()

    if not widget:
        raise IOError(f'Cannot load {ui_file_path}: {loader.errorString()}')

    return widget
