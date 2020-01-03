from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


def add_toolbar_action(toolbar, icon_path, text, parent, shortcut=None, before=None):
    '''
    A helper function to set up a toolbar action using the common configuration
    used for these actions.
    '''
    act = QAction(QIcon(icon_path), text, parent)

    if shortcut is not None:
        act.setShortcuts(shortcut)

    if before is None:
        toolbar.addAction(act)
    else:
        toolbar.insertAction(before, act)

    return act


def make_dockable(widget, title, parent):
    dockable = QDockWidget(title, parent=parent)
    dockable.setWidget(widget)
    return dockable


class PainterWrapper:
    def __init__(self, _painter):
        self._painter = _painter

    def __enter__(self):
        return self._painter

    def __exit__(self, type, value, traceback):
        self._painter.end()

        # If an exception occurred within the with block, reraise it by
        # returning False.  Otherwise return True.
        return traceback is None


def get_painter(widget):
    return PainterWrapper(QPainter(widget))
