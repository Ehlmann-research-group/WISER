from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


def add_toolbar_action(toolbar, icon_path, text, parent, shortcut=None):
    '''
    A helper function to set up a toolbar action using the common configuration
    used for these actions.
    '''
    act = QAction(QIcon(icon_path), text, parent)

    if shortcut is not None:
        act.setShortcuts(shortcut)

    toolbar.addAction(act)
    return act


def make_dockable(widget, title, parent):
    dockable = QDockWidget(title, parent=parent)
    dockable.setWidget(widget)
    return dockable
