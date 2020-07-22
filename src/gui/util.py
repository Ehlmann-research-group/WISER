import string
from typing import Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import gui.generated.resources


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


def make_dockable(widget: QWidget, title: str, parent: Optional[QWidget]) -> QDockWidget:
    dockable = QDockWidget(title, parent=parent)
    dockable.setWidget(widget)
    return dockable


class PainterWrapper:
    '''
    This class provides a context manager so that a QPainter object can be
    managed by a Python "with" statement.

    For more information see:
    https://docs.python.org/3/reference/datamodel.html#context-managers
    '''

    def __init__(self, _painter: QPainter):
        if _painter is None:
            raise ValueError('_painter cannot be None')

        self._painter: QPainter = _painter

    def __enter__(self) -> QPainter:
        return self._painter

    def __exit__(self, type, value, traceback) -> bool:
        self._painter.end()

        # If an exception occurred within the with block, reraise it by
        # returning False.  Otherwise return True.
        return traceback is None


def get_painter(widget: QWidget) -> PainterWrapper:
    '''
    This helper function makes a QPainter for writing to the specified QWidget,
    and then wraps it with a PainterWrapper context manager.  It is intended to
    be used with the Python "with" statement, like this:

    with get_painter(some_widget) as painter:
        painter.xxxx()  # Draw stuff
    '''
    return PainterWrapper(QPainter(widget))


def make_filename(s: str) -> str:
    '''
    This helper function makes a filename out of a string
    '''

    # Apply this filter to collapse any whitespace characters down to one space.
    # If the string is entirely whitespace, the result will be an empty string.
    s = ' '.join(s.strip().split())

    if len(s) == 0:
        raise ValueError('Cannot make a filename out of an empty string')

    # Filter out any characters that are not alphanumeric, or some basic
    # punctuation and spaces.
    result = ''
    for ch in s:
        if ch in string.ascii_letters or ch in string.digits or ch in ['_', '-', '.', ' ']:
            result += ch

    return result
