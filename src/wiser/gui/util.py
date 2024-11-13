import os
import random
import string

from typing import Dict, List, Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import matplotlib
import numpy as np

import wiser.gui.generated.resources

def get_plugin_fns(app_state):
    # Collect functions from all plugins.
    functions = {}
    print("showing bandmath dialog")
    for (plugin_name, plugin) in app_state.get_plugins().items():
        print(f"Found bandmath plugin: {plugin}")
        try:
            print(f"It's actually a plugin!")
            plugin_fns = plugin.get_bandmath_functions()
            print(f"plugin_fns: {plugin_fns}")

            # Make sure all function names are lowercase.
            for k in list(plugin_fns.keys()):
                lower_k = k.lower()
                if k != lower_k:
                    plugin_fns[lower_k] = plugin_fns[k]
                    del plugin_fns[k]
            print(f"plugin_fns after: {plugin_fns}")

            # If any functions appear multiple times, make sure to
            # report a warning about it.
            for k in plugin_fns.keys():
                if k in functions:
                    print(f'WARNING:  Function "{k}" is defined ' +
                            f'multiple times (last seen in plugin {plugin_name})')
            print(f"plugin_fns final: {plugin_fns}")

            functions.update(plugin_fns)
            print(f"functions: {functions}")
        except:
            pass
    return functions

def delete_all_files_in_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # List all files in the directory
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Check if it's a file and not a directory
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)  # Delete the file
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            else:
                print(f"Skipping: {file_path} (not a file)")
    else:
        print(f"Directory {folder_path} does not exist.")

def scale_qpoint_by_float(point: QPoint, scale: float):
    return QPoint(float(point.x() * scale), float(point.y() * scale))

def str_or_none(s: Optional[str]) -> str:
    '''
    Formats an optional string for logging.  If the argument is a string then
    the function returns the string in double-quotes.  If the argument is
    ``None`` then the function returns the string ``'None'``.
    '''
    if s is not None:
        return f'"{s}"'
    else:
        return 'None'


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
    This helper function makes a filename out of a string, following these
    rules:
    *   Any letters or numbers are left unchanged.
    *   Any sequences of whitespace characters (spaces, tabs and newlines) are
        collapsed down to a single space character.
    *   These punctuation characters are left as-is:  '-', '_', '.'.  All other
        punctuation characters are removed.

    If the function is given a string consisting of only whitespace, or an
    empty string, it will throw a ValueError.
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


def get_matplotlib_colors():
    '''
    Generates a list of all recognized matplotlib color names, which are
    suitable for displaying graphical plots.

    The definition of "suitable for displaying graphical plots" is currently
    that the color be dark enough to show up on a white background.
    '''
    names = matplotlib.colors.get_named_colors_mapping().keys()
    colors = []
    for name in names:
        if len(name) <= 1:
            continue

        if name.find(':') != -1:
            continue

        # Need to exclude colors that are too bright to show up on the white
        # background, so multiply all the components together and see if it's
        # "dark enough".
        # TODO(donnie):  May want to do this with HSV colorspace.
        rgba = matplotlib.colors.to_rgba_array(name).flatten()
        prod = np.prod(rgba)
        if prod >= 0.3:
            continue

        # print(f'Color {name} = {rgba} (type is {type(rgba)})')
        colors.append(name)

    return colors


def get_random_matplotlib_color(exclude_colors: List[str] = []) -> str:
    '''
    Returns a random matplotlib color name from the available matplotlib colors
    returned by the get_matplotlib_colors().
    '''
    all_names = get_matplotlib_colors()
    while True:
        name = random.choice(all_names)
        if name not in exclude_colors:
            return name


def get_color_icon(color_name: str, width: int = 16, height: int = 16) -> QIcon:
    '''
    Generate a QIcon of the specified color and optional size.  If the size is
    unspecified, a 16x16 icon is generated.

    *   color_name is a string color name recognized by matplotlib, which
        includes strings of the form "#RRGGBB" where R, G and B are hexadecimal
        digits.

    *   width is the icon's width in pixels, and defaults to 16.

    *   height is the icon's height in pixels, and defaults to 16.
    '''
    rgba = matplotlib.colors.to_rgba_array(color_name).flatten()

    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor.fromRgbF(rgba[0], rgba[1], rgba[2]))

    return QIcon(QPixmap.fromImage(img))


def clear_treewidget_selections(tree_widget: QTreeWidget) -> None:
    '''
    Given a QTreeWidget object, this function clears all selections of
    tree widget items.
    '''

    def clear_treewidget_item_selections(item: QTreeWidgetItem) -> None:
        if item.isSelected():
            item.setSelected(False)

        for i in range(item.childCount()):
            clear_treewidget_item_selections(item.child(i))

    for i in range(tree_widget.topLevelItemCount()):
        item = tree_widget.topLevelItem(i)
        clear_treewidget_item_selections(item)


def generate_unused_filename(basename: str, extension: str) -> str:
    '''
    Generates a filename that is currently unused on the filesystem.  The
    base name (including path) and extension to use are both specified as
    arguments to this function.

    If "basename.extension" is available as a filename, the function will
    return that string.

    If it is already used, the function will generate a filename of the form
    "basename_i.extension", where i starts at 1, and is incremented until
    the filename is unused.
    '''
    filename = f'{basename}.{extension}'
    if os.path.exists(filename):
        i = 1
        while True:
            filename = f'{basename}_{i}.{extension}'
            if not os.path.exists(filename):
                break

            i += 1

    return filename
