import os
import random
import string

from typing import Dict, List, Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import matplotlib
import numpy as np
from PIL import Image
import cv2

import math

def clear_widget(w: QWidget):
    # remove and delete any existing layout
    old_layout = w.layout()
    if old_layout is not None:
        while old_layout.count():
            item = old_layout.takeAt(0)
            # if it’s a widget, delete it
            if item.widget():
                item.widget().deleteLater()
            # if it’s a sub-layout, clear that too
            elif item.layout():
                clear_widget(item.layout().parentWidget())
        old_layout.deleteLater()

    # also remove any stray child widgets (Designer placeholder, etc)
    for child in w.findChildren(QWidget):
        child.setParent(None)

def get_plugin_fns(app_state):
    # Collect functions from all plugins.
    functions = {}
    for (plugin_name, plugin) in app_state.get_plugins().items():
        try:
            plugin_fns = plugin.get_bandmath_functions()

            # Make sure all function names are lowercase.
            for k in list(plugin_fns.keys()):
                lower_k = k.lower()
                if k != lower_k:
                    plugin_fns[lower_k] = plugin_fns[k]
                    del plugin_fns[k]

            # If any functions appear multiple times, make sure to
            # report a warning about it.
            for k in plugin_fns.keys():
                if k in functions:
                    print(f'WARNING:  Function "{k}" is defined ' +
                            f'multiple times (last seen in plugin {plugin_name})')

            functions.update(plugin_fns)
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

def cv2_rotate_scale_expand(img: np.ndarray,
                        angle: float,
                        scale: float = 1.0,
                        interp: int = 1,
                        mask_fill_value: float = 0
                        ) -> np.ndarray:
    """
    Rotate and scale an image array, expanding the output array
    so nothing gets clipped.

    Args:
    img    : HxW or HxWxC uint8/float32 array.
    angle  : rotation angle in degrees (positive = CCW).
    scale  : isotropic scale factor.
    interp : one of 'nearest','linear','cubic','lanczos'. Defaults to linear (1)

    Returns:
    Transformed array with dtype matching input.
    """
    # print(f"img array shape: {img.shape}")
    # print(f"img array type: {img.dtype}")
    _INTERPOLATIONS = {
        'nearest':  cv2.INTER_NEAREST,
        'linear':   cv2.INTER_LINEAR,
        'cubic':    cv2.INTER_CUBIC,
        'lanczos':  cv2.INTER_LANCZOS4,
    }
    # choose interpolation flag
    interp_flag = interp
    orig_mask = None
    if isinstance(img, np.ma.MaskedArray):
        orig_mask = img.mask
        img = img.filled(mask_fill_value)

    # 3. Build the rotation+scale matrix
    h, w = img.shape[:2]
    # print(f"h: {h}")
    # print(f"w: {w}")
    cx, cy = w/2, h/2
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # 4. Compute new canvas size so nothing is clipped
    abs_cos = abs(M[0,0]); abs_sin = abs(M[0,1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    # shift origin to centre result
    M[0,2] += (new_w/2 - cx)
    M[1,2] += (new_h/2 - cy)
    # print(f"new_w: {new_w}")
    # print(f"new_h: {new_h}")
    # print(f"interp: {interp}")
    # 5. Warp the image
    out = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=mask_fill_value
    )

    # 6. If there was a mask, warp it too and reapply
    if orig_mask is not None:
        # invert mask (True=masked) → valid=1, invalid=0
        valid = (~orig_mask).astype(np.uint8) * 255
        warped_valid = cv2.warpAffine(
            valid,
            M,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        warped_mask = ~(warped_valid.astype(bool))
        return np.ma.MaskedArray(out, mask=warped_mask)

    return out

def cv2_rotate_scale_expand_3D(img: np.ndarray,
                        angle: float,
                        scale: float = 1.0,
                        interp: str = 'linear',
                        mask_fill_value: float = 0
                        ) -> np.ndarray:
    """
    Rotate and scale an image array, expanding the output array
    so nothing gets clipped.

    Args:
    img    : HxW or HxWxC uint8/float32 array.
    angle  : rotation angle in degrees (positive = CCW).
    scale  : isotropic scale factor.
    interp : one of 'nearest','linear','cubic','lanczos'.

    Returns:
    Transformed array with dtype matching input.
    """
    _INTERPOLATIONS = {
        'nearest':  cv2.INTER_NEAREST,
        'linear':   cv2.INTER_LINEAR,
        'cubic':    cv2.INTER_CUBIC,
        'lanczos':  cv2.INTER_LANCZOS4,
    }
    # choose interpolation flag
    flag = _INTERPOLATIONS.get(interp, cv2.INTER_LINEAR)
    orig_mask = None
    if isinstance(img, np.ma.MaskedArray):
        orig_mask = img.mask
        img = img.filled(mask_fill_value)

    # 3. Build the rotation+scale matrix
    h, w = img.shape[1:3]
    cx, cy = w/2, h/2
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # 4. Compute new canvas size so nothing is clipped
    abs_cos = abs(M[0,0]); abs_sin = abs(M[0,1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    # shift origin to centre result
    M[0,2] += (new_w/2 - cx)
    M[1,2] += (new_h/2 - cy)

    # 5. Warp the image
    out = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=_INTERPOLATIONS.get(interp, cv2.INTER_LINEAR),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=mask_fill_value
    )

    # 6. If there was a mask, warp it too and reapply
    if orig_mask is not None:
        # invert mask (True=masked) → valid=1, invalid=0
        valid = (~orig_mask).astype(np.uint8) * 255
        warped_valid = cv2.warpAffine(
            valid,
            M,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        warped_mask = ~(warped_valid.astype(bool))
        return np.ma.MaskedArray(out, mask=warped_mask)

    return out


def pillow_rotate_scale_expand(
    arr: np.ndarray,
    angle: float,
    scale: float = 1.0,
    resample: str = 'bilinear',
) -> np.ndarray:
    """
    Rotate & scale an HxW or HxWxC array, expanding the output so nothing is clipped.

    Args:
    arr     : input array (uint8 or float) of shape (H,W) or (H,W,3/4).
    angle   : CCW rotation in degrees.
    scale   : uniform scale factor (1.0 = no change).
    resample: one of 'nearest','bilinear','bicubic','lanczos'.

    Returns:
    Transformed array, same dtype as input (floats are re-normalized).
    """
    # map human-readable names to Pillow resampling filters
    _RESAMPLE_MODES = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
    }
    # pick filter
    mode = _RESAMPLE_MODES.get(resample, Image.BILINEAR)

    # if float, normalize to [0,255] and cast
    is_float = np.issubdtype(arr.dtype, np.floating)
    if is_float:
        lo, hi = arr.min(), arr.max()
        arr_uint8 = ((arr - lo) / (hi - lo or 1) * 255).astype(np.uint8)
    else:
        arr_uint8 = arr
    print(f"arr_uint8.shape: {arr_uint8.shape}")
    print(f"arr.shape: {arr.shape}")
    # build PIL image
    img = Image.fromarray(arr_uint8.T)
    print(f"img.size: {img.size}")
    # 1) scale
    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), resample=mode)

    # 2) rotate + expand
    img = img.rotate(angle, resample=mode, expand=True)

    out = np.array(img)
    print(f"out.shape: {out.shape}")
    # if original was float, map back to original range
    if is_float:
        print(f"IS FLOAT")
        lo, hi = arr.min(), arr.max()
        out = out.astype(np.float32) / 255 * (hi - lo or 1) + lo

    return out

def rotate_scale_geotransform(gt, theta, scale, width, height, pivot):
    '''
    First we rotate the geo transform's res and rot like this:
    new_xres = xres_cos - xrot_sin
    new_xrot = xrot_cos + xres_sin

    new_yres = yres_cos - yrot_sin
    new_yrot = yrot_cos + yres_sin

    Then to get the new UL point by rotating all of the edge points by a pivot point. Finding the new minx and max x then rotate back
    '''
    ulx, xres, xrot, uly, yrot, yres = gt
    px, py = pivot
    rad = math.radians(theta)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    new_xres = xres * cos_t - xrot * sin_t
    new_xrot = xrot * cos_t + xres * sin_t

    new_yres = yres * cos_t - yrot * sin_t
    new_yrot = yrot * cos_t + yres * sin_t

    # If the scale is >1 (like 2) then we have more pixels so the resolution would be half as much
    # If the scale is <1 then we are downsampling so the resolution would be more per pixel (less pixels)
    new_xres_scaled = new_xres / scale
    new_xrot_scaled = new_xrot / scale

    new_yres_scaled = new_yres / scale
    new_yrot_scaled = new_yrot / scale


        
    def rot(v):
        x, y = v
        # shift so pivot is at (0,0)
        dx, dy = x - px, y - py
        # rotate about origin
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        # shift back
        return (rx + px, ry + py)

    def rot_inverse(v):
        x, y = v
        dx, dy = x - px, y - py
        # inverse rotation = rotate by –θ (or use transpose)
        ix = dx * cos_t + dy * sin_t
        iy = -dx * sin_t + dy * cos_t
        return (ix + px, iy + py)

    def pixel_to_spatial(pixel_x, pixel_y):
        spatial_x = ulx + xres * pixel_x + xrot * pixel_y
        spatial_y = uly + yres * pixel_y + yrot * pixel_x
        return (spatial_x, spatial_y)
    
    upper_left = pixel_to_spatial(0, 0)
    upper_right = pixel_to_spatial(width, 0)
    bottom_left = pixel_to_spatial(0, height)
    bottom_right = pixel_to_spatial(width, height)

    rotated_ul = rot(upper_left)
    rotated_ur = rot(upper_right)
    rotated_bl = rot(bottom_left)
    rotated_br = rot(bottom_right)

    rotated_min_x = min(rotated_ul[0], rotated_ur[0], rotated_bl[0], rotated_br[0])
    rotated_min_y = min(rotated_ul[1], rotated_ur[1], rotated_bl[1], rotated_br[1])

    new_spatial_ul = rot_inverse((rotated_min_x, rotated_min_y))

    return (new_spatial_ul[0], new_xres_scaled, new_xrot_scaled, new_spatial_ul[1], new_yrot, new_yrot_scaled)

def make_into_help_button(help_btn: QToolButton, link: str, tooltip_message: str = None):
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("App instance is not running!")
    help_icon = app.style().standardIcon(QStyle.SP_MessageBoxQuestion)
    help_btn.setIcon(help_icon)
    if tooltip_message is None:
        help_btn.setToolTip("Click for help")
    else:
        help_btn.setToolTip(tooltip_message)


    help_btn.clicked.connect(
        lambda: QDesktopServices.openUrl(
            QUrl(f"{link}")
        )
    )
