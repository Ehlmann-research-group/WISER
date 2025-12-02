import os
import random
import string

from typing import Dict, List, Optional, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import matplotlib
import numpy as np
from numba import types, prange
from scipy.interpolate import interp1d as _scipy_interp1d
from PIL import Image
import cv2
from astropy import units as u

import math
import enum

from wiser.utils.numba_wrapper import numba_njit_wrapper


class StateChange(enum.Enum):
    """
    Whether a spectrum was added, removed or edited
    in the spectrum plot
    """

    ITEM_ADDED = 1
    ITEM_EDITED = 2
    ITEM_REMOVED = 3


def compute_resid(target_image_cr, scale, ref_spectrum_cr):
    pass


compute_resid_sig = types.float32[:, :, :](  # return type
    types.float32[:, :, :],  # target_image_cr
    types.float32[:, :],  # scale
    types.float32[:],  # ref_spectrum_cr
)


@numba_njit_wrapper(
    non_njit_func=compute_resid,
    signature=compute_resid_sig,
    parallel=True,
    cache=True,
)
def compute_resid_numba(target_image_cr, scale2d, ref1d):
    rows, cols, bands = target_image_cr.shape
    out = np.empty_like(target_image_cr, dtype=np.float32)

    for k in prange(bands):
        # 2D slice of output for band k
        for i in prange(rows):
            for j in range(cols):
                out[i, j, k] = target_image_cr[i, j, k] - scale2d[i, j] * ref1d[k]

    return out


def nanmean_last_axis_3d_numpy(a: np.ndarray) -> np.ndarray:
    """
    NumPy version: compute nanmean over the last axis of a 3D array.

    Parameters
    ----------
    a : np.ndarray
        3D array (will be treated as float).

    Returns
    -------
    out : np.ndarray
        2D array of nanmeans over the last axis.
    """
    # axis = -1 means "last axis"
    return np.nanmean(a, axis=-1).astype(np.float32)


mean3d_last_axis_sig = types.float32[:, :](types.float32[:, :, :])


@numba_njit_wrapper(
    non_njit_func=nanmean_last_axis_3d_numpy,
    signature=mean3d_last_axis_sig,
    cache=True,
)
def nanmean_last_axis_3d(a):
    """
    Compute the nanmean over the last axis (axis=2) of a 3D float32 array.

    Parameters
    ----------
    a : float32[:, :, :]
        Input 3D array.

    Returns
    -------
    out : float32[:, :]
        2D array where out[i, j] is the mean of a[i, j, :]
        ignoring NaNs. If all values along the last axis are NaN,
        out[i, j] will be NaN (matching np.nanmean behavior).
    """
    n0 = a.shape[0]
    n1 = a.shape[1]
    n2 = a.shape[2]

    out = np.empty((n0, n1), dtype=np.float32)

    for i in range(n0):
        for j in range(n1):
            total = 0.0
            count = 0
            for k in range(n2):
                val = a[i, j, k]
                # ignore NaNs
                if not np.isnan(val):
                    total += val
                    count += 1

            if count > 0:
                out[i, j] = total / count
            else:
                # all-NaN slice -> NaN, like np.nanmean
                out[i, j] = np.float32(np.nan)

    return out


def dot3d(a, b):
    """
    Dot product of a 3D array of shape (y, x, b) and
    a 1D array of shape (b,). Returns a 2D array shaped (y, x).
    """
    # b_extended = b[np.newaxis, np.newaxis, :]  # reshape b to (b, 1, 1)
    # rows = a.shape[0]
    # cols = a.shape[1]
    # b_extended = np.repeat(b_extended, repeats=rows, axis=0)
    # b_extended = np.repeat(b_extended, repeats=cols, axis=1)
    print(f"!@#$, a.shape: {a.shape}, b_extended.shape: {b.shape}")
    return np.dot(a, b)


dot3d_sig = types.float32[:, :](types.float32[:, :, :], types.float32[:])


@numba_njit_wrapper(
    non_njit_func=dot3d,
    signature=dot3d_sig,
)
def dot3d_numba(a, b):
    """
    Dot product of a 3D array of shape (y, x, b)
    and a 1D array of shape (b,). Returns a 2D array
    shaped (y, x).
    """
    y = a.shape[0]
    x = a.shape[1]
    nb = a.shape[2]

    out = np.empty((y, x), dtype=np.float32)

    for i in prange(y):
        for j in range(x):
            s = 0.0
            for k in range(nb):
                s += a[i, j, k] * b[k]
            out[i, j] = s

    return out


def interp1d_monotonic(x, y, x_new):
    """Perform linear interpolation on strictly increasing `x` and `x_new`.

    This function wraps :func:`scipy.interpolate.interp1d` to mimic the
    behavior of a monotonic, single-pass linear interpolation routine.
    Both `x` and `x_new` are assumed to be strictly increasing. When
    `extrapolate` is False, values of `x_new` that fall outside the
    domain ``[x[0], x[-1]]`` are assigned `fill_value`. When
    `extrapolate` is True, values outside the domain are linearly
    extrapolated using the end segments of the data.

    Args:
        x (np.ndarray):
            A 1D float array of strictly increasing x-coordinates.
        y (np.ndarray):
            A 1D float array of values corresponding to `x`. Must have
            the same length as `x`.
        x_new (np.ndarray):
            A 1D float array of strictly increasing query points at
            which interpolation and extrapolation is
            evaluated.

    Returns:
        np.ndarray:
            A 1D float array of interpolated and possibly extrapolated
            values evaluated at each point in `x_new`.
    """
    f = _scipy_interp1d(
        x,
        y,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )

    # Evaluate at x_new
    out = f(x_new)
    return np.asarray(out, dtype=float)


def slice_to_bounds_3D(
    spectrum_arr: np.ndarray,
    wvls: np.ndarray,
    bad_bands: np.ndarray,
    min_wvl: np.float32,
    max_wvl: np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a 3D spectrum array along the band axis using wavelength bounds.

    The input cube is assumed to have shape (b, y, x), where the first axis
    is the spectral (band) dimension. `wvls` and `bad_bands` are 1D arrays
    of length `b`. A boolean mask is built over the band axis based on the
    wavelength bounds, and that mask is applied to `spectrum_arr`, `wvls`,
    and `bad_bands`.

    Args:
        spectrum_arr (np.ndarray):
            3D array of shape (b, y, x), where `b` is the number of bands.
        wvls (np.ndarray):
            1D float array of shape (b,), the wavelength for each band.
        bad_bands (np.ndarray):
            1D boolean array of shape (b,), flags for each band.
        min_wvl (np.float32):
            Minimum wavelength to keep. If None (in the pure Python version),
            no lower bound is applied.
        max_wvl (np.float32):
            Maximum wavelength to keep. If None (in the pure Python version),
            no upper bound is applied.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - spectrum_arr_sliced: 3D array of shape (b_kept, y, x)
            - wvls_sliced: 1D array of shape (b_kept,)
            - bad_bands_sliced: 1D boolean array of shape (b_kept,)
    """
    if spectrum_arr.ndim != 3:
        raise ValueError(f"spectrum_arr must be 3D (b, y, x); got shape {spectrum_arr.shape}")

    b, _, _ = spectrum_arr.shape

    if wvls.ndim != 1 or bad_bands.ndim != 1 or wvls.shape[0] != b or bad_bands.shape[0] != b:
        raise ValueError(
            "Shape mismatch: "
            f"spectrum_arr has shape {spectrum_arr.shape}, "
            f"wvls has shape {wvls.shape}, "
            f"bad_bands has shape {bad_bands.shape}"
        )

    mask = np.ones(wvls.shape, dtype=np.bool_)
    if min_wvl is not None:
        mask &= wvls >= min_wvl
    if max_wvl is not None:
        mask &= wvls <= max_wvl

    # Apply mask along band axis
    return spectrum_arr[mask, :, :], wvls[mask], bad_bands[mask]


# Numba signature for 3D version:
slice_bounds_3d_sig = types.Tuple(
    (
        types.float32[:, :, :],  # sliced spectrum_arr
        types.float32[:],  # sliced wvls
        types.boolean[:],  # sliced bad_bands
    )
)(
    types.float32[:, :, :],  # spectrum_arr
    types.float32[:],  # wvls
    types.boolean[:],  # bad_bands
    types.float32,  # min_wvl
    types.float32,  # max_wvl
)


@numba_njit_wrapper(
    non_njit_func=slice_to_bounds_3D,
    signature=slice_bounds_3d_sig,
    parallel=True,
)
def slice_to_bounds_3D_numba(
    spectrum_arr: np.ndarray,
    wvls: np.ndarray,
    bad_bands: np.ndarray,
    min_wvl: np.float32,
    max_wvl: np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-compatible version of slice_to_bounds_3D.

    See `slice_to_bounds_3D` for full documentation.
    """
    if spectrum_arr.ndim != 3:
        raise ValueError(f"spectrum_arr must be 3D (b, y, x); got shape {spectrum_arr.shape}")

    b, _, _ = spectrum_arr.shape

    if wvls.ndim != 1 or bad_bands.ndim != 1 or wvls.shape[0] != b or bad_bands.shape[0] != b:
        raise ValueError(
            "Shape mismatch: "
            f"spectrum_arr has shape {spectrum_arr.shape}, "
            f"wvls has shape {wvls.shape}, "
            f"bad_bands has shape {bad_bands.shape}"
        )

    mask = np.ones(wvls.shape, dtype=np.bool_)
    # NOTE: in pure njit with the explicit signature, min_wvl/max_wvl
    # are float32, so they cannot actually be None; these checks will
    # always be True there. They're still useful when called via the
    # non-njit fallback.
    if min_wvl is not None:
        mask &= wvls >= min_wvl
    if max_wvl is not None:
        mask &= wvls <= max_wvl

    return spectrum_arr[mask, :, :], wvls[mask], bad_bands[mask]


def slice_to_bounds_1D(
    spectrum_arr: np.ndarray,
    wvls: np.ndarray,
    bad_bands: np.ndarray,
    min_wvl: np.float32,
    max_wvl: np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        spectrum_arr.ndim != 1
        or spectrum_arr.shape[0] != wvls.shape[0]
        or spectrum_arr.shape[0] != bad_bands.shape[0]
    ):
        raise ValueError(
            f"Shape mismatch: reflectance has shape {spectrum_arr.shape}, "
            f"wavelengths has shape {wvls.shape}, "
            f"bad_bands has shape {bad_bands.shape}"
        )

    mask = np.ones(wvls.shape, dtype=np.bool_)
    if min_wvl is not None:
        mask &= wvls >= min_wvl
    if max_wvl is not None:
        mask &= wvls <= max_wvl

    return spectrum_arr[mask], wvls[mask], bad_bands[mask]


slice_bounds_sig = types.Tuple((types.float32[:], types.float32[:], types.boolean[:]))(
    types.float32[:], types.float32[:], types.boolean[:], types.float32, types.float32
)


@numba_njit_wrapper(
    non_njit_func=slice_to_bounds_1D,
    signature=slice_bounds_sig,
    parallel=True,
)
def slice_to_bounds_1D_numba(
    spectrum_arr: np.ndarray,
    wvls: np.ndarray,
    ref_bad_bands: np.ndarray,
    min_wvl: np.float32,
    max_wvl: np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spectrum_arr.ndim != 1 or spectrum_arr.shape[0] != wvls.shape[0]:
        raise ValueError(
            f"Shape mismatch: reflectance has shape {spectrum_arr.shape}, "
            f"wavelengths has shape {wvls.shape}, "
            f"bad_bands has shape {ref_bad_bands.shape}"
        )

    mask = np.ones(wvls.shape, dtype=np.bool_)
    if min_wvl is not None:
        mask &= wvls >= min_wvl
    if max_wvl is not None:
        mask &= wvls <= max_wvl

    return spectrum_arr[mask], wvls[mask], ref_bad_bands[mask]


interp1d_monotonic_sig = types.float32[:](types.float32[:], types.float32[:], types.float32[:])


@numba_njit_wrapper(
    non_njit_func=interp1d_monotonic,
    signature=interp1d_monotonic_sig,
    cache=True,
)
def interp1d_monotonic_numba(x, y, x_new):
    """
    Perform linear interpolation on strictly increasing `x` and `x_new` arrays.

    This function implements a single-pass, monotonic, linear interpolation
    algorithm that is compatible with Numba's `njit`. Both `x` and `x_new`
    must be strictly increasing. Values in `x_new` that fall outside the
    range `[x[0], x[-1]]` are returned as np.nan.

    Args:
        x (np.ndarray):
            A 1D float array of strictly increasing x-coordinates.
        y (np.ndarray):
            A 1D float array containing values corresponding to `x`.
        x_new (np.ndarray):
            A 1D float array of strictly increasing query points at which
            interpolation is evaluated.

    Returns:
        np.ndarray:
            A 1D float array of interpolated values evaluated at each point
            in `x_new`.
    """
    n = x.shape[0]
    m = x_new.shape[0]
    out = np.empty(m, dtype=np.float32)

    j = 0  # index into x

    for i in range(m):
        xn = x_new[i]

        # handle out-of-bounds (no extrapolation)
        if xn < x[0] or xn > x[n - 1]:
            out[i] = np.nan
            continue

        # advance j until we find x[j] <= xn <= x[j+1]
        # we know xn is >= previous x_new, so j never moves backwards
        while j < n - 2 and x[j + 1] < xn:
            j += 1

        x0 = x[j]
        x1 = x[j + 1]
        y0 = y[j]
        y1 = y[j + 1]

        t = (xn - x0) / (x1 - x0)
        out[i] = y0 + t * (y1 - y0)

    return out


def populate_combo_box_with_units(
    cbox: QComboBox, default_unit: Optional[u.Unit] = u.nanometer, use_none_unit=True
):
    # Helpful mapping of units (some duplicates removed for clarity)
    unit_options: list[tuple[str, Optional[u.Unit]]] = [
        ("None", None),  # Allow the caller to opt-out of unit conversion
        ("nm (nanometer)", u.nanometer),
        ("µm (micrometer)", u.micrometer),
        ("mm (millimeter)", u.millimeter),
        ("cm (centimeter)", u.centimeter),
        ("m (meter)", u.meter),
        ("Å (angstrom)", u.angstrom),
        ("cm⁻¹ (wavenumber)", u.cm**-1),
        ("GHz", u.GHz),
        ("MHz", u.MHz),
    ]

    for text, unit_obj in unit_options:
        if text == "None" and not use_none_unit:
            continue
        cbox.addItem(text, userData=unit_obj)

    # Default to nanometers if present.
    default_index = next((i for i in range(cbox.count()) if cbox.itemData(i) == default_unit), 0)
    cbox.setCurrentIndex(default_index)


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
    for plugin_name, plugin in app_state.get_plugins().items():
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
                    print(
                        f'WARNING:  Function "{k}" is defined '
                        + f"multiple times (last seen in plugin {plugin_name})"
                    )

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
    """
    Formats an optional string for logging.  If the argument is a string then
    the function returns the string in double-quotes.  If the argument is
    ``None`` then the function returns the string ``'None'``.
    """
    if s is not None:
        return f'"{s}"'
    else:
        return "None"


def add_toolbar_action(toolbar, icon_path, text, parent, shortcut=None, before=None):
    """
    A helper function to set up a toolbar action using the common configuration
    used for these actions.
    """
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
    """
    This class provides a context manager so that a QPainter object can be
    managed by a Python "with" statement.

    For more information see:
    https://docs.python.org/3/reference/datamodel.html#context-managers
    """

    def __init__(self, _painter: QPainter):
        if _painter is None:
            raise ValueError("_painter cannot be None")

        self._painter: QPainter = _painter

    def __enter__(self) -> QPainter:
        return self._painter

    def __exit__(self, type, value, traceback) -> bool:
        self._painter.end()

        # If an exception occurred within the with block, reraise it by
        # returning False.  Otherwise return True.
        return traceback is None


def get_painter(widget: QWidget) -> PainterWrapper:
    """
    This helper function makes a QPainter for writing to the specified QWidget,
    and then wraps it with a PainterWrapper context manager.  It is intended to
    be used with the Python "with" statement, like this:

    with get_painter(some_widget) as painter:
        painter.xxxx()  # Draw stuff
    """
    return PainterWrapper(QPainter(widget))


def make_filename(s: str) -> str:
    """
    This helper function makes a filename out of a string, following these
    rules:
    *   Any letters or numbers are left unchanged.
    *   Any sequences of whitespace characters (spaces, tabs and newlines) are
        collapsed down to a single space character.
    *   These punctuation characters are left as-is:  '-', '_', '.'.  All other
        punctuation characters are removed.

    If the function is given a string consisting of only whitespace, or an
    empty string, it will throw a ValueError.
    """

    # Apply this filter to collapse any whitespace characters down to one space.
    # If the string is entirely whitespace, the result will be an empty string.
    s = " ".join(s.strip().split())

    if len(s) == 0:
        raise ValueError("Cannot make a filename out of an empty string")

    # Filter out any characters that are not alphanumeric, or some basic
    # punctuation and spaces.
    result = ""
    for ch in s:
        if ch in string.ascii_letters or ch in string.digits or ch in ["_", "-", ".", " "]:
            result += ch

    return result


def get_matplotlib_colors():
    """
    Generates a list of all recognized matplotlib color names, which are
    suitable for displaying graphical plots.

    The definition of "suitable for displaying graphical plots" is currently
    that the color be dark enough to show up on a white background.
    """
    names = matplotlib.colors.get_named_colors_mapping().keys()
    colors = []
    for name in names:
        if len(name) <= 1:
            continue

        if name.find(":") != -1:
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
    """
    Returns a random matplotlib color name from the available matplotlib colors
    returned by the get_matplotlib_colors().
    """
    all_names = get_matplotlib_colors()
    while True:
        name = random.choice(all_names)
        if name not in exclude_colors:
            return name


def get_color_icon(color_name: str, width: int = 16, height: int = 16) -> QIcon:
    """
    Generate a QIcon of the specified color and optional size.  If the size is
    unspecified, a 16x16 icon is generated.

    *   color_name is a string color name recognized by matplotlib, which
        includes strings of the form "#RRGGBB" where R, G and B are hexadecimal
        digits.

    *   width is the icon's width in pixels, and defaults to 16.

    *   height is the icon's height in pixels, and defaults to 16.
    """
    rgba = matplotlib.colors.to_rgba_array(color_name).flatten()

    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor.fromRgbF(rgba[0], rgba[1], rgba[2]))

    return QIcon(QPixmap.fromImage(img))


def clear_treewidget_selections(tree_widget: QTreeWidget) -> None:
    """
    Given a QTreeWidget object, this function clears all selections of
    tree widget items.
    """

    def clear_treewidget_item_selections(item: QTreeWidgetItem) -> None:
        if item.isSelected():
            item.setSelected(False)

        for i in range(item.childCount()):
            clear_treewidget_item_selections(item.child(i))

    for i in range(tree_widget.topLevelItemCount()):
        item = tree_widget.topLevelItem(i)
        clear_treewidget_item_selections(item)


def generate_unused_filename(basename: str, extension: str) -> str:
    """
    Generates a filename that is currently unused on the filesystem.  The
    base name (including path) and extension to use are both specified as
    arguments to this function.

    If "basename.extension" is available as a filename, the function will
    return that string.

    If it is already used, the function will generate a filename of the form
    "basename_i.extension", where i starts at 1, and is incremented until
    the filename is unused.
    """
    filename = f"{basename}.{extension}"
    if os.path.exists(filename):
        i = 1
        while True:
            filename = f"{basename}_{i}.{extension}"
            if not os.path.exists(filename):
                break

            i += 1

    return filename


def cv2_rotate_scale_expand(
    img: np.ndarray,
    angle: float,
    scale: float = 1.0,
    interp: int = 1,
    mask_fill_value: float = 0,
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
    orig_mask = None
    if isinstance(img, np.ma.masked_array):
        orig_mask = img.mask
        img = img.filled(mask_fill_value)
        if not isinstance(orig_mask, np.ndarray):
            orig_mask = None

    # 3. Build the rotation+scale matrix
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # 4. Compute new canvas size so nothing is clipped
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    # shift origin to centre result
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    # 5. Warp the image
    out = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=mask_fill_value,
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
            borderValue=0,
        )
        warped_mask = ~(warped_valid.astype(bool))
        return np.ma.MaskedArray(out, mask=warped_mask)

    return out


def cv2_rotate_scale_expand_3D(
    img: np.ndarray,
    angle: float,
    scale: float = 1.0,
    interp: str = "linear",
    mask_fill_value: float = 0,
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
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    # choose interpolation flag
    orig_mask = None
    if isinstance(img, np.ma.MaskedArray):
        orig_mask = img.mask
        img = img.filled(mask_fill_value)

    # 3. Build the rotation+scale matrix
    h, w = img.shape[1:3]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # 4. Compute new canvas size so nothing is clipped
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    # shift origin to centre result
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy

    # 5. Warp the image
    out = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=_INTERPOLATIONS.get(interp, cv2.INTER_LINEAR),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=mask_fill_value,
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
            borderValue=0,
        )
        warped_mask = ~(warped_valid.astype(bool))
        return np.ma.MaskedArray(out, mask=warped_mask)

    return out


def pillow_rotate_scale_expand(
    arr: np.ndarray,
    angle: float,
    scale: float = 1.0,
    resample: str = "bilinear",
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
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
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
    # print(f"arr_uint8.shape: {arr_uint8.shape}")
    # print(f"arr.shape: {arr.shape}")
    # build PIL image
    img = Image.fromarray(arr_uint8.T)
    # print(f"img.size: {img.size}")
    # 1) scale
    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), resample=mode)

    # 2) rotate + expand
    img = img.rotate(angle, resample=mode, expand=True)

    out = np.array(img)
    # print(f"out.shape: {out.shape}")
    # if original was float, map back to original range
    if is_float:
        lo, hi = arr.min(), arr.max()
        # print(f"is_float true, lo: {lo}, hi: {hi}")
        out = out.astype(np.float32) / 255 * (hi - lo or 1) + lo

    return out


def pixel_coord_to_geo_coord(
    pixel_coord: Tuple[float, float],
    geo_transform: Tuple[float, float, float, float, float, float],
) -> Tuple[float, float]:
    """
    A helper function to translate a pixel-coordinate into a linear geographic
    coordinate using the geographic transform from GDAL.

    The geo_transform argument is a 6-tuple that specifies a 2D affine
    transformation, using the method exposed by GDAL.  See this URL for more
    details:  https://gdal.org/tutorials/geotransforms_tut.html
    """
    (pixel_x, pixel_y) = pixel_coord
    geo_x = geo_transform[0] + pixel_x * geo_transform[1] + pixel_y * geo_transform[2]
    geo_y = geo_transform[3] + pixel_x * geo_transform[4] + pixel_y * geo_transform[5]
    return (geo_x, geo_y)


def ulurll_to_gt(
    ul: Tuple[int, int],
    ur: Tuple[int, int],
    ll: Tuple[int, int],
    width: int,
    height: int,
):
    ulx, uly = ul[0], ul[1]
    urx, ury = ur[0], ur[1]
    llx, lly = ll[0], ll[1]
    gt = [
        ulx,
        (urx - ulx) / (width),
        (llx - ulx) / (height),
        uly,
        (ury - uly) / (width),
        (lly - uly) / (height),
    ]
    return gt


def rotate_scale_geotransform(gt, theta, width_orig, height_orig, width_rot, height_rot):
    """
    Rotates and scales the geo transform. Does so by using three corner points.
    """
    pix_px, pix_py = width_orig / 2, height_orig / 2
    rad = math.radians(theta)
    cos_t, sin_t = math.cos(rad), math.sin(rad)

    def rot_pix(v):
        x, y = v
        # shift so pivot is at (0,0)
        dx, dy = x - pix_px, y - pix_py
        # rotate about origin
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        # shift back
        return (rx + pix_px, ry + pix_py)

    def rot_inverse_pix(v):
        x, y = v
        dx, dy = x - pix_px, y - pix_py
        # inverse rotation = rotate by –θ (or use transpose)
        ix = dx * cos_t + dy * sin_t
        iy = -dx * sin_t + dy * cos_t
        return (ix + pix_px, iy + pix_py)

    upper_left = (0, 0)
    upper_right = (width_orig, 0)
    bottom_left = (0, height_orig)
    bottom_right = (width_orig, height_orig)

    rotated_ul = rot_pix(upper_left)
    rotated_ur = rot_pix(upper_right)
    rotated_bl = rot_pix(bottom_left)
    rotated_br = rot_pix(bottom_right)

    new_ul_x_rot = min(rotated_ul[0], rotated_ur[0], rotated_bl[0], rotated_br[0])
    new_ul_y_rot = min(rotated_ul[1], rotated_ur[1], rotated_bl[1], rotated_br[1])

    new_ur_x_rot = max(rotated_ul[0], rotated_ur[0], rotated_bl[0], rotated_br[0])
    new_ur_y_rot = min(rotated_ul[1], rotated_ur[1], rotated_bl[1], rotated_br[1])

    new_bl_x_rot = min(rotated_ul[0], rotated_ur[0], rotated_bl[0], rotated_br[0])
    new_bl_y_rot = max(rotated_ul[1], rotated_ur[1], rotated_bl[1], rotated_br[1])

    new_ul_pix = rot_inverse_pix((new_ul_x_rot, new_ul_y_rot))
    new_ur_pix = rot_inverse_pix((new_ur_x_rot, new_ur_y_rot))
    new_bl_pix = rot_inverse_pix((new_bl_x_rot, new_bl_y_rot))

    new_ul_spatial = pixel_coord_to_geo_coord(new_ul_pix, gt)
    new_ur_spatial = pixel_coord_to_geo_coord(new_ur_pix, gt)
    new_bl_spatial = pixel_coord_to_geo_coord(new_bl_pix, gt)

    new_gt = ulurll_to_gt(new_ul_spatial, new_ur_spatial, new_bl_spatial, width_rot, height_rot)

    return new_gt


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

    help_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(f"{link}")))
