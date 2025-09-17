"""Continuum Removal Plugin

This script allows the user to implement continuum removal in WISER on any images uploaded to WISER.

This plugin has 4 main functionalities:
    * Continuum remove a single spectrum
    * Continuum remove a collected spectra
    * Continuum remove a subset of the image
    * Continuum remove the whole image

This script requires that `numpy`, `pyside2`, and `scipy` be installed within the Python
environment you are running this script in.

This script requires the following .ui files to be in the same folder as this python script:
    * dimensions_bands.ui - GUI for dimension range and band range selection
    * error.ui - GUI for error message

Code originally written by Amy Wang, Cornell '23
"""

from __future__ import division

import numpy as np
from numba import njit, types
from numba.typed import List
import logging
import os

from typing import TYPE_CHECKING, Tuple

from wiser import plugins, raster
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from scipy.interpolate import interp1d

from wiser.utils.numba_wrapper import numba_njit_wrapper, convert_to_float32_if_needed

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.gui.app_state import ApplicationState


def crossProduct(o, a, b):
    """Code provided by Sahil Azad
    Calculates the cross product of two vectors oa and ob

    Parameters
    ----------
    o: list
        second to last value in upper hull list
    a: list
        last value in upper hull list
    b: list
        point list

    Returns
    ----------
    Cross product of two vectors oa and ob
    """

    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

@numba_njit_wrapper(non_njit_func=crossProduct)
def cross_product_numba(o, a, b):
    """Code provided by Sahil Azad
    Calculates the cross product of two vectors oa and ob

    Parameters
    ----------
    o: list
        second to last value in upper hull list
    a: list
        last value in upper hull list
    b: list
        point list

    Returns
    ----------
    Cross product of two vectors oa and ob
    """

    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def monotone(points):
    """Code provided by Sahil Azad
    Calculates the upper hull of the spectrum

    Parameters
    ----------
    points: ndarray
        Column stacked wavelengths and reflectanc values

    Returns
    ----------
    upper: list
        A list of points on the upper convex hull before interpolation
    """

    upper = []
    l_len = 0
    for p in points:
        while l_len >= 2 and crossProduct(upper[-2], upper[-1], p) <= 0:
            upper = upper[:-1]
            l_len -= 1
        upper.append(p)
        l_len += 1
    return upper

_point_t = types.UniTuple(types.float64, 2)
@numba_njit_wrapper(non_njit_func=monotone)
def monotone_numba(points):
    """Code provided by Sahil Azad
    Calculates the upper hull of the spectrum

    Parameters
    ----------
    points: ndarray
        Column stacked wavelengths and reflectanc values

    Returns
    ----------
    upper: list
        A list of points on the upper convex hull before interpolation
    """
    upper = List.empty_list(_point_t)

    for k in range(points.shape[0]):
        p = (points[k, 0], points[k, 1])
        while len(upper) >= 2 and cross_product_numba(
                upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    m = len(upper)
    hull_arr = np.empty((m, 2), dtype=np.float64)
    for i in range(m):
        hull_arr[i, 0], hull_arr[i, 1] = upper[i]
    return hull_arr

def continuum_removal(reflectance, waves):
    """Calculates the continuum removed spectrum

    Parameters
    ----------
    reflectance: list
        second to last value in upper hull list
    waves: list
        last value in upper hull list

    Returns
    ----------
    final: ndarray
        An array of points on the continuum removed spectrum
    iy_hull: ndarray
        An array of points on the convex hull
    """

    points = np.column_stack((waves, reflectance))
    hull = np.array(monotone(points))
    coords_con_hull = hull.transpose()
    order = np.argsort(coords_con_hull[0])
    xp = coords_con_hull[0][order]
    fp = coords_con_hull[1][order]
    iy_hull_np = np.interp(waves, xp, fp)
    norm = np.divide(reflectance, iy_hull_np)
    final = np.column_stack((waves, norm)).transpose(1, 0)[1]
    return final, iy_hull_np

@numba_njit_wrapper(non_njit_func=continuum_removal)
def continuum_removal_numba(reflectance, waves):
    """Calculates the continuum removed spectrum

    Parameters
    ----------
    reflectance: list
        second to last value in upper hull list
    waves: list
        last value in upper hull list

    Returns
    ----------
    final: ndarray
        An array of points on the continuum removed spectrum
    iy_hull: ndarray
        An array of points on the convex hull
    """
    points = np.column_stack((waves, reflectance))
    hull = monotone_numba(points)
    coords_con_hull = hull.transpose()
    order = np.argsort(coords_con_hull[0])
    xp = coords_con_hull[0][order]
    fp = coords_con_hull[1][order]
    iy_hull_np = np.interp(waves, xp, fp)
    norm = np.divide(reflectance, iy_hull_np)
    final = np.column_stack((waves, norm)).transpose(1, 0)[1]
    return final, iy_hull_np

def continuum_removal_image(image_data: np.ndarray, x_axis: np.ndarray, rows: int, cols: int, bands: int):
    image_spectra_2d = image_data.reshape(
        (rows * cols, bands)
    )  # [y][x][b] -> [y*x][b]
    results = np.empty_like(image_spectra_2d, dtype=np.float32)
    for i in range(image_spectra_2d.shape[0]):
        reflectance = image_spectra_2d[i]
        # TODO (Joshua G-K) Vectorize the continuum removal function
        continuum_removed, hull = continuum_removal(reflectance, x_axis)
        results[i] = continuum_removed
    results = results.reshape((rows, cols, bands))
    results = results.copy().transpose(
        2, 0, 1
    )  # [y][x][b] -> [b][y][x]
    return results

@numba_njit_wrapper(non_njit_func=continuum_removal_image)
def continuum_removal_image_numba(image_data: np.ndarray, x_axis: np.ndarray, rows: int, cols: int, bands: int):
    image_spectra_2d = image_data.reshape(
        (rows * cols, bands)
    )  # [y][x][b] -> [y*x][b]
    results = np.empty_like(image_spectra_2d, dtype=np.float32)
    for i in range(image_spectra_2d.shape[0]):
        reflectance = image_spectra_2d[i]
        # TODO (Joshua G-K) Vectorize the continuum removal function
        continuum_removed, hull = continuum_removal_numba(reflectance, x_axis)
        results[i] = continuum_removed
    results = results.reshape((rows, cols, bands))
    results = results.copy().transpose(
        2, 0, 1
    )  # [y][x][b] -> [b][y][x]
    return results

class ContinuumRemovalPlugin(plugins.ContextMenuPlugin):
    """
    A Class to represents the continuum removed result of a hyperspectral image

    Parameters
    ----------
    None

    Attributes
    ----------
    None
    """

    def __init__(self):
        logging.info("Continuum Removal")

    def add_context_menu_items(
        self, context_type: plugins.types.ContextMenuType, context_menu, context
    ):
        """Adds plugin to WISER as a context menu type plugin

        Parameters
        ----------
        context_type: ContextMenuType
            the plugin type and where it can be used
        context_menu: PySide2.QtWidgets.QMenu
            the context menu available to the plugin
        context: dict
            Available WISER classes
        """

        if context_type == plugins.ContextMenuType.SPECTRUM_PICK:
            act1 = context_menu.addAction(
                context_menu.tr("Continuum Removal: Single Spectrum")
            )
            act1.triggered.connect(
                lambda checked=False: self.single_spectrum(context=context)
            )

            act2 = context_menu.addAction(
                context_menu.tr("Continuum Removal: Collected Spectra")
            )
            act2.triggered.connect(
                lambda checked=False: self.collected_spectra(context=context)
            )

        if context_type == plugins.ContextMenuType.RASTER_VIEW:
            act3 = context_menu.addAction(context_menu.tr("Continuum Removal: Image"))
            act3.triggered.connect(
                lambda checked=False: self.dimension(context=context)
            )

    def error_box(self, message, context):
        """Displays desired error message and goes back to dimensions GUI when finished

        Parameters
        ----------
        message: str
            Error message to be displayed in the widget
        context: dict
            Available WISER classes
        """

        path2 = os.path.join(os.path.dirname(__file__), "error.ui")
        dialog = plugins.load_ui_file(path2)
        error_message = dialog.findChild(QLabel, "error_message")
        error_message.setText(message)

        if dialog.exec() == QDialog.Accepted:
            self.dimension(context)

    def set_entire_image(self, dialog, cols, rows):
        """Sets dimensions in the dimensions GUI to include the entire image

        Parameters
        ----------
        dialog: PySide2.QtWidgets.QDialog
            Dialog that shows the dimensions GUI
        cols: int
            Total number of columns in the image
        rows: int
            Total number of rows in the image
        """

        min_cols = dialog.findChild(QSpinBox, "min_cols")
        min_rows = dialog.findChild(QSpinBox, "min_rows")
        max_cols = dialog.findChild(QSpinBox, "max_cols")
        max_rows = dialog.findChild(QSpinBox, "max_rows")
        min_cols.setValue(0)
        min_rows.setValue(0)
        max_cols.setValue(cols)
        max_rows.setValue(rows)

    def set_all_bands(self, dialog, last):
        """Sets band range in the bands GUI to include all bands

        Parameters
        ----------
        dialog: PySide2.QtWidgets.QDialog
            Dialog that shows the dimensions GUI
        last: int
            Total number of bands
        """

        minimum = dialog.findChild(QComboBox, "min_bands")
        minimum.setCurrentIndex(0)
        maximum = dialog.findChild(QComboBox, "max_bands")
        maximum.setCurrentIndex(last)

    def combo_box_changed(self, combo, spin):
        """Changes value in QSpinBox to match with corresponding value in QComboBox

        Parameters
        ----------
        combo: PySide2.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available bands
        spin: PySide2.QtWidgets.QAbstractSpinBox.QSpinBox
            Editable spin box that has a range of all available band numbers
        """

        idx = combo.currentIndex()
        spin.setValue(idx)

    def spin_box_changed(self, combo, spin):
        """Changes value in QComboBox to match with corresponding value in QSpinBox

        Parameters
        ----------
        combo: PySide2.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available bands
        spin: PySide2.QtWidgets.QAbstractSpinBox.QSpinBox
            Editable spin box that has a range of all available band numbers
        """

        idx = spin.value()
        combo.setCurrentIndex(idx)

    def dimension(self, context):
        """Displays GUI to allow user to choose dimension and band range to continuum remove

        Parameters
        ----------
        context: dict
            Available WISER classes
        """

        path = os.path.join(os.path.dirname(__file__), "dimensions_bands.ui")
        dialog = plugins.load_ui_file(path)

        entire_image = dialog.findChild(QPushButton, "entire_image")
        min_cols = dialog.findChild(QSpinBox, "min_cols")
        min_rows = dialog.findChild(QSpinBox, "min_rows")
        max_cols = dialog.findChild(QSpinBox, "max_cols")
        max_rows = dialog.findChild(QSpinBox, "max_rows")
        min_spin = dialog.findChild(QSpinBox, "min_spin")
        max_spin = dialog.findChild(QSpinBox, "max_spin")

        dataset = context["dataset"]
        total_cols = dataset.get_shape()[-1]
        total_rows = dataset.get_shape()[-2]

        min_cols.setRange(0, total_cols - 1)
        min_rows.setRange(0, total_rows - 1)
        max_cols.setRange(0, total_cols - 1)
        max_rows.setRange(0, total_rows - 1)

        min_cols.setValue(0)
        min_rows.setValue(0)
        max_cols.setValue(total_cols - 1)
        max_rows.setValue(total_rows - 1)

        entire_image.clicked.connect(
            lambda checked=True: self.set_entire_image(
                dialog, total_cols - 1, total_rows - 1
            )
        )

        all_bands = dialog.findChild(QPushButton, "all_bands")
        minimum = dialog.findChild(QComboBox, "min_bands")
        maximum = dialog.findChild(QComboBox, "max_bands")

        bands = dataset.band_list()
        bands = list([i["description"] for i in bands])
        bands = list(f"Band {bands.index(i)}: " + i for i in bands)
        last = len(bands) - 1

        all_bands.clicked.connect(lambda checked=True: self.set_all_bands(dialog, last))

        minimum.addItems(bands)
        maximum.addItems(bands)
        min_spin.setRange(0, last)
        max_spin.setRange(0, last)

        minimum.setCurrentIndex(0)
        maximum.setCurrentIndex(last)
        min_spin.setValue(0)
        max_spin.setValue(last)

        minimum.currentIndexChanged.connect(
            lambda checked=True: self.combo_box_changed(minimum, min_spin)
        )
        min_spin.valueChanged.connect(
            lambda checked=True: self.spin_box_changed(minimum, min_spin)
        )
        maximum.currentIndexChanged.connect(
            lambda checked=True: self.combo_box_changed(maximum, max_spin)
        )
        max_spin.valueChanged.connect(
            lambda checked=True: self.spin_box_changed(maximum, max_spin)
        )

        if dialog.exec() == QDialog.Accepted:
            min_cols = min_cols.value()
            min_rows = min_rows.value()
            max_cols = max_cols.value()
            max_rows = max_rows.value()
            minimum = minimum.currentIndex()
            maximum = maximum.currentIndex()

            if (min_cols > max_cols) or (min_rows > max_rows) or minimum > maximum:
                self.error_box("Minimum must be less than the maximum", context)
            else:
                max_cols += 1
                max_rows += 1
                maximum += 1
                self.image(
                    min_cols, min_rows, max_cols, max_rows, minimum, maximum, context
                ) 

    def plot_continuum_removal(self, spec_object, context) -> Tuple[raster.spectrum.NumPyArraySpectrum, raster.spectrum.NumPyArraySpectrum]:
        """Plots the continuum removed spectrum and the convex hull

        Parameters
        ----------
        spec_object: wiser.raster.Spectrum
            Spectrum to be continuum removed
        context: dict
            Available WISER classes
        """

        spectrum = spec_object.get_spectrum()
        wavelengths_org = spec_object.get_wavelengths()  # type <astropy>
        wavelengths = np.array([i.value for i in wavelengths_org])[::-1]
        continuum_removed_spec, hull = continuum_removal_numba(spectrum, wavelengths)
        new_spec = raster.spectrum.NumPyArraySpectrum(continuum_removed_spec)
        new_spec.set_name(spec_object.get_name() + " Continuum Removed")
        new_spec.set_wavelengths(wavelengths_org)
        convex_hull = raster.spectrum.NumPyArraySpectrum(hull)
        convex_hull.set_name("Convex Hull " + spec_object.get_name())
        convex_hull.set_wavelengths(wavelengths_org)
        if context is not None:
            context["wiser"].collect_spectrum(new_spec)
            context["wiser"].collect_spectrum(convex_hull)
        return (new_spec, convex_hull)

    def single_spectrum(self, context):
        """Plots the continuum removed spectrum and the convex hull

        Parameters
        ----------
        context: dict
            Available WISER classes
        """

        self.plot_continuum_removal(context["spectrum"], context)

    def collected_spectra(self, context):
        """Plots the continuum removed spectra and the convex hulls

        Parameters
        ----------
        context: dict
            Available WISER classes
        """

        collectedSpectra = context["wiser"].get_collected_spectra()
        collected_cr = []
        for spectrum in collectedSpectra:
            collected_cr.append(self.plot_continuum_removal(spectrum, context))

    def image(
        self, min_cols, min_rows, max_cols, max_rows, min_band, max_band, context
    ):
        """Displays on WISER the continuum removed spectra of an image or a subset of the image

        Parameters
        ----------
        min_cols: int
            Minimum column number user chose (also width, XSize)
        min_rows: int
            Minimum row number user chose (also height, YSize)
        max_cols: int
            Maximum column number user chose
        max_rows: int
            Maximum row number user chose
        min_band: int
            Minimum band number user chose
        max_band: int
            Maximum band number user chose
        context: dict
            Available WISER classes
        """

        app_state: ApplicationState = context["wiser"]
        dataset: RasterDataSet = context["dataset"]
        dband = max_band - min_band
        dcols = max_cols - min_cols
        drows = max_rows - min_rows
        image_data = dataset.get_image_data_subset(min_cols, min_rows, min_band,
                                                   dcols, drows, dband)
         # A numpy array such that the pixel (x, y) values (spectrum value) of band b are at element array[b][y][x]
        filename = dataset.get_name()
        description = dataset.get_description()
        band_description = dataset.band_list()
        if "wavelength_str" in band_description[0]:
            x_axis = np.array([float(i["wavelength_str"]) for i in band_description])
        else:
            assert "index" in band_description[0], "No key named index in return value of dataset.band_list()"
            x_axis = np.array([float(i["index"]) for i in band_description])
        x_axis = x_axis[min_band:max_band]
        default_bands = dataset.default_display_bands()
        if default_bands is None:
            default_bands = [0, 1, 2]

        max_default = max(default_bands)

        # TODO (Joshua G-K): Add better logic here for when user selected bands don't match?
        if max_band < max_default:
            default_bands = [0, 1, 2]

        x_axis = x_axis[::-1]

        cols = max_cols - min_cols
        rows = max_rows - min_rows
        bands = max_band - min_band
        image_data, x_axis = convert_to_float32_if_needed(image_data, x_axis)
        image_data: np.ndarray = image_data.transpose(1, 2, 0)
        if not image_data.flags.c_contiguous:
            image_data = np.ascontiguousarray(image_data)
        if isinstance(image_data, np.ma.MaskedArray):
            image_data = image_data.data
        spectra = image_data
        new_image_data = continuum_removal_image(spectra, x_axis, rows, cols, bands)

        raster_data = raster.RasterDataLoader()
        new_data = raster_data.dataset_from_numpy_array(new_image_data, app_state.get_cache())
        new_data.set_name(f"Continuum Removal on {filename}")
        new_data.set_description(description)
        new_data.set_default_display_bands(default_bands)

        new_data.copy_spatial_metadata(dataset.get_spatial_metadata())
        new_data.copy_spectral_metadata(dataset.get_spectral_metadata())
        context["wiser"].add_dataset(new_data)
        return new_data
