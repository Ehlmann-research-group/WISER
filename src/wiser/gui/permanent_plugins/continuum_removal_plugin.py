"""Continuum Removal Plugin

This script allows the user to implement continuum removal in WISER on any images uploaded to WISER.

This plugin has 4 main functionalities:
    * Continuum remove a single spectrum
    * Continuum remove a collected spectra
    * Continuum remove a subset of the image
    * Continuum remove the whole image

This script requires that `numpy`, `pyside6`, and `scipy` be installed within the Python
environment you are running this script in.

Code originally written by Amy Wang, Cornell '23
"""

from __future__ import division

import numpy as np
from numba import njit, types
from numba.typed import List
import logging
import os
import numba

from typing import TYPE_CHECKING, Tuple

from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from scipy.interpolate import interp1d

from wiser import plugins, raster
from wiser.utils.numba_wrapper import numba_njit_wrapper, convert_to_float32_if_needed
from wiser.utils.primitives import DataRef, DatasetRegionRef, PriorityClass
from wiser.utils.primitives import ExternalRasterHandle
from wiser.utils.task_stage_utils import get_continuum_removal_image_pipeline
from wiser.utils.task_system import SemanticTask
from wiser.utils.worker_runtime import get_process_storage_client

from wiser.raster.spectrum import Spectrum
from wiser.raster.dataset import RasterDataSet, SpatialMetadata, SpectralMetadata

from wiser.gui.generated.continuum_removal_dimensions_bands_ui import Ui_ContinuumRemoval

if TYPE_CHECKING:
    from wiser.gui.app_services import AppServices
    from wiser.gui.app_state import ApplicationState


def cross_product(o, a, b):
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


_point_t = types.UniTuple(types.float32, 2)
cross_sig = types.float32(_point_t, _point_t, _point_t)


@numba_njit_wrapper(non_njit_func=cross_product, signature=cross_sig)
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
        Column stacked wavelengths and reflectance values (Nx2) where N
        is number of wavelength/reflectance value. Should be in increasing order.

    Returns
    ----------
    upper: list
        A list of points on the upper convex hull before interpolation
    """

    upper = []
    l_len = 0
    for i in range(points.shape[0]):
        p = points[i, :]
        # We do '> 0' because we are going clock wise around the hull
        while l_len >= 2 and cross_product(upper[-2], upper[-1], p) > 0:
            upper = upper[:-1]
            l_len -= 1
        upper.append(p)
        l_len += 1
    return upper


_point_t = types.UniTuple(types.float32, 2)
mono_sig = types.float32[:, :](types.float32[:, :])


@numba_njit_wrapper(non_njit_func=monotone, signature=mono_sig)
def monotone_numba(points):
    """
    Calculates the upper hull of the spectrum.

    Parameters
    ----------
    points: ndarray
        Column stacked wavelengths and reflectance values. Size (Nx2) where N
        is number of wavelength/reflectance value. Should be in increasing order.

    Returns
    ----------
    upper: list
        A list of points on the upper convex hull before interpolation. In increasing order.
    """
    upper = List.empty_list(_point_t)

    for k in range(points.shape[0]):
        p = (points[k, 0], points[k, 1])
        # We do '> 0' because we are going clock wise around the hull
        while len(upper) >= 2 and cross_product_numba(upper[-2], upper[-1], p) > 0:
            upper.pop()
        upper.append(p)

    m = len(upper)
    hull_arr = np.empty((m, 2), dtype=np.float32)
    for i in range(m):
        hull_arr[i, 0], hull_arr[i, 1] = upper[i]
    return hull_arr


def continuum_removal(reflectance, waves) -> Tuple[np.ndarray, np.ndarray]:
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
    norm[iy_hull_np == 0.0] = 1.0
    final = np.column_stack((waves, norm)).transpose(1, 0)[1]
    return final, iy_hull_np


cr_sig = types.Tuple((types.float32[:], types.float32[:]))(types.float32[:], types.float32[:])


@numba_njit_wrapper(non_njit_func=continuum_removal, signature=cr_sig, cache=True)
def continuum_removal_numba(reflectance: np.ndarray, waves: np.ndarray):
    """Calculates the continuum removed spectrum for a single spectrum using numba

    Parameters
    ----------
    reflectance: list
        second to last value in upper hull list
    waves: list
        last value in upper hull list
    mask: ndarray
        A numpy boolean array used to mask which points to consider in the hull.
        1 means consider. 0 means don't consider.

    Returns
    ----------
    final: ndarray
        An array of points on the continuum removed spectrum
    iy_hull: ndarray
        An array of points on the convex hull
    """
    # Build points in float32 (avoid upcast)
    # (Numba is fine with column_stack, but it can upcast; be explicit)
    points = np.empty((waves.shape[0], 2), dtype=np.float32)
    points[:, 0] = waves
    points[:, 1] = reflectance

    hull = monotone_numba(points)  # float32[:, :]
    coords_con_hull = hull.transpose()
    order = np.argsort(coords_con_hull[0])

    wavelength_hull_values = coords_con_hull[0][order]  # float32[:]
    # float32[:], the reflectance values at the points on the hull
    reflectance_hull_values = coords_con_hull[1][order]

    # np.interp commonly yields float64; Numba also prefers float64 here.
    #  Use float64 temporaries for interpolation, then cast back to float32.
    iy_hull64 = np.interp(
        waves.astype(np.float64),
        wavelength_hull_values.astype(np.float64),
        reflectance_hull_values.astype(np.float64),
    )
    iy_hull = iy_hull64.astype(np.float32)

    # Keep division in float32 and return float32 arrays
    norm = np.divide(reflectance, iy_hull)
    norm[iy_hull == 0.0] = 1.0  # Avoid NaNs

    # Returning (float32[:], float32[:]) to match cr_sig
    return norm, iy_hull


def continuum_removal_image(
    image_data: np.ndarray,
    bad_bands_arr: np.ndarray,
    x_axis: np.ndarray,
    rows: int,
    cols: int,
    bands: int,
) -> np.ndarray:
    """
    Given a 3D numpy array of image data and a 1D numpy array of x-axis values, calculates the continuum
    removed spectrum for each pixel in the image.

    Parameters
    ----------
    image_data: np.ndarray
        A 3D numpy array of image data  [y][x][b]
    bad_bands_arr: np.ndarray
        A 1D numpy array that has 1s where the band is bad and 0s where it is good
    x_axis: np.ndarray
        A 1D numpy array of x-axis values
    rows: int
        The number of rows in the image
    cols: int
        The number of columns in the image
    bands: int
        The number of bands in the image

    Returns
    ----------
    results: np.ndarray
        A 3D numpy array of continuum removed image data
    """
    rows_cols = rows * cols
    results = np.empty_like(image_data, dtype=np.float32)
    for i in range(rows_cols):
        row = i // cols
        col = i % cols
        reflectance = image_data[row, col, :]
        reflectance[bad_bands_arr] = np.nan
        continuum_removed, _ = continuum_removal(reflectance, x_axis)
        results[row, col] = continuum_removed
    results = results.copy().transpose(2, 0, 1)  # [y][x][b] -> [b][y][x]
    return results


cr_image_sig = types.float32[:, :, :](
    types.float32[:, :, :], types.boolean[:], types.float32[:], types.intp, types.intp, types.intp
)


@numba_njit_wrapper(non_njit_func=continuum_removal_image, signature=cr_image_sig, parallel=True)
def continuum_removal_image_numba(
    image_data: np.ndarray,
    bad_bands_arr: np.ndarray,
    x_axis: np.ndarray,
    rows: int,
    cols: int,
    bands: int,
):
    """
    Given a 3D numpy array of image data and a 1D numpy array of x-axis
    values, calculates the continuum removed spectrum for each pixel in
    the image using numba.

    Parameters
    ----------
    image_data: np.ndarray
        A 3D numpy array of image data. Expected to be in C-contiguous order
        and with dimensions [rows=y=height][cols=x=width][b]. This lets us
        have the best memory access to get spectra
    bad_bands_arr: np.ndarray
        A 1D numpy array that has 1s where the band is bad and 0s where it is good
    x_axis: np.ndarray
        A 1D numpy array of x-axis values (should be decreasing)
    rows: int
        The number of rows in the image
    cols: int
        The number of columns in the image
    bands: int
        The number of bands in the image

    Returns
    ----------
    results: np.ndarray
        A 3D numpy array of continuum removed image data. Returns
        array in [b][y][x] order
    """
    image_data = np.ascontiguousarray(image_data)
    rows = rows
    cols = cols
    rows_cols = rows * cols
    results = np.empty_like(image_data, dtype=np.float32)
    for i in numba.prange(rows_cols):
        row = i // cols
        col = i % cols
        reflectance = image_data[row, col, :]
        reflectance[bad_bands_arr] = np.float32(np.nan)
        continuum_removed, _ = continuum_removal_numba(reflectance, x_axis)
        results[row, col] = continuum_removed
    results = results.copy().transpose(2, 0, 1)  # [y][x][b] -> [b][y][x]
    return results


def _continuum_x_axis(dataset: RasterDataSet, min_band: int, max_band: int) -> np.ndarray:
    band_description = dataset.band_list()
    if "wavelength_str" in band_description[0]:
        x_axis = np.array([float(i["wavelength_str"]) for i in band_description])
    else:
        assert "index" in band_description[0], "No key named index in return value of dataset.band_list()"
        x_axis = np.array([float(i["index"]) for i in band_description])
    return x_axis[min_band:max_band]


def _continuum_default_bands(dataset: RasterDataSet, max_band: int):
    default_bands = dataset.default_display_bands()
    if default_bands is None:
        default_bands = [0, 1, 2]
    if max_band < max(default_bands):
        default_bands = [0, 1, 2]
    return default_bands


def _apply_continuum_removed_metadata(
    new_data: RasterDataSet,
    dataset: RasterDataSet,
    min_cols: int,
    min_rows: int,
    max_cols: int,
    max_rows: int,
    min_band: int,
    max_band: int,
) -> None:
    new_data.set_name(f"Continuum Removal on {dataset.get_name()}")
    new_data.set_description(dataset.get_description())
    new_data.set_default_display_bands(_continuum_default_bands(dataset, max_band))
    new_data.set_data_ignore_value(dataset.get_data_ignore_value())

    spatial_metadata = dataset.get_spatial_metadata()
    if spatial_metadata.get_spatial_ref():
        new_spatial_metadata = SpatialMetadata.subset_to_window(
            spatial_metadata, dataset, min_rows, max_rows, min_cols, max_cols
        )
        new_data.copy_spatial_metadata(new_spatial_metadata)

    if dataset.has_wavelengths():
        min_band_wvl = dataset.band_list()[min_band]["wavelength"]
        max_band_wvl = dataset.band_list()[max_band - 1]["wavelength"]
        source_spectral_metadata = dataset.get_spectral_metadata()
        new_spectral_metadata = SpectralMetadata.subset_by_wavelength_range(
            source_spectral_metadata, min_band_wvl, max_band_wvl
        )
        new_data.copy_spectral_metadata(new_spectral_metadata)


def _compute_continuum_removal_image_direct(
    dataset: RasterDataSet,
    min_cols: int,
    min_rows: int,
    max_cols: int,
    max_rows: int,
    min_band: int,
    max_band: int,
    app_state: "ApplicationState",
) -> RasterDataSet:
    dband = max_band - min_band
    dcols = max_cols - min_cols
    drows = max_rows - min_rows
    image_data = dataset.get_image_data_subset(min_cols, min_rows, min_band, dcols, drows, dband)
    x_axis = _continuum_x_axis(dataset, min_band, max_band)

    cols = np.int32(max_cols - min_cols)
    rows = np.int32(max_rows - min_rows)
    bands = np.int32(max_band - min_band)
    image_data, x_axis = convert_to_float32_if_needed(image_data, x_axis)
    image_data = image_data.transpose(1, 2, 0)
    if not image_data.flags.c_contiguous:
        image_data = np.ascontiguousarray(image_data)
    if isinstance(image_data, np.ma.MaskedArray):
        mask = image_data.mask
        image_data = image_data.data
        image_data[mask] = np.nan
    if image_data.dtype != np.float32:
        image_data = image_data.astype(np.float32)
    if dataset.get_bad_bands() is not None:
        bad_bands_arr = np.array(dataset.get_bad_bands())
        bad_bands_arr = np.logical_not(bad_bands_arr)
    else:
        bad_bands_arr = np.array([0] * dataset.num_bands(), dtype=np.bool_)
    bad_bands_arr = bad_bands_arr[min_band:max_band]
    new_image_data = continuum_removal_image_numba(image_data, bad_bands_arr, x_axis, rows, cols, bands)

    raster_data = raster.RasterDataLoader()
    new_data = raster_data.dataset_from_numpy_array(new_image_data, app_state.get_cache())
    _apply_continuum_removed_metadata(
        new_data, dataset, min_cols, min_rows, max_cols, max_rows, min_band, max_band
    )
    return new_data


class ContinuumRemovalImageTask(QObject, SemanticTask):
    result_ready = Signal(object)

    def __init__(
        self,
        app_state: "ApplicationState",
        source_dataset: RasterDataSet,
        input_ref: DataRef,
        min_cols: int,
        min_rows: int,
        max_cols: int,
        max_rows: int,
        min_band: int,
        max_band: int,
        output_ref_name: str = "continuum_removed_image",
    ):
        QObject.__init__(self)
        SemanticTask.__init__(
            self,
            priority_class=PriorityClass.BACKGROUND,
            input_ref=input_ref,
            algorithm_pipeline=get_continuum_removal_image_pipeline(
                dataset_ref=input_ref,
                min_cols=min_cols,
                min_rows=min_rows,
                max_cols=max_cols,
                max_rows=max_rows,
                min_band=min_band,
                max_band=max_band,
                output_ref_name=output_ref_name,
            ),
            task_title="Continuum Removal",
            task_variables={
                "Dataset": source_dataset.get_name(),
                "Rows": f"{min_rows}:{max_rows}",
                "Cols": f"{min_cols}:{max_cols}",
                "Bands": f"{min_band}:{max_band}",
            },
        )
        self.id = app_state.take_next_id()
        self._app_state = app_state
        self._source_dataset = source_dataset
        self._min_cols = min_cols
        self._min_rows = min_rows
        self._max_cols = max_cols
        self._max_rows = max_rows
        self._min_band = min_band
        self._max_band = max_band
        self._output_ref_name = output_ref_name
        self.result_ready.connect(self._load_result_into_wiser)

    def completion_callback(self, bindings: dict[str, DataRef]) -> None:
        output_ref = bindings.get(self._output_ref_name)
        if output_ref is None:
            raise KeyError(f"Missing continuum removal output binding: {self._output_ref_name}")

        storage_client = get_process_storage_client()
        data_meta = storage_client.get_meta(output_ref)
        height, width, bands = data_meta.shape
        output_region = DatasetRegionRef(y0=0, y1=height, x0=0, x1=width, b0=0, b1=bands)
        reduced_data, _ = storage_client.read_region(output_ref, output_region, filter_data=False)
        self.result_ready.emit(np.asarray(reduced_data))

    @Slot(object)
    def _load_result_into_wiser(self, reduced_data: object) -> None:
        reduced_array = np.asarray(reduced_data).transpose(2, 0, 1)
        loader = self._app_state.get_loader()
        cache = self._app_state.get_cache()
        reduced_dataset = loader.dataset_from_numpy_array(reduced_array, cache)
        _apply_continuum_removed_metadata(
            reduced_dataset,
            self._source_dataset,
            self._min_cols,
            self._min_rows,
            self._max_cols,
            self._max_rows,
            self._min_band,
            self._max_band,
        )
        self._app_state.add_dataset(reduced_dataset)


class ContinuumRemovalPlugin(plugins.ContextMenuPlugin):
    """
    A Class to represents the continuum removal plugin. Can do continuum removal on a single spectrum
    or an image.

    Parameters
    ----------
    None

    Attributes
    ----------
    None
    """

    def __init__(self):
        logging.info("Continuum Removal")

    def add_context_menu_items(self, context_type: plugins.types.ContextMenuType, context_menu, context):
        """Adds plugin to WISER as a context menu type plugin

        Parameters
        ----------
        context_type: ContextMenuType
            the plugin type and where it can be used
        context_menu: PySide6.QtWidgets.QMenu
            the context menu available to the plugin
        context: dict
            Available WISER classes
        """

        if context_type == plugins.ContextMenuType.SPECTRUM_PICK:
            act1 = context_menu.addAction(context_menu.tr("Continuum Removal: Single Spectrum"))
            act1.triggered.connect(lambda checked=False: self.single_spectrum(context=context))

            act2 = context_menu.addAction(context_menu.tr("Continuum Removal: Collected Spectra"))
            act2.triggered.connect(lambda checked=False: self.collected_spectra(context=context))

        if context_type == plugins.ContextMenuType.RASTER_VIEW:
            act3 = context_menu.addAction(context_menu.tr("Continuum Removal: Image"))
            act3.triggered.connect(lambda checked=False: self.dimension(context=context))

    def error_box(self, message, context):
        """Displays desired error message and goes back to dimensions GUI when finished

        Parameters
        ----------
        message: str
            Error message to be displayed in the widget
        context: dict
            Available WISER classes
        """

        QMessageBox.critical(None, "Error", message, QMessageBox.Ok)

        self.dimension(context)

    def set_entire_image(self, dialog, cols, rows):
        """Sets dimensions in the dimensions GUI to include the entire image

        Parameters
        ----------
        dialog: PySide6.QtWidgets.QDialog
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
        dialog: PySide6.QtWidgets.QDialog
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
        combo: PySide6.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available bands
        spin: PySide6.QtWidgets.QAbstractSpinBox.QSpinBox
            Editable spin box that has a range of all available band numbers
        """

        idx = combo.currentIndex()
        spin.setValue(idx)

    def spin_box_changed(self, combo, spin):
        """Changes value in QComboBox to match with corresponding value in QSpinBox

        Parameters
        ----------
        combo: PySide6.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available bands
        spin: PySide6.QtWidgets.QAbstractSpinBox.QSpinBox
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

        dialog = QDialog()
        dialog._ui = Ui_ContinuumRemoval()
        dialog._ui.setupUi(dialog)

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
            lambda checked=True: self.set_entire_image(dialog, total_cols - 1, total_rows - 1)
        )

        all_bands = dialog.findChild(QPushButton, "all_bands")
        minimum = dialog.findChild(QComboBox, "min_bands")
        maximum = dialog.findChild(QComboBox, "max_bands")

        num_bands = len(dataset.band_list())
        bands = [dataset.get_band_label(i) for i in range(num_bands)]
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

        minimum.currentIndexChanged.connect(lambda checked=True: self.combo_box_changed(minimum, min_spin))
        min_spin.valueChanged.connect(lambda checked=True: self.spin_box_changed(minimum, min_spin))
        maximum.currentIndexChanged.connect(lambda checked=True: self.combo_box_changed(maximum, max_spin))
        max_spin.valueChanged.connect(lambda checked=True: self.spin_box_changed(maximum, max_spin))

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
                self.image(min_cols, min_rows, max_cols, max_rows, minimum, maximum, context)

    def plot_continuum_removal(
        self, spec_object: Spectrum, context: dict
    ) -> Tuple[raster.spectrum.NumPyArraySpectrum, raster.spectrum.NumPyArraySpectrum]:
        """Plots the continuum removed spectrum and the convex hull

        Parameters
        ----------
        spec_object: wiser.raster.Spectrum
            Spectrum to be continuum removed
        context: dict
            Available WISER classes
        """

        spectrum = spec_object.get_spectrum()
        if spectrum.dtype != np.float32:
            spectrum = spectrum.astype(np.float32)
        wavelengths_org = spec_object.get_wavelengths()  # type <astropy>
        if spec_object.has_wavelengths():
            wavelengths = np.array([i.value for i in wavelengths_org])
        else:
            # In this case, we get a list of ints for the band indices
            wavelengths = np.array(wavelengths_org)
        if wavelengths.dtype != np.float32:
            wavelengths = wavelengths.astype(np.float32)

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
        self,
        min_cols,
        min_rows,
        max_cols,
        max_rows,
        min_band,
        max_band,
        context,
        in_test_mode: bool = False,
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
        app_services: "AppServices" = context.get("app_services")

        if app_services is None:
            app = getattr(app_state, "_app", None)
            app_services = getattr(app, "_app_services", None)

        if app_services is None or in_test_mode:
            new_data = _compute_continuum_removal_image_direct(
                dataset, min_cols, min_rows, max_cols, max_rows, min_band, max_band, app_state
            )
            context["wiser"].add_dataset(new_data)
            return new_data

        dataset_ref = app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=dataset)
        )
        cr_task = ContinuumRemovalImageTask(
            app_state=app_state,
            source_dataset=dataset,
            input_ref=dataset_ref,
            min_cols=min_cols,
            min_rows=min_rows,
            max_cols=max_cols,
            max_rows=max_rows,
            min_band=min_band,
            max_band=max_band,
        )
        self._last_continuum_removal_task = cr_task
        task_plan = app_services.task_planner.plan_semantic_task(cr_task)
        return app_services.task_manager.register_and_submit_task_plan(app_services.scheduler, task_plan)
