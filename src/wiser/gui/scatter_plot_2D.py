"""2D Scatter Plot Plugin

This script allows the user to create a density sliced 2D scatter plot from 2 bands of an
image or of 2 different images of the same dimension.

This script uses datasets and images to refer to the same thing: hyperspectral images

This script requires that `numpy`, `matplotlib`, `mpl_scatter_density `, and `pyside2`
be installed within the Python environment you are running this script in.

This script requires the following .ui files to be in the same folder as this python script:
    * error.ui - GUI for error message
    * layer_stacking.ui - GUI for dataset/image selection
    * dimensions.ui - GUI for dimension range selection
    * bands_table.ui - GUI for band selection

Primarily written by Amy Wang
"""

import matplotlib

from wiser.raster.serializable import SerializedForm


matplotlib.use("Qt5Agg")

import mpl_scatter_density  # adds projection='scatter_density'
import numpy as np
import logging
import os
import multiprocessing as mp
import multiprocessing.connection as mp_conn
import matplotlib.pyplot as plt

from typing import Callable, TYPE_CHECKING, Optional, List, Tuple

from matplotlib.pyplot import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from .generated.scatter_plot_axes_ui import Ui_ScatterPlotAxes
from .generated.scatter_plot_colormap_ui import Ui_ScatterPlotColormap
from .generated.interactive_scatter_plot_ui import Ui_ScatterPlotDialog

from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

from wiser.gui.loading_overlay import LoadingOverlay
from wiser.gui.parallel_task import ParallelTaskProcess
from wiser.gui.subprocessing_manager import ProcessManager
from wiser.gui.util import get_random_matplotlib_color, get_color_icon

from wiser.raster.dataset import RasterDataSet
from wiser.raster.selection import MultiPixelSelection
from wiser.raster.roi import RegionOfInterest

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

BOTTOM_SUBPLOT_MARGIN = 0.2

# Default colormap and density slice colors
# Modified version found here https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
WHITE_VIRDIS = LinearSegmentedColormap.from_list(
    "WHITE_VIRDIS",
    [
        (0, "#ffffff"),
        (1e-20, "#440053"),
        (0.1, "#404388"),
        (0.2, "#2a788e"),
        (0.4, "#21a784"),
        (0.6, "#78d151"),
        (0.8, "#fde624"),
        (1, "#f44336"),
    ],
    N=256,
)

DEFAULT_COLOR = ("white viridis", WHITE_VIRDIS)


def _create_scatter_plot_intensive_operations(
    x_dataset_serialized: SerializedForm,
    y_dataset_serialized: SerializedForm,
    x_band_idx: int,
    y_band_idx: int,
    child_conn: mp_conn.Connection,
    return_queue: mp.Queue,
):
    x_dataset = RasterDataSet.deserialize_into_class(
        x_dataset_serialized.get_serialize_value(), x_dataset_serialized.get_metadata()
    )
    y_dataset = RasterDataSet.deserialize_into_class(
        y_dataset_serialized.get_serialize_value(), y_dataset_serialized.get_metadata()
    )
    x_band = x_dataset.get_band_data(x_band_idx)
    y_band = y_dataset.get_band_data(y_band_idx)

    cols1 = x_dataset.get_width()
    rows1 = x_dataset.get_height()
    cols2 = y_dataset.get_width()
    rows2 = y_dataset.get_height()

    x = x_band.reshape(rows1 * cols1)
    y = y_band.reshape(rows2 * cols2)

    # Safe mins/maxes for axes panel defaults
    new_x = np.array([n for n in x if np.isnan(n) == False])
    new_y = np.array([m for m in y if np.isnan(m) == False])
    default_x_min = np.nanmin(new_x)
    default_x_max = np.nanmax(new_x)
    default_y_min = np.nanmin(new_y)
    default_y_max = np.nanmax(new_y)

    return_queue.put(
        {
            "default_x_min": default_x_min,
            "default_x_max": default_x_max,
            "default_y_min": default_y_min,
            "default_y_max": default_y_max,
            "rows": rows1,
            "cols": cols1,
            "x_flat": x,
            "y_flat": y,
            "xy": np.column_stack((x, y)),
            "valid_mask": np.isfinite(np.column_stack((x, y))).all(axis=1),
        }
    )


class ScatterPlot2DDialog(QDialog):
    """
    A Class to represents the density sliced 2D scatter plot of 2 bands

    Parameters
    ----------
    None

    Attributes
    ----------
    colormap_choice: matplotlib.colors.LinearSegmentedColormap or str
        Colormap for the density slice on the scatter plot. Default is viridis
    x_min: int
        Minimum value on the x-axis
    x_max: int
        Maximum value on the x-axis
    y_min: int
        Minimum value on the y-axis
    y_max: int
        Maximum value on the y-axis
    n: int
        Number of time the same 2 bands have been plotted
    """

    def __init__(
        self,
        interactive_callback: Callable,
        clear_interactive_callback: Callable,
        app_state: "ApplicationState",
        testing=False,
        parent=None,
    ):
        super().__init__(parent=parent)
        self._ui = Ui_ScatterPlotDialog()
        self._ui.setupUi(self)

        self._loader = LoadingOverlay(self._ui.wdgt_plot)
        self._process_manager = None

        self._app_state = app_state
        self._app_state.dataset_added.connect(self._on_dataset_added)
        self._app_state.dataset_removed.connect(self._on_dataset_removed)

        self._interactive_callback = interactive_callback
        self._clear_interactive_callback = clear_interactive_callback
        self._colormap_choice: Tuple[str, LinearSegmentedColormap] = DEFAULT_COLOR
        self._x_min = None
        self._x_max = None
        self._y_min = None
        self._y_max = None
        self._n = 0

        # --- selection state ---
        self._ax = None
        self._canvas = None
        self._selector = None
        self._sel_artist = None
        self._count_label = None

        self._x_flat = None
        self._y_flat = None
        self._xy = None
        self._valid_mask = None
        self._selected_idx = np.array([], dtype=int)
        self._rows = 0
        self._cols = 0

        # --- testing state ---
        self._testing = testing

        # Plot mode: start with density scatter plot
        self._use_density_scatter = True
        # default highlight color for polygon selections (matches MainView default)
        self._highlight_color_str = "#ff0000"

        self._init_band_dataset_choosers()
        self._init_plot()

    def _init_plot(self):
        self._figure = Figure()
        # We the plot to have space on the bottom to account for the
        # band name and dataset name being on two lines.
        self._figure.subplots_adjust(bottom=BOTTOM_SUBPLOT_MARGIN)
        canvas = FigureCanvas(self._figure)
        self._canvas = canvas  # keep a handle for redraws
        self._navi_toolbar = NavigationToolbar(canvas, None)

        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        self._count_label = QLabel(self.tr("0 pts"))
        self._btn_clear = QPushButton(self.tr("Clear selection"))
        self._btn_create_roi = QPushButton(self.tr("Create ROI from Selection"))
        self._btn_change_cmap = QPushButton(self.tr("Color Map"))
        self._btn_change_axes = QPushButton(self.tr("Axes Limits"))
        self._btn_toggle_density = QPushButton(self.tr("To Scatter"))
        self._btn_highlight_color = QPushButton()
        self._btn_highlight_color.setToolTip(self.tr("Set highlight color"))
        self._btn_highlight_color.setIcon(get_color_icon(self._highlight_color_str))
        self._btn_highlight_color.setText(self.tr("Highlight"))
        ctrl_layout.addWidget(self._count_label)
        ctrl_layout.addStretch(1)
        ctrl_layout.addWidget(self._btn_clear)
        ctrl_layout.addWidget(self._btn_create_roi)
        ctrl_layout.addWidget(self._btn_change_cmap)
        ctrl_layout.addWidget(self._btn_change_axes)
        ctrl_layout.addWidget(self._btn_toggle_density)
        ctrl_layout.addWidget(self._btn_highlight_color)
        ctrl.setMaximumHeight(45)

        self._btn_change_cmap.clicked.connect(lambda checked=True: self._colormap_chooser())

        self._btn_change_axes.clicked.connect(
            lambda checked=True: QMessageBox.information(
                self,
                self.tr("Information"),
                self.tr("No plot to change axes limits of"),
            )
        )

        # Toggle density/regular scatter
        self._btn_toggle_density.clicked.connect(self._on_toggle_density_clicked)

        # Choose highlight color for polygon selection
        self._btn_highlight_color.clicked.connect(self._on_choose_highlight_color)

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addWidget(self._navi_toolbar)
        layout.addWidget(ctrl)  # <-- add control strip just under the toolbar
        layout.addWidget(canvas)
        self._ui.wdgt_plot.setLayout(layout)

        self._btn_clear.clicked.connect(self._clear_selection_overlay)
        self._btn_create_roi.clicked.connect(self._create_roi_from_selection)

        self._ui.btn_create_plot.clicked.connect(lambda checked: self._create_scatter_plot())

    def _on_choose_highlight_color(self):
        color = QColorDialog.getColor(parent=self, initial=QColor(self._highlight_color_str))
        if color.isValid():
            self._highlight_color_str = color.name()
            self._btn_highlight_color.setIcon(get_color_icon(self._highlight_color_str))
            if self._sel_artist is not None:
                try:
                    self._sel_artist.set_markeredgecolor(self._highlight_color_str)
                except Exception:
                    pass
            if self._canvas is not None:
                self._canvas.draw_idle()

    def _init_band_dataset_choosers(self):
        """Initializes the band and dataset choosers for the x and y axes to have
        the datasets that the app currently has.
        """
        cbox_x_dataset = self._ui.cbox_x_dataset
        cbox_y_dataset = self._ui.cbox_y_dataset
        cbox_render_ds = self._ui.cbox_render_ds

        datasets = self._app_state.get_datasets()

        cbox_x_dataset.clear()
        cbox_x_dataset.addItem(self.tr("(no data)"), -1)
        for dataset in datasets:
            cbox_x_dataset.addItem(dataset.get_name(), dataset.get_id())
        cbox_x_dataset.currentIndexChanged.connect(lambda checked: self._on_cbox_dataset_changed())

        cbox_y_dataset.clear()
        cbox_y_dataset.addItem(self.tr("(no data)"), -1)
        for dataset in datasets:
            cbox_y_dataset.addItem(dataset.get_name(), dataset.get_id())
        cbox_y_dataset.currentIndexChanged.connect(lambda checked: self._on_cbox_dataset_changed())

        cbox_render_ds.clear()
        cbox_render_ds.addItem(self.tr("(no data)"), -1)
        for dataset in datasets:
            cbox_render_ds.addItem(dataset.get_name(), dataset.get_id())

        self._sync_band_choosers()

    def _sync_band_choosers(self):
        """Synchronizes the band choosers for the x and y axes with
        the datasets chosen for the x and y axes.
        """
        cbox_x_band = self._ui.cbox_x_band
        spin_box_x_band = self._ui.sbox_x_band_number

        cbox_y_band = self._ui.cbox_y_band
        spin_box_y_band = self._ui.sbox_y_band_number

        x_dataset_id = self._ui.cbox_x_dataset.currentData()
        if self._app_state.has_dataset(x_dataset_id):
            dataset = self._app_state.get_dataset(x_dataset_id)
            bands = dataset.band_list()
            descriptions = list([band["description"] for band in bands])
            if descriptions[0]:
                band_descriptions = list(
                    (
                        f"Band {descriptions.index(descr)}: " + descr,
                        descriptions.index(descr),
                    )
                    for descr in descriptions
                )
            else:
                band_descriptions = list((f"Band {i}", i) for i in range(len(descriptions)))
            cbox_x_band.clear()
            for descr, index in band_descriptions:
                cbox_x_band.addItem(self.tr(f"{descr}"), index)
            spin_box_x_band.setRange(0, len(band_descriptions) - 1)
        else:
            cbox_x_band.clear()

        y_dataset_id = self._ui.cbox_y_dataset.currentData()
        if self._app_state.has_dataset(y_dataset_id):
            dataset = self._app_state.get_dataset(y_dataset_id)
            bands = dataset.band_list()
            descriptions = list([band["description"] for band in bands])
            if descriptions[0]:
                band_descriptions = list(
                    (
                        f"Band {descriptions.index(descr)}: " + descr,
                        descriptions.index(descr),
                    )
                    for descr in descriptions
                )
            else:
                band_descriptions = list((f"Band {i}", i) for i in range(len(descriptions)))
            cbox_y_band.clear()
            for descr, index in band_descriptions:
                cbox_y_band.addItem(self.tr(f"{descr}"), index)
            spin_box_y_band.setRange(0, len(band_descriptions) - 1)
        else:
            cbox_y_band.clear()

        # Keeps the combo box and spin box in sync
        cbox_x_band.currentIndexChanged.connect(
            lambda checked: self._combo_box_changed(cbox_x_band, spin_box_x_band)
        )
        spin_box_x_band.valueChanged.connect(
            lambda checked: self._spin_box_changed(cbox_x_band, spin_box_x_band)
        )

        # Keeps the combo box and spin box in sync
        cbox_y_band.currentIndexChanged.connect(
            lambda checked: self._combo_box_changed(cbox_y_band, spin_box_y_band)
        )
        spin_box_y_band.valueChanged.connect(
            lambda checked: self._spin_box_changed(cbox_y_band, spin_box_y_band)
        )

    def _check_bands(self) -> bool:
        """Checks to make sure we have valid bands for the x and y axes."""
        x_dataset = self.get_x_dataset()
        y_dataset = self.get_y_dataset()
        if x_dataset is None or y_dataset is None:
            return False
        x_band_idx = self._ui.cbox_x_band.currentData()
        y_band_idx = self._ui.cbox_y_band.currentData()
        if x_band_idx is None or y_band_idx is None:
            return False
        return True

    def _check_dataset_compatibility(self) -> Optional[List[str]]:
        """
        Checks to make sure the datasets are compatible and returns a list of errors if they are not.
        If no list of errors is returned, the datasets are compatible.
        """
        errors = []
        x_id = self._ui.cbox_x_dataset.currentData()
        if not self._app_state.has_dataset(x_id):
            errors.append("X dataset not found")
        else:
            x_dataset = self._app_state.get_dataset(x_id)
            x_dataset_dims = (x_dataset.get_width(), x_dataset.get_height())

        y_id = self._ui.cbox_y_dataset.currentData()
        if not self._app_state.has_dataset(y_id):
            errors.append("Y dataset not found")
        else:
            y_dataset = self._app_state.get_dataset(y_id)
            y_dataset_dims = (y_dataset.get_width(), y_dataset.get_height())

        render_id = self._ui.cbox_render_ds.currentData()
        if not self._app_state.has_dataset(render_id):
            errors.append("Render dataset not found")
        else:
            render_dataset = self._app_state.get_dataset(render_id)
            render_dataset_dims = (
                render_dataset.get_width(),
                render_dataset.get_height(),
            )
        if not errors and x_dataset_dims != render_dataset_dims and y_dataset_dims != render_dataset_dims:
            errors.append(
                "X dataset, Y dataset, and render dataset must have the same dimensions\n"
                f"X dataset dimensions: {x_dataset_dims}\n"
                f"Y dataset dimensions: {y_dataset_dims}\n"
                f"Render dataset dimensions: {render_dataset_dims}"
            )
        return errors

    def get_x_band(self) -> Optional[np.ndarray]:
        x_dataset = self.get_x_dataset()
        if x_dataset is None:
            return None
        idx = self._ui.cbox_x_band.currentData()
        if idx is None:
            return None
        return x_dataset.get_band_data(idx)

    def get_y_band(self) -> Optional[np.ndarray]:
        y_dataset = self.get_y_dataset()
        if y_dataset is None:
            return None
        idx = self._ui.cbox_y_band.currentData()
        if idx is None:
            return None
        return y_dataset.get_band_data(idx)

    def get_x_dataset(self) -> Optional["RasterDataSet"]:
        idx = self._ui.cbox_x_dataset.currentData()
        if not self._app_state.has_dataset(idx):
            return None
        return self._app_state.get_dataset(idx)

    def get_y_dataset(self) -> Optional["RasterDataSet"]:
        idx = self._ui.cbox_y_dataset.currentData()
        if not self._app_state.has_dataset(idx):
            return None
        return self._app_state.get_dataset(idx)

    def _on_dataset_added(self, ds_id: int):
        self._init_band_dataset_choosers()

    def _on_dataset_removed(self, ds_id: int):
        self._init_band_dataset_choosers()

    def _on_cbox_dataset_changed(self):
        self._sync_band_choosers()

    def _colormap_chooser(self):
        dialog_cmap_chooser = QDialog(self)
        ui_cmap_chooser = Ui_ScatterPlotColormap()
        ui_cmap_chooser.setupUi(dialog_cmap_chooser)
        dialog_cmap_chooser.setFixedSize(dialog_cmap_chooser.size())
        img_cmap = ui_cmap_chooser.img_cmap
        cbox_cmap = ui_cmap_chooser.cbox_cmap

        cbox_cmap.addItem(DEFAULT_COLOR[0], DEFAULT_COLOR[1])
        for cmap_text in plt.colormaps():
            cbox_cmap.addItem(cmap_text, cm.get_cmap(cmap_text, 256))

        # Ensure we select the current color choice in the combo box
        idx = cbox_cmap.findText(self._colormap_choice[0])
        if idx != -1:
            cbox_cmap.setCurrentIndex(idx)
        self._colormap_images(cbox_cmap, img_cmap)
        # In case the color was not there, we set our color map choice to whatever the
        # combo box is currently set to.
        self._colormap_choice = (cbox_cmap.currentText(), cbox_cmap.currentData())

        cbox_cmap.currentTextChanged.connect(
            lambda checked=True: self._colormap_img_changed(cbox_cmap, img_cmap)
        )

        if dialog_cmap_chooser.exec() == QDialog.Accepted:
            self._save_colormap_choice(cbox_cmap)
            self._create_scatter_plot()

    def _default_axes(
        self,
        x_min: QSpinBox,
        x_max: QSpinBox,
        y_min: QSpinBox,
        y_max: QSpinBox,
        default_x_min: int,
        default_x_max: int,
        default_y_min: int,
        default_y_max: int,
    ):
        """Sets axes limits to the largest and smallest possible values for each axis

        Parameters
        ----------
        x_min: PySide2.QtWidgets.QAbstractSpinBox.QDoubleSpinBox
            Editable QDoubleSpinBox that holds the minimum x-axis limit
        x_max: PySide2.QtWidgets.QAbstractSpinBox.QDoubleSpinBox
            Editable QDoubleSpinBox that holds the maximum x-axis limit
        y_min: PySide2.QtWidgets.QAbstractSpinBox.QDoubleSpinBox
            Editable QDoubleSpinBox that holds the minimum y-axis limit
        y_max: PySide2.QtWidgets.QAbstractSpinBox.QDoubleSpinBox
            Editable QDoubleSpinBox that holds the maximum y-axis limit
        default_x_min: int
            Smallest value of all x values
        default_x_max: int
            Largest value of all x values
        default_y_min: int
            Smallest value of all y values
        default_y_max: int
            Largest value of all y values
        """

        x_range = max(default_x_max - default_x_min, 0)
        y_range = max(default_y_max - default_y_min, 0)
        x_min.setValue(default_x_min - x_range / 10)
        x_max.setValue(default_x_max + x_range / 10)
        y_min.setValue(default_y_min - y_range / 10)
        y_max.setValue(default_y_max + y_range / 10)

    def _axes_chooser(self, default_x_min, default_x_max, default_y_min, default_y_max):
        """Displays GUI that allows user to choose the axes limits

        Parameters
        ----------
        b1: int
            Band number for user chosen band on the x-axis
        b2: int
            Band number for user chosen band on the y-axis
        i1: int
            Index of user chosen image in all images uploaded on WISER
        i2: int
            Index of user chosen image in all images uploaded on WISER
        default_x_min: int
            Smallest value of all x values
        default_x_max: int
            Largest value of all x values
        default_y_min: int
            Smallest value of all y values
        default_y_max: int
            Largest value of all y values
        context: dict
            Available WISER classes
        """
        dialog = QDialog(self)
        ui = Ui_ScatterPlotAxes()
        ui.setupUi(dialog)
        dialog.setFixedSize(dialog.size())
        default = ui.default_axes
        x_min = ui.x_min
        x_max = ui.x_max
        y_min = ui.y_min
        y_max = ui.y_max

        x_min.setRange(-1000000, 1000000)
        x_max.setRange(-1000000, 1000000)
        y_min.setRange(-1000000, 1000000)
        y_max.setRange(-1000000, 1000000)

        x_min.setValue(self._ax.get_xlim()[0])
        x_max.setValue(self._ax.get_xlim()[1])
        y_min.setValue(self._ax.get_ylim()[0])
        y_max.setValue(self._ax.get_ylim()[1])

        default.clicked.connect(
            lambda checked=True: self._default_axes(
                x_min,
                x_max,
                y_min,
                y_max,
                default_x_min,
                default_x_max,
                default_y_min,
                default_y_max,
            )
        )

        if dialog.exec() == QDialog.Accepted:
            if (x_min.value() >= x_max.value()) or (y_min.value() >= y_max.value()):
                QMessageBox.warning(
                    self,
                    self.tr("Error"),
                    self.tr(
                        f"Minimum must be less than the maximum\n"
                        f"X min: {x_min.value()}\tX max: {x_max.value()}\n"
                        f"Y min: {y_min.value()}\tY max: {y_max.value()}"
                    ),
                )

            else:
                self._ax.set_xlim(x_min.value(), x_max.value())
                self._ax.set_ylim(y_min.value(), y_max.value())
                self._canvas.draw_idle()

    def _colormap_img_changed(self, cmap_box: QComboBox, cmap_img: QLabel):
        """Sets colormap_choice to the user chosen colormap

        Parameters
        ----------
        cmap_box: PySide2.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available colormap names
        colormap_img: PySide2.QtWidgets.QLabel
            Label that contains an image of the chosen colormap within the QComboBox
        """
        self._colormap_images(cmap_box, cmap_img)

    def _save_colormap_choice(self, cmap_box: QComboBox):
        """Saves the colormap choice to the class variable _colormap_choice

        Parameters
        ----------
        cmap_box: PySide2.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available colormap names
        """
        self._colormap_choice = (cmap_box.currentText(), cmap_box.currentData())

    def _colormap_images(self, colormap_box: QComboBox, colormap_img: QLabel):
        """Code provided by Donnie Pinkston
        Changes the colormap GUI to indicate which colormap the user has chosen

        Parameters
        ----------
        colormapbox: PySide2.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available colormap names
        colormap_img: PySide2.QtWidgets.QLabel
            Label that contains an image of the chosen colormap within the QComboBox
        """

        cmap = colormap_box.currentData()
        img = QImage(cmap.N, 24, QImage.Format_RGB32)
        for x in range(cmap.N):
            rgba = cmap(x, bytes=True)
            c = QColor(rgba[0], rgba[1], rgba[2])
            for y in range(img.height()):
                img.setPixelColor(x, y, c)

        colormap_img.setPixmap(QPixmap.fromImage(img))

    def _image_changed(self, datasets, image, combo, spin):
        """Changes value in QComboBox to display the user chosen option

        Parameters
        ----------
        datasets: list
            List of all available datasets in WISER
        image: PySide2.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available datasets
        combo: PySide2.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available bands
        spin: PySide2.QtWidgets.QAbstractSpinBox.QSpinBox
            Editable spin box that has a range of all available band numbers
        """

        image_index = image.currentIndex()
        image = datasets[image_index]
        bands = image.band_list()
        bands = list([i["description"] for i in bands])
        bands = list(f"Band {bands.index(i)}: " + i for i in bands)

        combo.clear()
        combo.addItems(bands)
        spin.setRange(0, len(bands) - 1)

    def _combo_box_changed(self, combo, spin):
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

    def _spin_box_changed(self, combo, spin):
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

    def _create_scatter_plot(self):
        errors = self._check_dataset_compatibility()
        if errors:
            QMessageBox.warning(self, self.tr("Error"), self.tr("\n\n".join(errors)))
            return
        bands_valid = self._check_bands()
        if not bands_valid:
            self._clear_scatter_density_plot()
            return
        self._loader.start()
        # self._check_bands() above lets us get dataset and bands without worrying
        x_dataset = self.get_x_dataset()
        y_dataset = self.get_y_dataset()
        x_band_idx = self._ui.cbox_x_band.currentData()
        y_band_idx = self._ui.cbox_y_band.currentData()
        kwargs = {
            "x_dataset_serialized": x_dataset.get_serialized_form(),
            "y_dataset_serialized": y_dataset.get_serialized_form(),
            "x_band_idx": x_band_idx,
            "y_band_idx": y_band_idx,
        }
        self._process_manager = ProcessManager(_create_scatter_plot_intensive_operations, kwargs)
        task = self._process_manager.get_task()
        task.succeeded.connect(self._create_scatter_plot_gui_updates)
        task.error.connect(self._on_create_scatter_plot_error)
        self._process_manager.start_task(blocking=self._testing)

    def _on_create_scatter_plot_error(self, task: ParallelTaskProcess):
        QMessageBox.critical(
            self,
            self.tr("Error"),
            self.tr(f"An error occurred while creating the scatter plot.\n\nError:\n{task.get_error()}"),
        )

    def _create_scatter_plot_gui_updates(self, task: ParallelTaskProcess):
        result = task.get_result()
        self._rows, self._cols = result["rows"], result["cols"]
        self._x_flat = result["x_flat"]
        self._y_flat = result["y_flat"]
        self._xy = result["xy"]
        self._valid_mask = result["valid_mask"]

        default_x_min = result["default_x_min"]
        default_x_max = result["default_x_max"]
        default_y_min = result["default_y_min"]
        default_y_max = result["default_y_max"]

        self._btn_change_axes.clicked.disconnect()
        self._btn_change_axes.clicked.connect(
            lambda checked=True: self._axes_chooser(
                default_x_min, default_x_max, default_y_min, default_y_max
            )
        )

        # Since we just created a new plot, we set the min and max to the
        # plot's min and max. We only use these values when we first
        # create the plot
        self._x_min = default_x_min
        self._x_max = default_x_max
        self._y_min = default_y_min
        self._y_max = default_y_max

        x_dataset = self.get_x_dataset()
        y_dataset = self.get_y_dataset()
        x_band_idx = self._ui.cbox_x_band.currentData()
        y_band_idx = self._ui.cbox_y_band.currentData()

        # --- draw density plot and keep axes ---

        # Create the display string for the x and y bands
        x_wvl = x_dataset.get_band_info()[x_band_idx].get("description", None)
        x_wvl_str = f": {x_wvl}" if x_wvl else ""
        x_band_description = f"{x_dataset.get_name()}\nBand {x_band_idx}" + x_wvl_str

        y_wvl = y_dataset.get_band_info()[y_band_idx].get("description", None)
        y_wvl_str = f": {y_wvl}" if y_wvl else ""
        y_band_description = f"{y_dataset.get_name()}\nBand {y_band_idx}" + y_wvl_str
        if self._use_density_scatter:
            ax = self._using_mpl_scatter_density(
                self._figure,
                self._x_flat[self._valid_mask],
                self._y_flat[self._valid_mask],
                x_band_description,
                y_band_description,
                self._colormap_choice,
            )
        else:
            ax = self._using_mpl_scatter(
                self._figure,
                self._x_flat[self._valid_mask],
                self._y_flat[self._valid_mask],
                x_band_description,
                y_band_description,
                self._colormap_choice,
            )
        self._ax = ax
        self._canvas.draw_idle()
        self._loader.stop()

        self._selector = PolygonSelector(
            self._ax,
            self._on_polygon_select,
            useblit=True,
        )

        self._btn_clear.clicked.connect(self._clear_selection_overlay)

    def _on_toggle_density_clicked(self):
        """Toggle between density scatter and regular scatter, then rebuild the plot."""
        self._use_density_scatter = not self._use_density_scatter
        # Update button label to reflect next action
        if self._use_density_scatter:
            self._btn_toggle_density.setText(self.tr("To Scatter"))
        else:
            self._btn_toggle_density.setText(self.tr("To Density"))
        # Recreate the plot with current settings
        self._create_scatter_plot()

    def _using_mpl_scatter_density(
        self,
        fig: Figure,
        x: np.ndarray,
        y: np.ndarray,
        b1_description: str,
        b2_description: str,
        colormap: Tuple[str, LinearSegmentedColormap],
    ):
        """Modified version found here https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
        Creates a scatter plot of x and y and density slices it to show density of points plotted

        Parameters
        ----------
        fig: matplotlib.figure
            Figure to be plotted on
        x: ndarray
            All y-axis values
        y: ndarray
            All y-axis values
        b1_description: str
            Band number for user chosen band on the x-axis
        b2_description: str
            Band number for user chosen band on the y-axis
            Band number for user chosen band on the y-axis
        colormap: matplotlib.colors.LinearSegmentedColormap or str
            Colormap for the density slice on the scatter plot. Default is DEFAULT_COLOR
        """
        fig.clear()
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        ax.set_xlabel(b1_description)
        ax.set_ylabel(b2_description)
        ax.set_title("2D scatter plot of two bands with colormap")
        density = ax.scatter_density(x, y, cmap=colormap[1])
        x_range = max(self._x_max - self._x_min, 0)
        y_range = max(self._y_max - self._y_min, 0)
        ax.set_xlim(self._x_min - x_range / 10, self._x_max + x_range / 10)
        ax.set_ylim(self._y_min - y_range / 10, self._y_max + y_range / 10)
        fig.colorbar(density, label="Number of points per spectral value")

        return ax

    def _using_mpl_scatter(
        self,
        fig: Figure,
        x: np.ndarray,
        y: np.ndarray,
        b1_description: str,
        b2_description: str,
        colormap: Tuple[str, LinearSegmentedColormap],
    ):
        """Create a regular Matplotlib scatter plot (no density projection).

        Parameters mirror _using_mpl_scatter_density so callers can swap either.
        """
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(b1_description)
        ax.set_ylabel(b2_description)
        ax.set_title("2D scatter plot of two bands")
        # Use a small marker size for large datasets; single color for simplicity
        ax.scatter(x, y, s=1, c="#1f77b4", alpha=0.6, linewidths=0)
        x_range = max(self._x_max - self._x_min, 0)
        y_range = max(self._y_max - self._y_min, 0)
        ax.set_xlim(self._x_min - x_range / 10, self._x_max + x_range / 10)
        ax.set_ylim(self._y_min - y_range / 10, self._y_max + y_range / 10)

        return ax

    def _clear_scatter_density_plot(self):
        fig = self._figure
        fig.clear()

    def _on_polygon_select(self, verts):
        """
        Callback for PolygonSelector: updates selected indices, overlays a small
        scatter for visual feedback, and updates the live count.
        Finish the polygon with a double click or the 'enter' key.
        """
        if self._xy is None:
            return
        path = Path(verts)
        in_poly = path.contains_points(self._xy[self._valid_mask])
        # Map back to full-length mask
        mask = np.zeros(self._xy.shape[0], dtype=bool)
        mask[np.nonzero(self._valid_mask)[0]] = in_poly
        self._selected_idx = np.nonzero(mask)[0]

        # Visual feedback: overlay up to N points to avoid rendering millions
        if self._sel_artist is not None:
            try:
                self._sel_artist.remove()
            except Exception:
                pass
            self._sel_artist = None

        n_sel = int(mask.sum())
        self._count_label.setText(f"{n_sel:,} pts")

        if n_sel > 0:
            show_cap = 5000
            idx_to_show = self._selected_idx
            if len(idx_to_show) > show_cap:
                idx_to_show = np.random.choice(idx_to_show, size=show_cap, replace=False)

        self._sel_artist = self._ax.plot(
            self._x_flat[idx_to_show],
            self._y_flat[idx_to_show],
            marker="o",
            linestyle="None",
            markersize=2,
            alpha=0.7,
            markerfacecolor="none",
            markeredgecolor=self._highlight_color_str,
        )[0]

        if self._canvas is not None:
            self._canvas.draw_idle()

        render_ds_id = self._ui.cbox_render_ds.currentData()
        self._interactive_callback(self.get_selected_points(), render_ds_id, self._highlight_color_str)

    def _clear_selection_overlay(self):
        """Clear current selection, remove overlay, and reset polygon vertices."""
        # 1) clear stored indices + UI count
        self._selected_idx = np.array([], dtype=int)
        if self._count_label is not None:
            self._count_label.setText("0 pts")

        # 2) remove the red feedback overlay
        if self._sel_artist is not None:
            try:
                self._sel_artist.remove()
            except Exception:
                pass
            self._sel_artist = None

        # 3) clear the polygon itself
        if self._selector is not None:
            cleared = False
            # Preferred: built-in clear() (Matplotlib >= ~3.3)
            if hasattr(self._selector, "clear"):
                try:
                    self._selector.clear()
                    cleared = True
                except Exception:
                    cleared = False

            # Fallback: recreate the selector fresh
            if not cleared:
                try:
                    self._selector.disconnect_events()
                except Exception:
                    pass
                self._selector = PolygonSelector(
                    self._ax,
                    self._on_polygon_select,
                    useblit=True,
                )

        # 4) redraw
        if self._canvas is not None:
            self._canvas.draw_idle()

        self._clear_interactive_callback()

    def _create_roi_from_selection(self):
        """Creates a ROI from the selection"""
        if self._selected_idx is None or len(self._selected_idx) == 0:
            QMessageBox.information(self, self.tr("No selection"), self.tr("No points are selected."))
            return
        rows, cols, _, _ = self.get_selected_points()
        coords = zip(rows.tolist(), cols.tolist())
        # We switch because MultiPixelSelection expects the points to be in the
        # format (column, row)
        points = [QPoint(coord[1], coord[0]) for coord in coords]
        selection = MultiPixelSelection(points)
        roi = RegionOfInterest(
            self._app_state.unique_roi_name(self.tr("Scatter Plot Selection")),
            color=get_random_matplotlib_color(),
        )
        roi.add_selection(selection)
        self._app_state.add_roi(roi)

    def _save_selection(self):
        """
        Save selected pixels as CSV with columns:
        row, col, x_band_value, y_band_value
        """
        if self._selected_idx is None or len(self._selected_idx) == 0:
            QMessageBox.information(self._dlg, "No selection", "No points are selected.")
            return

        rows, cols = np.unravel_index(self._selected_idx, (self._rows, self._cols))
        x_vals = self._x_flat[self._selected_idx]
        y_vals = self._y_flat[self._selected_idx]
        arr = np.column_stack([rows, cols, x_vals, y_vals])

        path, _ = QFileDialog.getSaveFileName(self._dlg, "Save selected points", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            np.savetxt(path, arr, delimiter=",", header="row,col,x_band,y_band", comments="")
            QMessageBox.information(self._dlg, "Saved", f"Saved {arr.shape[0]:,} points.")
        except Exception as e:
            QMessageBox.critical(self._dlg, "Save failed", str(e))

    def get_selected_points(self):
        """
        Programmatic access:
        Returns (rows, cols, x_band_values, y_band_values)
        for the *current* polygon selection.
        """
        if self._selected_idx is None or len(self._selected_idx) == 0:
            return (np.array([], dtype=int),) * 2 + (np.array([]),) * 2
        rows, cols = np.unravel_index(self._selected_idx, (self._rows, self._cols))
        return (
            rows,
            cols,
            self._x_flat[self._selected_idx],
            self._y_flat[self._selected_idx],
        )

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key_Escape:
            self._clear_selection_overlay()
            e.accept()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        super().closeEvent(e)
        self._clear_interactive_callback()
