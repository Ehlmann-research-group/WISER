"""2D Scatter Plot Plugin

This script allows the user to create a density sliced 2D scatter plot from 2 bands of an
image or of 2 different images of the same dimension.

This script uses datasets and images to refer to the same thing: hyperspectral images

This script requires that `numpy`, `matplotlib`, `mpl_scatter_density `, and `pyside2` be installed within the Python
environment you are running this script in.

This script requires the following .ui files to be in the same folder as this python script:
    * error.ui - GUI for error message
    * layer_stacking.ui - GUI for dataset/image selection
    * dimensions.ui - GUI for dimension range selection
    * bands_table.ui - GUI for band selection

Primarily written by Amy Wang
"""

import matplotlib


matplotlib.use("Qt5Agg")

import mpl_scatter_density  # adds projection='scatter_density'
import numpy as np
import logging
import os
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

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.dataset import RasterDataSet

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

DEFAULT_COLOR = ('white viridis', WHITE_VIRDIS)


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

    def __init__(self, interactive_callback: Callable,
                 clear_interactive_callback: Callable,
                 app_state: 'ApplicationState', parent=None):
        super().__init__(parent=parent)
        self._ui = Ui_ScatterPlotDialog()
        self._ui.setupUi(self)

        self._app_state = app_state
        self._app_state.dataset_added.connect(self._on_dataset_added)
        self._app_state.dataset_removed.connect(self._on_dataset_removed)

        self._interactive_callback = interactive_callback
        self._clear_interactive_callback = clear_interactive_callback
        self._colormap_choice: Tuple[str, LinearSegmentedColormap] = DEFAULT_COLOR
        self._x_min = 0
        self._x_max = 100
        self._y_min = 0
        self._y_max = 100
        self.n = 0

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
        self._btn_change_cmap = QPushButton(self.tr("Color Map"))
        self._btn_change_axes = QPushButton(self.tr("Axes Limits"))
        ctrl_layout.addWidget(self._count_label)
        ctrl_layout.addStretch(1)
        ctrl_layout.addWidget(self._btn_clear)
        ctrl_layout.addWidget(self._btn_change_cmap)
        ctrl_layout.addWidget(self._btn_change_axes)

        self._btn_change_cmap.clicked.connect(
            lambda checked=True: self._colormap_chooser()
        )

        self._btn_change_axes.clicked.connect(
            lambda checked=True: QMessageBox.information(self,
                self.tr("Information"),
                self.tr("No plot to change axes limits of")
            )
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addWidget(self._navi_toolbar)
        layout.addWidget(ctrl)          # <-- add control strip just under the toolbar
        layout.addWidget(canvas)
        self._ui.wdgt_plot.setLayout(layout)

        self._btn_clear.clicked.connect(self._clear_selection_overlay)

        self._ui.btn_create_plot.clicked.connect(
            lambda checked: self._create_scatter_plot()
        )

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
        cbox_x_dataset.currentIndexChanged.connect(
            lambda checked: self._on_cbox_dataset_changed()
        )

        cbox_y_dataset.clear()
        cbox_y_dataset.addItem(self.tr("(no data)"), -1)
        for dataset in datasets:
            cbox_y_dataset.addItem(dataset.get_name(), dataset.get_id())
        cbox_y_dataset.currentIndexChanged.connect(
            lambda checked: self._on_cbox_dataset_changed()
        )

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
                band_descriptions = list((f"Band {descriptions.index(descr)}: " + descr, descriptions.index(descr)) \
                                            for descr in descriptions)
            else:
                band_descriptions = list((f"Band {i}", i) \
                                            for i in range(len(descriptions)))
            cbox_x_band.clear()      
            for descr, index in band_descriptions:
                cbox_x_band.addItem(self.tr(f'{descr}'), index)
            spin_box_x_band.setRange(0, len(band_descriptions) - 1)
        else:
            cbox_x_band.clear()
        
        y_dataset_id = self._ui.cbox_y_dataset.currentData()
        if self._app_state.has_dataset(y_dataset_id):
            dataset = self._app_state.get_dataset(y_dataset_id)
            bands = dataset.band_list()
            descriptions = list([band["description"] for band in bands])
            if descriptions[0]:
                band_descriptions = list((f"Band {descriptions.index(descr)}: " + descr, descriptions.index(descr)) \
                                            for descr in descriptions)
            else:
                band_descriptions = list((f"Band {i}", i) \
                                            for i in range(len(descriptions)))
            cbox_y_band.clear()
            for descr, index in band_descriptions:
                cbox_y_band.addItem(self.tr(f'{descr}'), index)
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

    def _check_dataset_compatibility(self) -> Optional[List[str]]:
        '''
        Checks to make sure the datasets are compatible and returns a list of errors if they are not.
        If no list of errors is returned, the datasets are compatible.
        '''
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
            render_dataset_dims = (render_dataset.get_width(), render_dataset.get_height())
        if not errors and x_dataset_dims != render_dataset_dims and y_dataset_dims != render_dataset_dims:
            errors.append("X dataset, Y dataset, and render dataset must have the same dimensions\n" +
                          f"X dataset dimensions: {x_dataset_dims}\nY dataset dimensions: {y_dataset_dims}\nRender dataset dimensions: {render_dataset_dims}")
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

    def get_x_dataset(self) -> Optional['RasterDataSet']:
        idx = self._ui.cbox_x_dataset.currentData()
        if not self._app_state.has_dataset(idx):
            return None
        return self._app_state.get_dataset(idx)
    
    def get_y_dataset(self) -> Optional['RasterDataSet']:
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
            lambda checked=True: self._colormap_img_changed(
                cbox_cmap, img_cmap
            )
        )

        if dialog_cmap_chooser.exec() == QDialog.Accepted:
            self._save_colormap_choice(cbox_cmap)
            self._create_scatter_plot()

    def _default_axes(
        self,
        x_min,
        x_max,
        y_min,
        y_max,
        default_x_min,
        default_x_max,
        default_y_min,
        default_y_max,
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

        x_min.setValue(default_x_min)
        x_max.setValue(default_x_max)
        y_min.setValue(default_y_min)
        y_max.setValue(default_y_max)

    def _axes_chooser(
        self,
        default_x_min,
        default_x_max,
        default_y_min,
        default_y_max
    ):
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

        x_min.setValue(default_x_min)
        x_max.setValue(default_x_max)
        y_min.setValue(default_y_min)
        y_max.setValue(default_y_max)

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
            if (self._x_min >= self._x_max) or (self._y_min >= self._y_max):
                QMessageBox.warning(self,
                    self.tr("Error"),
                    self.tr(f"Minimum must be less than the maximum\nX min: {x_min}\tX max: {x_max}\nY min: {y_min}\tY max: {y_max}")
                )

            else:
                self._x_min = x_min.value()
                self._x_max = x_max.value()
                self._y_min = y_min.value()
                self._y_max = y_max.value()
                self._create_scatter_plot()

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
            QMessageBox.warning(self,
                self.tr("Error"),
                self.tr("\n\n".join(errors))
            )
            return
        x_dataset = self.get_x_dataset()
        y_dataset = self.get_y_dataset()
        if x_dataset is None or y_dataset is None:
            self._clear_scatter_density_plot()
            return
        x_band = self.get_x_band()
        y_band = self.get_y_band()
        if x_band is None or y_band is None:
            self._clear_scatter_density_plot()
            return
    
        cols1 = x_dataset.get_width()
        rows1 = x_dataset.get_height()
        cols2 = y_dataset.get_width()
        rows2 = y_dataset.get_height()

        x = x_band.reshape(rows1 * cols1)
        y = y_band.reshape(rows2 * cols2)

        # Safe mins/maxes for axes panel defaults
        new_x = [n for n in x if np.isnan(n) == False]
        new_y = [m for m in y if np.isnan(m) == False]
        default_x_min = min(new_x); default_x_max = max(new_x)
        default_y_min = min(new_y); default_y_max = max(new_y)


        self._btn_change_axes.clicked.disconnect()
        self._btn_change_axes.clicked.connect(
            lambda checked=True: self._axes_chooser(
                default_x_min, default_x_max, default_y_min, default_y_max
            )
        )

        x_band_idx = self._ui.cbox_x_band.currentData()
        y_band_idx = self._ui.cbox_y_band.currentData()
        # --- draw density plot and keep axes ---
        # Create the display string for the x and y bands
        x_wvl = x_dataset.get_band_info()[x_band_idx].get('description', None)
        x_wvl_str = f': {x_wvl}' if x_wvl else ''
        x_band_description = f'{x_dataset.get_name()}\nBand {x_band_idx}' + x_wvl_str

        y_wvl = y_dataset.get_band_info()[y_band_idx].get('description', None)
        y_wvl_str = f': {y_wvl}' if y_wvl else ''
        y_band_description = f'{y_dataset.get_name()}\nBand {y_band_idx}' + y_wvl_str
        ax = self._using_mpl_scatter_density(
            self._figure, x, y, x_band_description, y_band_description,
            self._x_min, self._x_max, self._y_min,
            self._y_max, self._colormap_choice
        )
        self._ax = ax
        self._canvas.draw_idle()

        self._rows, self._cols = rows1, cols1
        self._x_flat = x
        self._y_flat = y
        self._xy = np.column_stack((self._x_flat, self._y_flat))
        self._valid_mask = np.isfinite(self._xy).all(axis=1)

        self._selector = PolygonSelector(
            self._ax,
            self._on_polygon_select,  # defined below
            useblit=True,
        )

        self._btn_clear.clicked.connect(self._clear_selection_overlay)

    def _using_mpl_scatter_density(
        self,
        fig: Figure,
        x:np.ndarray,
        y:np.ndarray,
        b1_description: str,
        b2_description: str,
        x_min: int, x_max:
        int, y_min: int,
        y_max: int,
        colormap: Tuple[str, LinearSegmentedColormap]
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
        x_min: int
            Smallest value of all x values
        x_max: int
            Largest value of all x values
        y_min: int
            Smallest value of all y values
        y_max: int
            Largest value of all y values
        colormap: matplotlib.colors.LinearSegmentedColormap or str
            Colormap for the density slice on the scatter plot. Default is DEFAULT_COLOR
        """
        new_x = [n for n in x if np.isnan(n) == False]
        new_y = [m for m in y if np.isnan(m) == False]

        if self.n <= 1:
            x_min = min(new_x); x_max = max(new_x)
            y_min = min(new_y); y_max = max(new_y)
            self._x_min, self._x_max = x_min, x_max
            self._y_min, self._y_max = y_min, y_max

        fig.clear()
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        ax.set_xlabel(b1_description)
        ax.set_ylabel(b2_description)
        ax.set_title("2D scatter plot of two bands with colormap")
        density = ax.scatter_density(x, y, cmap=colormap[1])
        x_range = max(x_max-x_min, 0)
        y_range = max(y_max-y_min, 0)
        ax.set_xlim(x_min-x_range/10, x_max+x_range/10)
        ax.set_ylim(y_min-y_range/10, y_max+y_range/10)
        fig.colorbar(density, label="Number of points per spectral value")

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

            # small hollow markers; lightweight
            self._sel_artist = self._ax.plot(
                self._x_flat[idx_to_show],
                self._y_flat[idx_to_show],
                marker='o', linestyle='None', markersize=2, alpha=0.7,
                markerfacecolor='none', markeredgecolor='red'
            )[0]

        if self._canvas is not None:
            self._canvas.draw_idle()

        render_ds_id = self._ui.cbox_render_ds.currentData()
        self._interactive_callback(self.get_selected_points(), render_ds_id)

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

        path, _ = QFileDialog.getSaveFileName(
            self._dlg, "Save selected points", "", "CSV Files (*.csv)"
        )
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
            return (np.array([], dtype=int),)*2 + (np.array([]),)*2
        rows, cols = np.unravel_index(self._selected_idx, (self._rows, self._cols))
        return rows, cols, self._x_flat[self._selected_idx], self._y_flat[self._selected_idx]
    
    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key_Escape:
            self._clear_selection_overlay() 
            e.accept()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        super().closeEvent(e)
        self._clear_interactive_callback()
