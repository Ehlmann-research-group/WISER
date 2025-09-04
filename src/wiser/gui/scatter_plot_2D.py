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

from typing import Callable

from matplotlib.pyplot import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from wiser import plugins, raster
from .generated.scatter_plot_2D_ui import Ui_ScatterPlotDialog
from .generated.scatter_plot_band_chooser_ui import Ui_ScatterPlotBandChooser
from .generated.scatter_plot_error_ui import Ui_ScatterPlotError
from .generated.scatter_plot_axes_ui import Ui_ScatterPlotAxes
from .generated.scatter_plot_colormap_ui import Ui_ScatterPlotColormap

from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *

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


class ScatterPlot2DDialog(QDialog):
    """
    A Class to represents the density sliced 2D scatter plot of 2 bands

    Parameters
    ----------
    None

    Attributes
    ----------
    colormap_choice: matplotlib.colors.LinearSegmentedColormap or str
        Colormap for the density slice on the scatter plot. Default is WHITE_VIRDIS
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

        self._app_state = app_state

        self._interactive_callback = interactive_callback
        self._clear_interactive_callback = clear_interactive_callback
        logging.info("2D scatter plot")
        self.colormap_choice = WHITE_VIRDIS
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 100
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
        if context_type == plugins.ContextMenuType.RASTER_VIEW:
            self.n = 0
            act1 = context_menu.addAction(context_menu.tr("2D scatter plot"))
            act1.triggered.connect(
                lambda checked=False: self.band_chooser()
            )

    def error_box_dimensions(
        self,
        message,
        context,
    ):
        """Displays desired error message and goes back to dimensions chooser GUI when finished

        Parameters
        ----------
        message: str
            Error message to be displayed in the widget
        context: dict
            Available WISER classes
        """
        dialog = QDialog(self)
        ui = Ui_ScatterPlotError()
        ui.setupUi(dialog)
        error_message = ui.error_message
        error_message.setText(message)

        if dialog.exec() == QDialog.Accepted:
            self.band_chooser()

    def error_box_axes(
        self,
        message,
        context,
        b1,
        b2,
        i1,
        i2,
        default_x_min,
        default_x_max,
        default_y_min,
        default_y_max,
    ):
        """Displays desired error message and goes back to axes limits chooser GUI when finished

        Parameters
        ----------
        message: str
            Error message to be displayed in the widget
        chosen_datasets: list
            List of user chosen datasets that are to be layer stacked
        context: dict
            Available WISER classes
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
        """
        dialog = QDialog(self)
        ui = Ui_ScatterPlotError()
        ui.setupUi(dialog)
        error_message = ui.error_message
        error_message.setText(message)

        if dialog.exec() == QDialog.Accepted:
            self.axes_chooser(
                b1,
                b2,
                i1,
                i2,
                default_x_min,
                default_x_max,
                default_y_min,
                default_y_max,
                context,
            )

    def default_axes(
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

    def axes_chooser(
        self,
        b1,
        b2,
        i1,
        i2,
        default_x_min,
        default_x_max,
        default_y_min,
        default_y_max,
        context,
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
            lambda checked=True: self.default_axes(
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
            self.x_min = x_min.value()
            self.x_max = x_max.value()
            self.y_min = y_min.value()
            self.y_max = y_max.value()

            if (self.x_min >= self.x_max) or (self.y_min >= self.y_max):
                self.error_box_axes(
                    "Minimum must be less than the maximum",
                    context,
                    b1,
                    b2,
                    i1,
                    i2,
                    default_x_min,
                    default_x_max,
                    default_y_min,
                    default_y_max,
                )

            else:
                self.scatter_plot(
                    b1,
                    b2,
                    i1,
                    i2,
                    context,
                    self.x_min,
                    self.x_max,
                    self.y_min,
                    self.y_max,
                    self.colormap_choice,
                )

    def colormap_changed(self, default, colormap_box, colormap_img):
        """Sets colormap_choice to the user chosen colormap

        Parameters
        ----------
        default: PySide2.QtWidgets.QAbstractButton.QCheckBox
            sets colotmap_choice to the default option, WHITE_VIRDIS, when checked
        colormapbox: PySide2.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available colormap names
        colormap_img: PySide2.QtWidgets.QLabel
            Label that contains an image of the chosen colormap within the QComboBox
        """

        if default.isChecked():
            self.colormap_choice = WHITE_VIRDIS
        else:
            self.colormap_choice = colormap_box.currentText()
            self.colormap_images(colormap_box, colormap_img)

    def colormap_images(self, colormap_box, colormap_img):
        """Code provided by Donnie Pinkston
        Changes the colormap GUI to indicate which colormap the user has chosen

        Parameters
        ----------
        colormapbox: PySide2.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available colormap names
        colormap_img: PySide2.QtWidgets.QLabel
            Label that contains an image of the chosen colormap within the QComboBox
        """

        cmap = cm.get_cmap(colormap_box.currentText(), 256)
        img = QImage(cmap.N, 24, QImage.Format_RGB32)
        for x in range(cmap.N):
            rgba = cmap(x, bytes=True)
            c = QColor(rgba[0], rgba[1], rgba[2])
            for y in range(img.height()):
                img.setPixelColor(x, y, c)

        colormap_img.setPixmap(QPixmap.fromImage(img))

    def colormap_chooser(self, b1, b2, i1, i2, context):
        """Displays GUI that allows user to choose the colormap for the density slice

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
        context: dict
            Available WISER classes
        """
        dialog = QDialog(self)
        ui = Ui_ScatterPlotColormap()
        ui.setupUi(dialog)
        colormap_img = ui.colormap_img
        colormap_box = ui.colormap_box
        default = ui.default_colormap

        for cmap in plt.colormaps():
            colormap_box.addItem(cmap)

        self.colormap_images(colormap_box, colormap_img)
        self.colormap_choice = colormap_box.currentText()

        colormap_box.currentTextChanged.connect(
            lambda checked=True: self.colormap_changed(
                default, colormap_box, colormap_img
            )
        )

        default.stateChanged.connect(
            lambda checked=True: self.colormap_changed(
                default, colormap_box, colormap_img
            )
        )

        if dialog.exec() == QDialog.Accepted:
            self.scatter_plot(
                b1,
                b2,
                i1,
                i2,
                context,
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
                self.colormap_choice,
            )

    def image_changed(self, datasets, image, combo, spin):
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

    def check_dimensions(self, image, datasets):
        """Compares the dimensions of the two chosen datasets

        Parameters
        ----------
        image: PySide2.QtWidgets.QComboBox
            Combo box that displayes a drop down menu of all available datasets
        datasets: list
            List of all available datasets in WISER

        Returns
        ----------
        Returns t=True if the dimensions of the two images match
        Returns False otherwise
        """

        idx = image.currentIndex()
        image_data = datasets[idx]
        image_data = image_data.get_shape()
        rows = image_data[-1]
        cols = image_data[-2]
        return rows, cols

    def band_chooser(self):
        """
        Displays GUI that allows user to choose which 2 bands from which image/images to plot
        """
        dialog = QDialog(self)
        ui = Ui_ScatterPlotBandChooser()
        ui.setupUi(dialog)
        band1 = ui.band1
        band2 = ui.band2
        band1_box = ui.band1_box
        band2_box = ui.band2_box
        image1chooser = ui.image1
        image2chooser = ui.image2

        datasets = self._app_state.get_datasets()
        all_datasets = []
        for data in datasets:
            all_datasets.append(data.get_name())
        image1chooser.addItems(all_datasets)
        image2chooser.addItems(all_datasets)

        bands = datasets[0].band_list()
        bands = list([i["description"] for i in bands])
        bands = list(f"Band {bands.index(i)}: " + i for i in bands)

        band1.addItems(bands)
        band2.addItems(bands)
        band1_box.setRange(0, len(bands) - 1)
        band2_box.setRange(0, len(bands) - 1)

        image1chooser.currentIndexChanged.connect(
            lambda checked=True: self.image_changed(
                datasets, image1chooser, band1, band1_box
            )
        )
        image2chooser.currentIndexChanged.connect(
            lambda checked=True: self.image_changed(
                datasets, image2chooser, band2, band2_box
            )
        )
        band1.currentIndexChanged.connect(
            lambda checked=True: self.combo_box_changed(band1, band1_box)
        )
        band1_box.valueChanged.connect(
            lambda checked=True: self.spin_box_changed(band1, band1_box)
        )
        band2.currentIndexChanged.connect(
            lambda checked=True: self.combo_box_changed(band2, band2_box)
        )
        band2_box.valueChanged.connect(
            lambda checked=True: self.spin_box_changed(band2, band2_box)
        )

        if dialog.exec() == QDialog.Accepted:
            rows1, cols1 = self.check_dimensions(image1chooser, datasets)
            rows2, cols2 = self.check_dimensions(image2chooser, datasets)

            if (rows2 != rows1) or (cols2 != cols1):
                self.error_box_dimensions(
                    "All datasets must have the same spatial dimensions!", self._app_state
                )
            else:
                band1 = band1.currentIndex()
                band2 = band2.currentIndex()
                image1 = image1chooser.currentIndex()
                image2 = image2chooser.currentIndex()

                self.scatter_plot(
                    band1,
                    band2,
                    image1,
                    image2,
                    self._app_state,
                    self.x_min,
                    self.x_max,
                    self.y_min,
                    self.y_max,
                )

    def using_mpl_scatter_density(
        self, fig, x, y, b1, b2, x_min, x_max, y_min, y_max, colormap
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
        b1: int
            Band number for user chosen band on the x-axis
        b2: int
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
            Colormap for the density slice on the scatter plot. Default is WHITE_VIRDIS
        """
        new_x = [n for n in x if np.isnan(n) == False]
        new_y = [m for m in y if np.isnan(m) == False]  # <-- bugfix: was filtering x

        if self.n <= 1:
            x_min = min(new_x); x_max = max(new_x)
            y_min = min(new_y); y_max = max(new_y)
            self.x_min, self.x_max = x_min, x_max
            self.y_min, self.y_max = y_min, y_max

        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        ax.set_xlabel(f"band {b1}")
        ax.set_ylabel(f"band {b2}")
        ax.set_title("2D scatter plot of two bands with colormap")
        density = ax.scatter_density(x, y, cmap=colormap)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        fig.colorbar(density, label="Number of points per spectral value")

        return ax  # <-- return the axes for the selector

    def scatter_plot(
        self,
        band1,
        band2,
        image1,
        image2,
        app_state,
        x_min,
        x_max,
        y_min,
        y_max,
        color=WHITE_VIRDIS,
    ):
        """Displays the widget that holds the density sliced 2D scatter plot

        Parameters
        ----------
        band1: int
            Band number for user chosen band on the x-axis
        band2: int
            Band number for user chosen band on the y-axis
        image1: int
            Index of user chosen image in all images uploaded on WISER
        image2: int
            Index of user chosen image in all images uploaded on WISER
        app_state: dict
            Available WISER classes
        x_min: int
            Minimum x-axis limit
        x_max: int
            Maximum x-axis limit
        y_min: int
            Minimum y-axis limit
        y_max: int
            Maximum y-axis limit
        color: matplotlib.colors.LinearSegmentedColormap or str
            Colormap for the density slice on the scatter plot. Default is WHITE_VIRDIS
        """
        self.n += 1
        dialog = QDialog(self)
        ui = Ui_ScatterPlotDialog()
        ui.setupUi(dialog)
        plot_widget = ui.plot_widget
        colormap_button = ui.colormap
        axes_button = ui.limits
        dialog.setModal(False)

        colormap_button.clicked.connect(
            lambda checked=True: self.colormap_chooser(band1, band2, image1, image2, self._app_state)
        )

        figure = Figure()
        canvas = FigureCanvas(figure)
        self._canvas = canvas  # <-- keep a handle for redraws
        navi_toolbar = NavigationToolbar(canvas, None)

        # --- NEW: small control strip (count + Save + Clear) ---
        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        self._count_label = QLabel("0 pts")
        btn_save = QPushButton("Save selection…")
        btn_clear = QPushButton("Clear selection")
        ctrl_layout.addWidget(self._count_label)
        ctrl_layout.addStretch(1)
        ctrl_layout.addWidget(btn_clear)
        ctrl_layout.addWidget(btn_save)

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addWidget(navi_toolbar)
        layout.addWidget(ctrl)          # <-- add control strip just under the toolbar
        layout.addWidget(canvas)
        plot_widget.setLayout(layout)

        datasets = self._app_state.get_datasets()
        rows1 = datasets[image1].get_shape()[-1]
        cols1 = datasets[image1].get_shape()[-2]
        rows2 = datasets[image2].get_shape()[-1]
        cols2 = datasets[image2].get_shape()[-2]

        x = datasets[image1].get_band_data(band1).reshape(rows1 * cols1)
        y = datasets[image2].get_band_data(band2).reshape(rows2 * cols2)

        # Safe mins/maxes for axes panel defaults
        new_x = [n for n in x if np.isnan(n) == False]
        new_y = [m for m in y if np.isnan(m) == False]
        default_x_min = min(new_x); default_x_max = max(new_x)
        default_y_min = min(new_y); default_y_max = max(new_y)

        axes_button.clicked.connect(
            lambda checked=True: self.axes_chooser(
                band1, band2, image1, image2,
                default_x_min, default_x_max, default_y_min, default_y_max,
                self._app_state,
            )
        )

        # --- draw density plot and keep axes ---
        ax = self.using_mpl_scatter_density(
            figure, x, y, band1, band2, x_min, x_max, y_min, y_max, color
        )
        self._ax = ax

        # --- selection backing arrays ---
        # Note: assumes dims matched earlier in the workflow
        self._rows, self._cols = rows1, cols1
        self._x_flat = x
        self._y_flat = y
        self._xy = np.column_stack((self._x_flat, self._y_flat))
        self._valid_mask = np.isfinite(self._xy).all(axis=1)

        # --- install polygon selector on the density axes ---
        self._selector = PolygonSelector(
            self._ax,
            self._on_polygon_select,  # defined below
            useblit=True,
        )

        # --- wire buttons ---
        btn_save.clicked.connect(self._save_selection)
        btn_clear.clicked.connect(self._clear_selection_overlay)

        dialog.show()

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

        self._interactive_callback(self.get_selected_points())

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

    def closeEvent(self, e):
        super().closeEvent(e)
        self._clear_interactive_callback()
