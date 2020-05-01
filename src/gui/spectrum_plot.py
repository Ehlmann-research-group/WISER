from enum import Enum
import os, random

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .spectrum_plot_config import SpectrumPlotConfigDialog
from .util import add_toolbar_action

from raster.spectra import SpectrumType, SpectrumAverageMode, calc_rect_spectrum

import matplotlib
matplotlib.use('Qt5Agg')
# TODO(donnie):  Seems to generate errors:
# matplotlib.rcParams['backend.qt5'] = 'PySide2'

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvas


def get_matplotlib_colors():
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
        rgba = matplotlib.colors.to_rgba_array(name)
        prod = np.prod(rgba)
        if prod >= 0.6:
            continue

        # print(f'Color {name} = {rgba} (type is {type(rgba)})')
        colors.append(name)

    return colors


def get_random_matplotlib_color(exclude_colors=[]):
    all_names = get_matplotlib_colors()
    while True:
        name = random.choice(all_names)
        if name not in exclude_colors:
            return name



class SpectrumPlotInfo:
    def __init__(self, dataset, plot_type, **kwargs):
        if plot_type not in SpectrumType:
            raise ValueError(f'Unrecognized plot_type {plot_type}')

        self._dataset = dataset
        self._plot_type = plot_type

        if plot_type in [SpectrumType.SINGLE_PIXEL, SpectrumType.AREA_AVERAGE]:
            self._point = kwargs['point']

        if plot_type == SpectrumType.AREA_AVERAGE:
            self._area = kwargs['area']
            self._avg_mode = kwargs['avg_mode']

            if self._area[0] % 2 != 1 or self._area[1] % 2 != 1:
                raise ValueError(f'area values must be odd; got {self._area}')

        self._description = kwargs.get('description')
        if self._description is None:
            self._description = self._generate_description()

        self._color = kwargs.get('color', get_random_matplotlib_color())

        # The calculated spectrum data.
        self._spectrum = None
        self._calculate_spectrum()

    def _generate_description(self):
        '''
        Generate and return a description of this spectrum plot, based on where
        it's from.
        '''
        if self._plot_type == SpectrumType.SINGLE_PIXEL:
            return f'Spectrum at ({self._point.x()}, {self._point.y()})'

        elif self._plot_type == SpectrumType.AREA_AVERAGE:
            values = {
                SpectrumAverageMode.MEAN : 'Mean',
                SpectrumAverageMode.MEDIAN : 'Median',
            }

            return f'{values[self._avg_mode]} of {self._area[0]}x{self._area[1]} ' + \
                   f'area around ({self._point.x()}, {self._point.y()})'

        else:
            return 'TODO:  Unknown spectrum type'

    def _calculate_spectrum(self):
        if self._plot_type == SpectrumType.SINGLE_PIXEL:
            p = self._point
            self._spectrum = self._dataset.get_all_bands_at(p.x(), p.y())

        elif self._plot_type == SpectrumType.AREA_AVERAGE:
            p = self._point
            (width, height) = self._area
            rect = QRect(p.x() - width / 2, p.y() - height / 2, width, height)
            self._spectrum = calc_rect_spectrum(self._dataset, rect,
                                                mode=self._avg_mode)

        else:
            raise ValueError(f'Plot-type {self._plot_type} is currently unsupported')

    def get_description(self):
        return self._description

    def get_dataset(self):
        return self._dataset

    def get_spectrum(self):
        return self._spectrum

    def get_color(self):
        return self._color


class SpectrumPlot(QWidget):
    '''
    This widget provides a spectrum-plot window in the user interface.
    '''

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)

        # Initialize widget's internal state

        self._app_state = app_state

        self._default_click_spectrum_type = SpectrumType.SINGLE_PIXEL
        self._default_area_avg_x = 3
        self._default_area_avg_y = 3
        self._default_average_mode = SpectrumAverageMode.MEAN

        self._active_spectrum = None
        self._active_spectrum_color = None

        self._collected_spectra = []

        # Initialize contents of the widget

        self._init_ui()


    def _init_ui(self):

        #==================================================
        # TOOLBAR

        self._toolbar = QToolBar(self.tr('Spectrum Toolbar'), parent=self)
        self._toolbar.setIconSize(QSize(20, 20))

        self._act_collect_spectrum = add_toolbar_action(self._toolbar,
            'resources/collect-spectrum.svg', self.tr('Collect spectrum'), self)
        self._act_collect_spectrum.triggered.connect(self._on_collect_spectrum)

        self._act_load_spectra = add_toolbar_action(self._toolbar,
            'resources/load-spectra.svg', self.tr('Load spectral library'), self)
        self._act_load_spectra.triggered.connect(self._on_load_spectra)

        self._act_configure = add_toolbar_action(self._toolbar,
            'resources/configure.svg', self.tr('Configure'), self)
        self._act_configure.triggered.connect(self._on_configure)

        # TODO(donnie):  Get rid of this eventually.  Similar functionality will
        #     be exposed in the spectral library tree.
        # self._act_clear_all_plots = add_toolbar_action(self._toolbar,
        #     'resources/clear-all-plots.svg', self.tr('Clear all plots'), self)
        # self._act_clear_all_plots.triggered.connect(self._on_clear_all_plots)

        #==================================================
        # Set up Matplotlib

        self._figure, self._axes = plt.subplots(tight_layout=True)

        self._axes.tick_params(direction='in', labelsize=4, pad=2, width=0.5,
            bottom=True, left=True, top=False, right=False)

        self._figure_canvas = FigureCanvas(self._figure)

        self._font_props = matplotlib.font_manager.FontProperties(size=4)

        # self.axes.set_autoscalex_on(True)
        # self.axes.set_autoscaley_on(False)
        # self.axes.set_ylim((0, 1))

        #==================================================
        # Widget for managing spectral library

        self._spectra_tree = QTreeWidget()
        self._spectra_tree.setColumnCount(1)
        self._spectra_tree.setHeaderLabels([self.tr('Spectra')])

        self._treeitem_active = QTreeWidgetItem()
        self._spectra_tree.addTopLevelItem(self._treeitem_active)
        self._treeitem_active.setHidden(True)

        self._treeitem_collected = QTreeWidgetItem([self.tr('Collected Spectra (unsaved)')])
        self._spectra_tree.addTopLevelItem(self._treeitem_collected)
        self._treeitem_collected.setHidden(True)

        # Events from the spectral library widget

        self._spectra_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._spectra_tree.customContextMenuRequested.connect(self._on_tree_context_menu)


        #==================================================
        # Widget layout

        '''
        layout = QGridLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        layout.setMenuBar(self._toolbar)

        layout.addWidget(self._figure_canvas, 0, 0)
        layout.addWidget(self._spectra_tree, 1, 0)
        '''

        splitter = QSplitter(Qt.Vertical, parent=self)
        splitter.addWidget(self._figure_canvas)
        splitter.addWidget(self._spectra_tree)

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        layout.setMenuBar(self._toolbar)
        layout.addWidget(splitter)
        self.setLayout(layout)


    def sizeHint(self):
        ''' The default size of the spectrum-plot widget is 400x200. '''
        return QSize(400, 200)


    def clear_all_plots(self):
        self._spectra.clear()
        self._draw_spectra()


    def _on_configure(self):
        '''
        This event-handler gets called when the user invokes the spectrum
        configuration dialog.
        '''
        cfg_dialog = SpectrumPlotConfigDialog(self,
            default_click_spectrum_type=self._default_click_spectrum_type,
            default_avg_mode=self._default_average_mode,
            default_area_avg_x=self._default_area_avg_x,
            default_area_avg_y=self._default_area_avg_y)

        result = cfg_dialog.exec_()

        if result == QDialog.Accepted:
            self._default_click_spectrum_type = \
                cfg_dialog.get_default_click_spectrum_type()

            self._default_average_mode = cfg_dialog.get_default_avg_mode()

            self._default_area_avg_x = cfg_dialog.get_default_area_avg_x()
            self._default_area_avg_y = cfg_dialog.get_default_area_avg_y()


    def _on_load_spectra(self):
        # TODO(donnie):  This should probably be on the main application.
        #     It can live here for now, but it will need to be migrated
        #     elsewhere in the future.

        # These are all file formats that will appear in the file-open dialog
        supported_formats = [
            self.tr('ENVI spectral libraries (*.hdr *.sli)'),
            self.tr('CSV files (*.csv)'),
            self.tr('All files (*)'),
        ]

        # TODO(donnie):  Get current directory from application state
        selected = QFileDialog.getOpenFileName(self,
            self.tr("Open Spectal Library File"),
            os.getcwd(), ';;'.join(supported_formats))

        if len(selected[0]) > 0:
            try:
                self.open_file(selected[0])
            except:
                mbox = QMessageBox(QMessageBox.Critical,
                    self.tr('Could not open file'), QMessageBox.Ok, self)

                mbox.setText(self.tr('The file could not be opened.'))
                mbox.setInformativeText(file_path)

                # TODO(donnie):  Add exception-trace info here, using
                #     mbox.setDetailedText()

                mbox.exec()


    def _on_collect_spectrum(self):
        if self._active_spectrum is None:
            # The "collect spectrum" button shouldn't be enabled if there is no
            # active spectrum!
            print('TODO:  shouldn\'t be able to collect spectrum when no active spectrum!')
            return

        self._collected_spectra.append(self._active_spectrum)
        self._active_spectrum_color = None

        treeitem_collected = QTreeWidgetItem([self._active_spectrum.get_description()])

        self._treeitem_collected.setHidden(False)
        self._treeitem_collected.setExpanded(True)
        self._treeitem_collected.addChild(treeitem_collected)

        self.clear_active_spectrum()

    def _on_tree_context_menu(self, pos):
        # Figure out which tree item was clicked on.
        treeitem = self._spectra_tree.itemAt(pos)
        if treeitem is None:
            return

        menu = QMenu(self)

        if treeitem is self._treeitem_active:
            # This is the Active spectrum
            act = menu.addAction(self.tr('Collect'))
            act = menu.addAction(self.tr('Edit...'))
            menu.addSeparator()
            act = menu.addAction(self.tr('Discard...'))

        elif treeitem is self._treeitem_collected:
            # This is the Collected Spectra group; these are unsaved spectra
            # that the user has collected.
            act = menu.addAction(self.tr('Save to file...'))
            menu.addSeparator()
            act = menu.addAction(self.tr('Discard all...'))

        elif treeitem.parent() is None:
            # This is a spectral library
            # act = menu.addAction(self.tr('Save edits...'))
            act = menu.addAction(self.tr('Unload library'))

        else:
            # This is a specific spectrum plot (other than the active spectrum)
            act = menu.addAction(self.tr('Edit...'))
            menu.addSeparator()
            act = menu.addAction(self.tr('Discard...'))

        global_pos = self._spectra_tree.mapToGlobal(pos)
        menu.exec_(global_pos)


    '''
    def _on_clear_all_plots(self):
        answer = QMessageBox.question(self, self.tr('Clear all plots?'),
            self.tr('About to delete {} plots, are you sure?'),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if answer == QMessageBox.Yes:
            self.clear_all_plots()
    '''

    def _draw_spectra(self):
        self._axes.clear()

        all_spectra = []

        if self._active_spectrum is not None:
            all_spectra.append(self._active_spectrum)

        all_spectra.extend(self._collected_spectra)

        # If all datasets have wavelength data, we will set the X-axis title to
        # indicate that these are wavelengths.
        use_wavelengths = True
        for info in all_spectra:
            if not info.get_dataset().has_wavelengths():
                use_wavelengths = False
                break

        if use_wavelengths:
            # If we can use wavelengths, each spectrum is a series of
            # (wavelength, value) coordinates, which we can plot.  This allows
            # the graphs to look correct, even in the face of bad bands, plots
            # from different datasets with different wavelengths, etc.

            self._axes.set_xlabel('Wavelength (nm)', labelpad=0, fontproperties=self._font_props)
            self._axes.set_ylabel('Value', labelpad=0, fontproperties=self._font_props)

            # Plot each spectrum against its corresponding wavelength values
            for info in all_spectra:
                dataset = info.get_dataset()
                spectrum = info.get_spectrum()
                wavelengths = [b['wavelength'].value for b in dataset.band_list()]
                self._axes.plot(wavelengths, spectrum, color=info.get_color(), linewidth=0.5)
        else:
            # If we don't have wavelengths, each spectrum is just a series of
            # values.  We can of course plot this, but we can't guarantee it
            # will be meaningful if there are multiple plots from different
            # datasets to display.

            self._axes.set_xlabel('Band Index', labelpad=0, fontproperties=self._font_props)
            self._axes.set_ylabel('Value', labelpad=0, fontproperties=self._font_props)

            for info in all_spectra:
                dataset = info.get_dataset()
                spectrum = info.get_spectrum()
                self._axes.plot(spectrum, linewidth=0.5)

        self._figure_canvas.draw()


    def set_active_spectrum(self, dataset, coord):
        if self._active_spectrum_color is None:
            self._active_spectrum_color = get_random_matplotlib_color()

        if self._default_click_spectrum_type == SpectrumType.SINGLE_PIXEL:
            info = SpectrumPlotInfo(dataset, SpectrumType.SINGLE_PIXEL,
                point=coord, color=self._active_spectrum_color)

        elif self._default_click_spectrum_type == SpectrumType.AREA_AVERAGE:

            info = SpectrumPlotInfo(dataset, SpectrumType.AREA_AVERAGE,
                point=coord, color=self._active_spectrum_color,
                area=(self._default_area_avg_x, self._default_area_avg_y),
                avg_mode=self._default_average_mode)

        else:
            raise ValueError(f'Unrecognized value for default click spectrum type:  {self._default_click_spectrum_type}')

        self._active_spectrum = info

        self._treeitem_active.setText(0, info.get_description())
        self._treeitem_active.setHidden(False)

        self._act_collect_spectrum.setEnabled(True)
        self._draw_spectra()


    def clear_active_spectrum(self):
        self._active_spectrum = None
        self._treeitem_active.setHidden(True)

        self._act_collect_spectrum.setEnabled(False)
        self._draw_spectra()
