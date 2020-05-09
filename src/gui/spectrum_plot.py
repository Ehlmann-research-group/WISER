from enum import Enum
import os, random

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .spectrum_plot_config import SpectrumPlotConfigDialog
from .util import add_toolbar_action

from raster.envi_spectral_library import ENVISpectralLibrary
from raster.spectra import SpectrumType, SpectrumAverageMode, calc_rect_spectrum

import matplotlib
matplotlib.use('Qt5Agg')
# TODO(donnie):  Seems to generate errors:
# matplotlib.rcParams['backend.qt5'] = 'PySide2'

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvas


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
        if prod >= 0.4:
            continue

        # print(f'Color {name} = {rgba} (type is {type(rgba)})')
        colors.append(name)

    return colors


def get_random_matplotlib_color(exclude_colors=[]):
    '''
    Returns a random matplotlib color name from the available matplotlib colors
    returned by the get_matplotlib_colors().
    '''
    all_names = get_matplotlib_colors()
    while True:
        name = random.choice(all_names)
        if name not in exclude_colors:
            return name


def get_color_icon(color_name, width=16, height=16):
    '''
    Generate a QIcon of the specified color and optional size.  If the size is
    unspecified, a 24x24 icon is generated.

    *   color_name is a string color name recognized by matplotlib, which
        includes strings of the form "#RRGGBB" where R, G and B are hexadecimal
        digits.

    *   width is the icon's width in pixels, and defaults to 24.

    *   height is the icon's height in pixels, and defaults to 24.
    '''
    rgba = matplotlib.colors.to_rgba_array(color_name).flatten()

    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor.fromRgbF(rgba[0], rgba[1], rgba[2]))

    return QIcon(QPixmap.fromImage(img))


'''
class SpectrumTreeEntry(Enum):
    ACTIVE_SPECTRUM = 1
    COLLECTED_SPECTRA_GROUP = 2
    COLLECTED_SPECTRUM = 3
    SPECTRAL_LIBRARY_GROUP = 4
    LIBRARY_SPECTRUM = 5


class SpectrumTreeItemData:
    def __init__(self, item_type):
        self._plot_visible = False
        self._plot_color = None

    def is_plot_visible(self):
        return self._plot_visible

    def set_plot_visible(self, visible):
        self._plot_visible = visible

    def get_plot_color(self):
        return self._plot_color

    def set_plot_color(self, color):
        self._plot_color = color
'''

class SpectrumInfo:
    def __init__(self):
        self._color = None
        self._visible = False

    def get_name(self):
        pass

    def has_wavelengths(self):
        pass

    def get_wavelengths(self):
        pass

    def get_spectrum(self):
        pass

    def is_visible(self):
        return self._visible

    def set_visible(self, visible):
        self._visible = visible

        if visible and self._color is None:
            self.set_color(get_random_matplotlib_color())

    def get_color(self):
        return self._color

    def set_color(self, color):
        self._color = color


class CollectedSpectrum(SpectrumInfo):
    def __init__(self, dataset, plot_type, **kwargs):
        super().__init__()

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

        self._name = kwargs.get('name')
        if self._name is None:
            self._name = self._generate_name()

        self.set_color(kwargs.get('color', get_random_matplotlib_color()))

        # The calculated spectrum data.
        self._spectrum = None
        self._calculate_spectrum()

    def _generate_name(self):
        '''
        Generate and return a name for this spectrum plot, based on where
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

    def get_name(self):
        return self._name

    def get_dataset(self):
        return self._dataset

    def get_point(self):
        return self._point

    def get_plot_type(self):
        return self._plot_type

    def get_area(self):
        return self._area

    def get_avg_mode(self):
        return self._avg_mode

    def has_wavelengths(self):
        return self._dataset.has_wavelengths()

    def get_wavelengths(self):
        return [b['wavelength'].value for b in self._dataset.band_list()]

    def get_spectrum(self):
        return self._spectrum

class LibrarySpectrum(SpectrumInfo):
    def __init__(self, spectral_library, index):
        super().__init__()

        self._spectral_library = spectral_library
        self._spectrum_index = index

    def get_name(self):
        return self._spectral_library.get_spectrum_name(self._spectrum_index)

    def has_wavelengths(self):
        return self._spectral_library.has_wavelengths()

    def get_wavelengths(self):
        return [b['wavelength'].value for b in self._spectral_library.band_list()]

    def get_spectrum(self):
        return self._spectral_library.get_spectrum(self._spectrum_index)




class SpectrumPlot(QWidget):
    '''
    This widget provides a spectrum-plot window in the user interface.
    '''

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)

        # Initialize widget's internal state

        self._app_state = app_state

        # For user mouse clicks, these are the parameters for generating spectra

        self._default_click_spectrum_type = SpectrumType.SINGLE_PIXEL
        self._default_area_avg_x = 3
        self._default_area_avg_y = 3
        self._default_average_mode = SpectrumAverageMode.MEAN

        # State for the "active spectrum", i.e. spectra generated by user clicks

        self._active_spectrum = None
        self._active_spectrum_color = None

        # These are the in-memory-only collected spectra

        self._collected_spectra = []

        # These are the SpectrumInfo objects for all library spectra.  They are
        # stored in a dictionary with the spectral library as the key, so that
        # when a spectral library is unloaded, it's straightforward to remove
        # the corresponding spectra.

        self._library_spectra = {}

        # This is the currently selected treeview item.  Initially, no item is
        # selected, so we set this to None.
        self._selected_treeview_item = None

        # Initialize UI components of the widget

        self._init_ui()

        # Set up event handlers

        self._app_state.spectral_library_added.connect(self._on_spectral_library_added)
        self._app_state.spectral_library_removed.connect(self._on_spectral_library_removed)


    def _init_ui(self):

        #==================================================
        # TOOLBAR

        self._toolbar = QToolBar(self.tr('Spectrum Toolbar'), parent=self)
        self._toolbar.setIconSize(QSize(20, 20))

        self._act_collect_spectrum = add_toolbar_action(self._toolbar,
            'resources/collect-spectrum.svg', self.tr('Collect spectrum'), self)
        self._act_collect_spectrum.triggered.connect(self._on_collect_spectrum)

        self._act_load_spectral_library = add_toolbar_action(self._toolbar,
            'resources/load-spectra.svg', self.tr('Load spectral library'), self)
        self._act_load_spectral_library.triggered.connect(self._on_load_spectral_library)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._toolbar.addWidget(spacer)

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
        self._spectra_tree.setHeaderLabels([self.tr('Spectra and Spectral Libraries')])

        self._treeitem_active = QTreeWidgetItem()
        self._spectra_tree.addTopLevelItem(self._treeitem_active)
        self._treeitem_active.setHidden(True)

        self._treeitem_collected = QTreeWidgetItem([self.tr('Collected Spectra (unsaved)')])
        self._spectra_tree.addTopLevelItem(self._treeitem_collected)
        self._treeitem_collected.setHidden(True)

        # Events from the spectral library widget

        self._spectra_tree.currentItemChanged.connect(self._on_tree_selection_changed)
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


    # def clear_all_plots(self):
    #     self._spectra.clear()
    #     self._draw_spectra()


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


    def _on_load_spectral_library(self):
        # TODO(donnie):  This should probably be on the main application.
        #     It can live here for now, but it will need to be migrated
        #     elsewhere in the future.

        # These are all file formats that will appear in the file-open dialog
        supported_formats = [
            self.tr('ENVI spectral libraries (*.hdr *.sli)'),
            self.tr('All files (*)'),
        ]

        # TODO(donnie):  Get current directory from application state
        selected = QFileDialog.getOpenFileName(self,
            self.tr("Open Spectal Library File"),
            os.getcwd(), ';;'.join(supported_formats))
        if len(selected[0]) > 0:
            try:
                # Load the spectral library into the application state
                self._app_state.open_file(selected[0])
            except:
                mbox = QMessageBox(QMessageBox.Critical,
                    self.tr('Could not open file'),
                    self.tr('The file could not be opened.'),
                    QMessageBox.Ok, parent=self)

                # TODO(donnie):  Add exception-trace info here, using
                # mbox.setInformativeText(file_path)
                # mbox.setDetailedText()

                mbox.exec()


    def _on_spectral_library_added(self, index):
        # TODO(donnie):  Put spectra / spectral library info onto each tree item
        #     so we can implement context menus properly.

        spectral_library = self._app_state.get_spectral_library(index)

        treeitem_library = QTreeWidgetItem([spectral_library.get_name()])
        treeitem_library.setData(0, Qt.UserRole, spectral_library)
        self._spectra_tree.addTopLevelItem(treeitem_library)

        info_list = []
        for i in range(spectral_library.num_spectra()):
            name = spectral_library.get_spectrum_name(i)

            info = LibrarySpectrum(spectral_library, i)  # UI state
            info_list.append(info)

            treeitem_spectrum = QTreeWidgetItem([name])
            treeitem_spectrum.setData(0, Qt.UserRole, info)

            treeitem_library.addChild(treeitem_spectrum)

        self._library_spectra[index] = info_list


    def _on_spectral_library_removed(self, index):
        # TODO(donnie):  Implement
        pass


    def _on_collect_spectrum(self):
        if self._active_spectrum is None:
            # The "collect spectrum" button shouldn't be enabled if there is no
            # active spectrum!
            print('TODO:  shouldn\'t be able to collect spectrum when no active spectrum!')
            return

        collected_spectrum = self._active_spectrum
        self._collected_spectra.append(collected_spectrum)
        self._active_spectrum_color = None

        treeitem_collected = QTreeWidgetItem([self._active_spectrum.get_name()])
        treeitem_collected.setData(0, Qt.UserRole, collected_spectrum)
        treeitem_collected.setIcon(0, get_color_icon(collected_spectrum.get_color()))

        self._treeitem_collected.setHidden(False)
        self._treeitem_collected.setExpanded(True)

        self._treeitem_collected.addChild(treeitem_collected)

        # If the active spectrum is the currently selected one, disable the
        # selection, since the active item is now hidden.
        if self._selected_treeview_item is self._treeitem_active:
            # self._treeitem_active.setSelected(False)
            self._spectra_tree.setCurrentItem(self._treeitem_collected, 0, QItemSelectionModel.SelectCurrent)
            self._selected_treeview_item = None

        self.clear_active_spectrum()


    def _on_tree_selection_changed(self, current, previous):
        # print(f'selection changed.  current.data = {current.data(0, Qt.UserRole)}')
        self._selected_treeview_item = current
        self._draw_spectra()


    def _on_tree_context_menu(self, pos):
        # Figure out which tree item was clicked on.
        treeitem = self._spectra_tree.itemAt(pos)
        if treeitem is None:
            return

        menu = QMenu(self)

        if treeitem is self._treeitem_active:
            # This is the Active spectrum.  It is always visible, so there will
            # be no show/hide option here.
            act = menu.addAction(self.tr('Collect'))
            act = menu.addAction(self.tr('Edit...'))
            menu.addSeparator()
            act = menu.addAction(self.tr('Discard...'))

        elif treeitem is self._treeitem_collected:
            # This is the Collected Spectra group; these are unsaved spectra
            # that the user has collected.

            act = menu.addAction(self.tr('Show all spectra'))
            act.setData(treeitem)
            act.triggered.connect(lambda checked, action=act : self._on_show_all_spectra(checked, action))

            act = menu.addAction(self.tr('Hide all spectra'))
            act.setData(treeitem)
            act.triggered.connect(lambda checked, action=act : self._on_hide_all_spectra(checked, action))

            act = menu.addAction(self.tr('Save to file...'))
            menu.addSeparator()
            act = menu.addAction(self.tr('Discard all...'))

        elif treeitem.parent() is None:
            # This is a spectral library
            # act = menu.addAction(self.tr('Save edits...'))

            act = menu.addAction(self.tr('Show all spectra'))
            act.setData(treeitem)
            act.triggered.connect(lambda checked, action=act : self._on_show_all_spectra(checked, action))

            act = menu.addAction(self.tr('Hide all spectra'))
            act.setData(treeitem)
            act.triggered.connect(lambda checked, action=act : self._on_hide_all_spectra(checked, action))

            menu.addSeparator()
            act = menu.addAction(self.tr('Unload library'))

        else:
            # This is a specific spectrum plot (other than the active spectrum),
            # either in the collected spectra, or in a spectral library.

            info = treeitem.data(0, Qt.UserRole)

            # TODO(donnie):  Show/hide option
            act = menu.addAction(self.tr('Show spectrum'))
            act.setCheckable(True)
            act.setChecked(info.is_visible())
            act.setData(info)
            act.triggered.connect(lambda checked, action=act : self._on_show_spectrum(checked, action))

            act = menu.addAction(self.tr('Edit...'))
            menu.addSeparator()
            act = menu.addAction(self.tr('Discard...'))

        global_pos = self._spectra_tree.mapToGlobal(pos)
        menu.exec_(global_pos)


    def _on_show_spectrum(self, checked, action):
        print(f'action = {action}')
        treeitem = action.data()

        info = treeitem.data(0, Qt.UserRole)
        info.set_visible(True)
        self._draw_spectra()


    def _on_show_all_spectra(self, checked, action):
        print(f'action = {action}')
        treeitem = action.data()

        if treeitem is self._treeitem_collected:
            for info in self._collected_spectra:
                info.set_visible(True)

        else:
            # The action is for a spectral library
            library_index = treeitem.data(0, Qt.UserRole)

            for info in self._library_spectra[library_index]:
                info.set_visible(True)

        self._draw_spectra()


    def _on_hide_all_spectra(self, checked, action):
        treeitem = action.data()

        if treeitem is self._treeitem_collected:
            for info in self._collected_spectra:
                info.set_visible(False)

        else:
            # The action is for a spectral library
            library_index = treeitem.data(0, Qt.UserRole)

            for info in self._library_spectra[library_index]:
                info.set_visible(False)

        self._draw_spectra()


    def _draw_spectra(self):
        self._axes.clear()

        # Build up a list of all spectra that could be displayed.

        all_spectra = []

        selected_spectrum = None
        if self._selected_treeview_item is not None:
            selected_spectrum = self._selected_treeview_item.data(0, Qt.UserRole)

        if self._active_spectrum is not None:
            all_spectra.append(self._active_spectrum)

        # Always include all collected spectra.
        all_spectra.extend(self._collected_spectra)

        # Include all library spectra.
        for info_list in self._library_spectra.values():
            all_spectra.extend(info_list)

        # Filter down spectra to only the ones that are displayed
        all_spectra = [s for s in all_spectra if s.is_visible()]

        # If all datasets have wavelength data, we will set the X-axis title to
        # indicate that these are wavelengths.
        use_wavelengths = True
        for info in all_spectra:
            if not info.has_wavelengths():
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
                linewidth = 0.5
                if info is selected_spectrum:
                    linewidth = 1

                wavelengths = info.get_wavelengths()
                spectrum = info.get_spectrum()
                self._axes.plot(wavelengths, spectrum, color=info.get_color(), linewidth=linewidth)
        else:
            # If we don't have wavelengths, each spectrum is just a series of
            # values.  We can of course plot this, but we can't guarantee it
            # will be meaningful if there are multiple plots from different
            # datasets to display.

            self._axes.set_xlabel('Band Index', labelpad=0, fontproperties=self._font_props)
            self._axes.set_ylabel('Value', labelpad=0, fontproperties=self._font_props)

            for info in all_spectra:
                linewidth = 0.5
                if info is selected_spectrum:
                    linewidth = 1

                dataset = info.get_dataset()
                spectrum = info.get_spectrum()
                self._axes.plot(spectrum, linewidth=linewidth)

        self._figure_canvas.draw()


    def set_active_spectrum(self, dataset, coord):
        '''
        Sets the current "active spectrum" in the spectral plot window, and
        updates the info in the spectrum tree.  If there is a previous "active
        spectrum," it is discarded.
        '''
        if self._active_spectrum_color is None:
            self._active_spectrum_color = get_random_matplotlib_color()

        if self._default_click_spectrum_type == SpectrumType.SINGLE_PIXEL:
            info = CollectedSpectrum(dataset, SpectrumType.SINGLE_PIXEL,
                point=coord, color=self._active_spectrum_color)

        elif self._default_click_spectrum_type == SpectrumType.AREA_AVERAGE:

            info = CollectedSpectrum(dataset, SpectrumType.AREA_AVERAGE,
                point=coord, color=self._active_spectrum_color,
                area=(self._default_area_avg_x, self._default_area_avg_y),
                avg_mode=self._default_average_mode)

        else:
            raise ValueError(f'Unrecognized value for default click spectrum type:  {self._default_click_spectrum_type}')

        info.set_visible(True)
        self._active_spectrum = info

        self._treeitem_active.setText(0, info.get_name())
        self._treeitem_active.setHidden(False)
        self._treeitem_active.setData(0, Qt.UserRole, info)
        self._treeitem_active.setIcon(0, get_color_icon(self._active_spectrum_color))

        self._act_collect_spectrum.setEnabled(True)
        self._draw_spectra()


    def clear_active_spectrum(self):
        self._active_spectrum = None
        self._treeitem_active.setHidden(True)

        self._act_collect_spectrum.setEnabled(False)
        self._draw_spectra()
