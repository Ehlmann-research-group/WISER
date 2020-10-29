from enum import Enum
import os, traceback

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import gui.generated.resources

from .app_state import ApplicationState, StateChange
from .export_plot_image import ExportPlotImageDialog
from .spectrum_plot_config import SpectrumPlotConfigDialog
from .spectrum_info_editor import SpectrumInfoEditor

from .util import add_toolbar_action, get_random_matplotlib_color, get_color_icon

from raster.envi_spectral_library import ENVISpectralLibrary
from raster.spectra import SpectrumType, SpectrumAverageMode, calc_rect_spectrum
from raster.spectra_export import export_spectrum_list
from raster.spectrum_info import SpectrumInfo, LibrarySpectrum
import raster.units

import matplotlib
matplotlib.use('Qt5Agg')
# TODO(donnie):  Seems to generate errors:
# matplotlib.rcParams['backend.qt5'] = 'PySide2'

import matplotlib.pyplot as plt

import numpy as np
from astropy import units as u

from matplotlib.backends.backend_qt5agg import FigureCanvas

from typing import List, Optional, Tuple


def generate_ticks(min_value: float, max_value: float, tick_interval: float) -> List[float]:
    '''
    Given the specified range and tick interval, this function returns a list of
    values within that range where tick marks are to appear.  If no tick marks
    will fall within that range, an empty list is returned.  For example, if the
    min_value is 5 and the max_value is 7, and the tick_interval is 100, an
    empty list will be returned since no tick marks will fall within that range.
    '''

    # print(f'range=[{min_value} , {max_value}] - interval={tick_interval}')

    ticks = []

    # Instead of looping on specific tick values, which has the potential to
    # accumulate errors, compute the integer "tick count," which is how many
    # ticks to get past the min_value.  Then, we can increment the tick
    # count to move forward, rather than adding in the tick_interval value.

    tick_count = int(min_value // tick_interval)
    if abs(tick_count * tick_interval - min_value) <= 1e-6:
        # The start of the range is near a tick mark.
        ticks.append(tick_count * tick_interval)

    while True:
        tick_count += 1
        tick_value = tick_count * tick_interval
        if tick_value > max_value:
            break

        ticks.append(tick_value)

    # print(f'ticks={ticks}')
    return ticks


def get_font_properties(font_name: str, font_size: float) \
        -> matplotlib.font_manager.FontProperties:
    # FontProperties(family=None, style=None, variant=None, weight=None,
    #                stretch=None, size=None, fname=None)
    return matplotlib.font_manager.FontProperties(family=font_name,
        size=font_size)


class SpectrumPlotCanvas(FigureCanvas):
    '''
    This is a simple subclass of the matplotlib FigureCanvas that adds in
    support for responding to context-menu events from Qt.  All other matplotlib
    FigureCanvas functionality is supported.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_context_menu = None

    def set_context_menu_fn(self, on_context_menu):
        '''
        Set the on-context-menu function to the specified function.
        The function must have the same signature as
        QWidget.contextMenuEvent(event).
        '''
        self._on_context_menu = on_context_menu

    def contextMenuEvent(self, event):
        '''
        This function is called by Qt when the user requests a context menu.
        '''
        if self._on_context_menu is not None:
            self._on_context_menu(event)


class SpectrumDisplayInfo:
    '''
    This class is used within the spectrum-plot window to record display
    information for a specific spectrum being displayed.
    '''

    def __init__(self, spectrum: SpectrumInfo):
        '''
        *   id is the numeric ID assigned to the spectrum
        *   line2d is the matplotlib line for the spectrum's data
        '''
        self._spectrum = spectrum

        if self._spectrum.get_color() is None:
            self._spectrum.set_color(get_random_matplotlib_color())

        self._icon: Optional[QIcon] = None
        self._line2d = None

    def reset(self) -> None:
        self._icon = None
        self.remove_plot()


    def get_icon(self) -> QIcon:
        if self._icon is None:
            self._icon = get_color_icon(self._spectrum.get_color())

        return self._icon


    def get_spectrum(self) -> SpectrumInfo:
        return self._spectrum


    def generate_plot(self, axes, use_wavelengths, wavelength_units=u.nm):
        # If we already have a plot, remove it.
        self.remove_plot()

        values = self._spectrum.get_spectrum()
        color = self._spectrum.get_color()
        linewidth = 0.5

        if use_wavelengths:
            # We should only be told to use wavelengths if all displayed spectra
            # have wavelengths for the bands.
            assert(self._spectrum.has_wavelengths())

            # If we can use wavelengths, each spectrum is a series of
            # (wavelength, value) coordinates, which we can plot.  This allows
            # the graphs to look correct, even in the face of bad bands, plots
            # from different datasets with different wavelengths, etc.

            wavelengths = raster.units.get_band_values(
                self._spectrum.get_wavelengths(), wavelength_units)

            lines = axes.plot(wavelengths, values, color=color,
                linewidth=linewidth, label=self._spectrum.get_name())
            assert(len(lines) == 1)
            self._line2d = lines[0]
        else:
            # If we don't have wavelengths, each spectrum is just a series of
            # values.  We can of course plot this, but we can't guarantee it
            # will be meaningful if there are multiple plots from different
            # datasets to display.

            lines = axes.plot(values, color=color, linewidth=linewidth,
                label=self._spectrum.get_name())
            assert(len(lines) == 1)
            self._line2d = lines[0]


    def remove_plot(self):
        if self._line2d is not None:
            self._line2d.remove()
            self._line2d = None


class SpectrumPointDisplayInfo:
    '''
    This class is used within the spectrum-plot window to record display
    information for a specific point being displayed.
    '''

    def __init__(self, spectrum, band_index: int, use_wavelength: bool,
                 band_units=None):
        # Info we need about the spectrum to show the desired point
        self._spectrum = spectrum
        self._band_index = band_index
        self._use_wavelength = use_wavelength
        self._band_units = band_units

        # matplotlib information for the point
        self._point_hline: Optional[matplotlib.lines.Line2D] = None
        self._point_vline: Optional[matplotlib.lines.Line2D] = None
        self._scatter: Optional[matplotlib.collections.PathCollection] = None
        self._label: Optional[matplotlib.text.Text] = None


    def get_spectrum(self):
        ''' Returns the spectrum with the point being displayed. '''
        return self._spectrum


    def get_band_index(self):
        ''' Returns the band-index tof the point on the spectrum to show. '''
        return self._band_index


    def generate_plot(self, axes):
        assert self._point_hline is None
        assert self._point_vline is None
        assert self._scatter is None
        assert self._label is None

        if self._use_wavelength:
            x_value = self._spectrum.get_wavelengths()[self._band_index]

            if self._band_units is not None:
                x_value = raster.units.convert_spectral(x_value, self._band_units)
        else:
            x_value = self._band_index

        y_value = self._spectrum.get_spectrum()[self._band_index]

        self._point_vline = axes.axvline(x=x_value.value,
            linewidth=0.5, linestyle='dotted', color='black')

        self._point_hline = axes.axhline(y=y_value,
            linewidth=0.5, linestyle='dotted', color='black')

        # Other possible markers:
        #     '+' plus
        #     's' square
        #     '.' small circle
        #     'o' big circle
        #     '_' horizontal line
        #     'D' diamond
        self._scatter = axes.scatter(x_value, y_value, # label=self._text,
            marker='s', s=3, linewidth=0.5, color='black')

        # Put some text in the top left corner - use the axis coordinate system
        # to achieve this.
        label = f'{x_value} = {y_value}'
        self._label = axes.text(0.02, 0.98, label,
            fontsize=4, # backgroundcolor='#ffffff7f',
            bbox={'pad':1, 'color':'white', 'alpha':0.8, 'fill':True},
            horizontalalignment='left', verticalalignment='top',
            transform=axes.transAxes)


    def remove_plot(self):
        if self._point_vline is not None:
            self._point_vline.remove()
            self._point_vline = None

            self._point_hline.remove()
            self._point_hline = None

            self._scatter.remove()
            self._scatter = None

            self._label.remove()
            self._label = None


class SpectrumPlot(QWidget):
    '''
    This widget provides a spectrum-plot window in the user interface.
    '''

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)

        # Initialize widget's internal state

        self._app_state = app_state

        #=====================================================================
        # General configuration for the spectrum plot

        # Are we displaying a legend?
        self._legend_location: Optional[str] = None

        # Font information for the plot
        self._font_name = None
        self._title_font_size: int = 8
        self._axes_font_size: int = 7
        self._legend_font_size: int = 5
        self._point_font_size: int = 5

        # X-range and Y-range information.  If the range is unspecified on a
        # given axis then the plot is auto-ranged along that axis.
        self._x_range: Optional[Tuple[float, float]] = None
        self._y_range: Optional[Tuple[float, float]] = None

        # Major and minor tick information for both axes.  If the value is None
        # then that tick is disabled.
        self._x_major_tick_interval: Optional[float] = None
        self._x_minor_tick_interval: Optional[float] = None
        self._y_major_tick_interval: Optional[float] = None
        self._y_minor_tick_interval: Optional[float] = None

        # Since the X-axis often has units associated with it, this field holds
        # the units used for the plot.
        self._x_units = None

        # Display information for all spectra being plotted
        self._spectrum_display_info: Dict[int, SpectrumDisplayInfo] = {}
        self._plot_uses_wavelengths: Optional[bool] = None
        self._displayed_spectra_with_wavelengths = 0

        # Display information for mouse clicks
        self._click: Optional[SpectrumPointDisplayInfo] = None

        # Display state for the "active spectrum"
        self._active_spectrum_color = None

        # This is the currently selected treeview item.  Initially, no item is
        # selected, so we set this to None.
        self._selected_treeview_item = None

        # Initialize UI components of the widget

        self._init_ui()

        # Set up event handlers

        self._app_state.active_spectrum_changed.connect(self._on_active_spectrum_changed)
        self._app_state.collected_spectra_changed.connect(self._on_collected_spectra_changed)

        self._app_state.spectral_library_added.connect(self._on_spectral_library_added)
        self._app_state.spectral_library_removed.connect(self._on_spectral_library_removed)


    def _init_ui(self):

        #==================================================
        # TOOLBAR

        self._toolbar = QToolBar(self.tr('Spectrum Toolbar'), parent=self)
        self._toolbar.setIconSize(QSize(20, 20))

        self._act_collect_spectrum = add_toolbar_action(self._toolbar,
            ':/icons/collect-spectrum.svg', self.tr('Collect spectrum'), self)
        self._act_collect_spectrum.triggered.connect(self._on_collect_spectrum)

        self._act_load_spectral_library = add_toolbar_action(self._toolbar,
            ':/icons/load-spectra.svg', self.tr('Load spectral library'), self)
        self._act_load_spectral_library.triggered.connect(self._on_load_spectral_library)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._toolbar.addWidget(spacer)

        self._act_configure = add_toolbar_action(self._toolbar,
            ':/icons/configure.svg', self.tr('Configure'), self)
        self._act_configure.triggered.connect(self._on_configure)

        # TODO(donnie):  Get rid of this eventually.  Similar functionality will
        #     be exposed in the spectral library tree.
        # self._act_clear_all_plots = add_toolbar_action(self._toolbar,
        #     ':/icons/clear-all-plots.svg', self.tr('Clear all plots'), self)
        # self._act_clear_all_plots.triggered.connect(self._on_clear_all_plots)

        #==================================================
        # Set up Matplotlib

        self._figure, self._axes = plt.subplots(tight_layout=True)

        self._axes.tick_params(direction='in', labelsize=4, pad=2, width=0.5,
            bottom=True, left=True, top=False, right=False)

        self._figure_canvas = SpectrumPlotCanvas(self._figure)

        # self.axes.set_autoscalex_on(True)
        # self.axes.set_autoscaley_on(False)
        # self.axes.set_ylim((0, 1))

        #==================================================
        # Tree-widget for managing spectral library

        self._spectra_tree = QTreeWidget()
        self._spectra_tree.setColumnCount(1)
        self._spectra_tree.setHeaderLabels([self.tr('Spectra and Spectral Libraries')])

        # The first item always represents the active spectrum.
        self._treeitem_active = QTreeWidgetItem()
        self._spectra_tree.addTopLevelItem(self._treeitem_active)
        self._treeitem_active.setHidden(True)

        # The second item always represents the collected spectra.
        self._treeitem_collected = QTreeWidgetItem([self.tr('Collected Spectra')])
        self._spectra_tree.addTopLevelItem(self._treeitem_collected)
        self._treeitem_collected.setHidden(True)

        # Subsequent top-level tree items represent spectral libraries.
        self._library_treeitems: Dict[int, QTreeWidgetItem] = {}

        # Events from the spectral library widget

        self._spectra_tree.currentItemChanged.connect(self._on_tree_selection_changed)
        self._spectra_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._spectra_tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        #==================================================
        # Spectrum-Edit Dialog

        self._spectrum_edit_dialog = SpectrumInfoEditor()

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

        #==================================================
        # Events

        if self._app_state.get_config('feature-flags.interactive-spectrum-plot', default=False, as_type=bool):
            self._figure_canvas.setContextMenuPolicy(Qt.DefaultContextMenu)
            self._figure_canvas.set_context_menu_fn(self._on_plot_context_menu)

            self._figure_canvas.mpl_connect('button_release_event', self._on_mpl_button_release_event)
            # self._figure_canvas.mpl_connect('pick_event', self._on_mpl_pick_event)



    def sizeHint(self):
        ''' The default size of the spectrum-plot widget is 400x200. '''
        return QSize(400, 200)


    def get_app_state(self):
        return self._app_state


    def get_axes(self):
        return self._axes


    def get_x_range(self) -> Optional[Tuple[float, float]]:
        return self._x_range


    def set_x_range(self, range: Optional[Tuple[float, float]]):
        self._x_range = range


    def get_y_range(self) -> Optional[Tuple[float, float]]:
        return self._y_range


    def set_y_range(self, range: Optional[Tuple[float, float]]) -> None:
        self._y_range = range



    def get_x_major_tick_interval(self) -> Optional[float]:
        return self._x_major_tick_interval

    def set_x_major_tick_interval(self, interval: Optional[float]) -> None:
        if interval is not None and interval <= 0:
            raise ValueError(f'interval must be >= 0; got {interval}')
        self._x_major_tick_interval = interval

    def get_x_minor_tick_interval(self) -> Optional[float]:
        return self._x_minor_tick_interval

    def set_x_minor_tick_interval(self, interval: Optional[float]) -> None:
        if interval is not None and interval <= 0:
            raise ValueError(f'interval must be >= 0; got {interval}')
        self._x_minor_tick_interval = interval


    def get_y_major_tick_interval(self) -> Optional[float]:
        return self._y_major_tick_interval

    def set_y_major_tick_interval(self, interval: Optional[float]) -> None:
        if interval is not None and interval <= 0:
            raise ValueError(f'interval must be >= 0; got {interval}')
        self._y_major_tick_interval = interval

    def get_y_minor_tick_interval(self) -> Optional[float]:
        return self._y_minor_tick_interval

    def set_y_minor_tick_interval(self, interval: Optional[float]) -> None:
        if interval is not None and interval <= 0:
            raise ValueError(f'interval must be >= 0; got {interval}')
        self._y_minor_tick_interval = interval


    def get_legend(self) -> Optional[str]:
        '''
        If the legend is enabled, this fucntion returns the current location of
        the legend.  If the legend is disabled then this function returns None.
        '''
        return self._legend_location


    def set_legend(self, location: Optional[str]) -> None:
        '''
        Enables or disables the legend on the plot, along with the location that
        the legend should appear.

        If location is a string specifying where to place the legend, then the
        legend will be displayed accordingly.  Possible location values are as
        specified by the pyplot.legend() function:
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend

        If location is None then the legend is hidden.
        '''
        self._legend_location = location


    def _add_spectrum_to_plot(self, spectrum, treeitem):
        display_info = SpectrumDisplayInfo(spectrum)
        self._spectrum_display_info[spectrum.get_id()] = display_info

        # Figure out whether we should use wavelengths or not in the plot.
        use_wavelengths = False
        if spectrum.has_wavelengths():
            # TODO(donnie):  This is ugly.  Find a way to expose wavelength units
            #     on datasets and spectra.
            self._x_units = spectrum.get_wavelengths()[0].unit

            self._displayed_spectra_with_wavelengths += 1
            if self._displayed_spectra_with_wavelengths == len(self._spectrum_display_info):
                use_wavelengths = True

        if use_wavelengths == self._plot_uses_wavelengths:
            # Nothing has changed, so just generate a plot for the new spectrum
            display_info.generate_plot(self._axes, use_wavelengths)

        else:
            # Need to regenerate all plots with the new "use wavelengths" value

            axes_font = get_font_properties(self._font_name, self._axes_font_size)
            if use_wavelengths:
                self._axes.set_xlabel(f'Wavelength ({self._x_units})',
                    labelpad=0, fontproperties=axes_font)
                self._axes.set_ylabel('Value', labelpad=0, fontproperties=axes_font)
            else:
                self._axes.set_xlabel('Band Index', labelpad=0, fontproperties=axes_font)
                self._axes.set_ylabel('Value', labelpad=0, fontproperties=axes_font)

            for other_info in self._spectrum_display_info.values():
                other_info.generate_plot(self._axes, use_wavelengths)

            self._plot_uses_wavelengths = use_wavelengths

        # Show the plot's color in the tree widget
        treeitem.setIcon(0, display_info.get_icon())

        return display_info


    def _remove_spectrum_from_plot(self, spectrum, treeitem):
        id = spectrum.get_id()
        display_info = self._spectrum_display_info.get(id)
        if display_info is None:
            # Already not displaying the spectrum - we are done!
            return

        del self._spectrum_display_info[id]

        # Figure out whether we should use wavelengths or not in the plot.
        if spectrum.has_wavelengths():
            self._displayed_spectra_with_wavelengths -= 1

        display_info.remove_plot()

        # Hide the plot's color in the tree widget
        treeitem.setIcon(0, QIcon())

        # Are we showing a point on this spectrum?
        if self._click is not None and self._click.get_spectrum() is spectrum:
            self._click.remove_plot()
            self._click = None


    def _on_plot_context_menu(self, event):
        '''
        This function shows a context menu on the plot window.
        '''

        menu = QMenu(self)

        # TODO(donnie):  Eventually, probably expose options for picked spectra

        act = menu.addAction(self.tr('Configure plot...'))
        act.triggered.connect(self._on_configure)

        act = menu.addAction(self.tr('Export plot to image...'))
        act.triggered.connect(self._on_export_plot_to_image)

        menu.exec_(event.globalPos())


    def _on_mpl_button_release_event(self, event):

        ''' Debug info for spectrum-plot clicks
        print('-' * 70)
        print('_on_mpl_button_release_event')
        print(f'MPL Event name={event.name} type={type(event)} ' +
              f'x={event.x} y={event.y} ' +
              f'xdata={event.xdata} ydata={event.ydata}')
        print(f'event.inaxes={event.inaxes}')
        print(f'event.guiEvent={event.guiEvent}')
        '''

        # The guiEvent's type is QMouseEvent.  If this wasn't a left mouse-click
        # then ignore the event.  (For example, if this was a right-click, Qt
        # will respond with a context-menu notification.)
        if event.guiEvent.button() != Qt.LeftButton:
            return

        # If the event doesn't indicate a mouse-click within the axes then
        # ignore the event.
        if event.xdata is None or event.ydata is None:
            return

        # Clear any click display-info from previous mouse clicks
        if self._click is not None:
            self._click.remove_plot()
            self._click = None

        # Find all spectra with X-axis ranges that correspond to the xdata value
        # from the mouse click.

        closest_spectrum = None
        closest_index = None
        closest_x_value = None
        closest_y_value = None
        closest_distance = None

        if self._plot_uses_wavelengths:
            # Find the spectrum that has a wavelength near the clicked
            # wavelength (x-value), and that also has the closest spectrum-value
            # (y-value) for that wavelength.

            click_x = event.xdata * self._x_units
            click_y = event.ydata

            for (id, display_info) in self._spectrum_display_info.items():
                spectrum = display_info.get_spectrum()
                assert spectrum.has_wavelengths()

                # See if the spectrum has a wavelength "near" the x-coordinate
                # of the mouse-click.
                # TODO(donnie):  Filter out bad wavelengths here.
                wavelengths = spectrum.get_wavelengths()

                # TODO(donnie):  specify max_distance so that we are within a
                #     few pixels of the x_data value.
                index = raster.units.find_closest_wavelength(wavelengths, click_x)
                if index is None:
                    continue

                # Compute the spectrum's y-distance from the y-coordinate of
                # the mouse-click.

                x_value = wavelengths[index].value
                y_value = spectrum.get_spectrum()[index]
                if np.isnan(y_value):
                    continue

                distance = abs(y_value - click_y)

                if closest_distance is None or distance < closest_distance:
                    closest_spectrum = spectrum
                    closest_index = index
                    closest_x_value = x_value
                    closest_y_value = y_value
                    closest_distance = distance

            if closest_spectrum is not None:
                # print(f'Closest spectrum value:  id={closest_spectrum.get_id()} index={closest_index} y_value={closest_y_value}')

                self._click = SpectrumPointDisplayInfo(closest_spectrum, closest_index, True)
                self._click.generate_plot(self._axes)

        else:
            # Find the spectrum that has valid bands near the clicked x-value.
            pass

        self._draw_spectra()


    def _on_export_plot_to_image(self, act):
        dialog = ExportPlotImageDialog(self._figure, parent=self)
        dialog.exec_()


    def _on_configure(self):
        '''
        This event-handler gets called when the user invokes the spectrum
        configuration dialog.
        '''
        cfg_dialog = SpectrumPlotConfigDialog(self, parent=self)
        cfg_dialog.exec_()
        self._draw_spectra()


    def _on_load_spectral_library(self):
        '''
        This function handles the "load spectral library" button on the spectrum
        display widget.  It gets the filename of a spectrum file from the user,
        then asks the application-state to load it.  If the app-state succeeds,
        it will fire an event that this widget receives, which will cause it to
        show the list of spectra in the library.
        '''

        # TODO(donnie):  This should probably be on the main application.
        #     It can live here for now, but it will need to be migrated
        #     elsewhere in the future.

        # These are all file formats that will appear in the file-open dialog
        supported_formats = [
            self.tr('ENVI spectral libraries (*.hdr *.sli)'),
            self.tr('All files (*)'),
        ]

        selected = QFileDialog.getOpenFileName(self,
            self.tr("Open Spectal Library File"),
            self._app_state.get_current_dir(), ';;'.join(supported_formats))

        if len(selected[0]) > 0:
            try:
                # Load the spectral library into the application state
                self._app_state.open_file(selected[0])
            except Exception as e:
                mbox = QMessageBox(QMessageBox.Critical,
                    self.tr('Could not open file'),
                    self.tr('The file {0} could not be opened.').format(selected[0]),
                    QMessageBox.Ok, parent=self)

                mbox.setInformativeText(str(e))
                mbox.setDetailedText(traceback.format_exc())

                mbox.exec()


    def _on_spectral_library_added(self, lib_id: int):
        '''
        This function handles the signal that a spectral library was added to
        the application state.  It updates the user interface to show every
        spectrum in the library.

        The argument to the function is the ID assigned to the spectral library
        in the application state.
        '''
        # TODO(donnie):  Put spectra / spectral library info onto each tree item
        #     so we can implement context menus properly.

        spectral_library = self._app_state.get_spectral_library(lib_id)

        # Add a new top-level tree item for the spectral library.  Set the tree
        # item's user data value to the library's ID
        treeitem_library = QTreeWidgetItem([spectral_library.get_name()])
        treeitem_library.setData(0, Qt.UserRole, lib_id)
        self._spectra_tree.addTopLevelItem(treeitem_library)
        self._library_treeitems[lib_id] = treeitem_library

        # Create a tree-item for every spectrum in the library.
        for i in range(spectral_library.num_spectra()):
            spectrum = LibrarySpectrum(spectral_library, i)
            treeitem_spectrum = QTreeWidgetItem([spectrum.get_name()])
            treeitem_spectrum.setData(0, Qt.UserRole, spectrum)

            treeitem_library.addChild(treeitem_spectrum)


    def _on_spectral_library_removed(self, lib_id: int):
        '''
        This function handles the signal that a spectral library was removed
        from the application state.  It updates the user interface to remove the
        library's spectra from the UI.

        The argument to the function is the ID assigned to the spectral library
        in the application state.
        '''
        # Look up the library's tree-item so we can clean up all state
        # associated with the library.

        treeitem = self._library_treeitems[lib_id]

        # First step:  delete the widget entry from the library-spectra mapping.
        del self._library_treeitems[lib_id]

        # Remove all visible spectra in the library from the spectrum-plot
        for i in range(treeitem.childCount()):
            child_treeitem = treeitem.child(i)
            spectrum = child_treeitem.data(0, Qt.UserRole)
            display_info = self._spectrum_display_info.get(spectrum.get_id())
            if display_info is not None:
                # Make invisible
                self._remove_spectrum_from_plot(spectrum, child_treeitem)

        # Remove the tree-item for the library from the tree widget.
        index = self._spectra_tree.indexOfTopLevelItem(treeitem)
        self._spectra_tree.takeTopLevelItem(index)

        # If any of the library's spectra were drawn, update the plot state.
        self._draw_spectra()


    def _on_active_spectrum_changed(self):
        '''
        This function handles the signal that the application's active spectrum
        value was changed (possibly to None).  It synchronizes the UI with the
        new value.

        Active spectrum values will only be RasterDataSetSpectrum objects.
        '''

        # Do we have an old spectrum to remove from the UI?

        old_spectrum = self._treeitem_active.data(0, Qt.UserRole)
        if old_spectrum is not None:
            self._remove_spectrum_from_plot(old_spectrum, self._treeitem_active)

        # Update the UI to match the new state.

        spectrum = self._app_state.get_active_spectrum()
        if spectrum is not None:
            # There is a (possibly new) active spectrum.  Set up to display it.

            # If the active spectrum specifies a color, use that color.
            # Otherwise, use our current "active spectrum color" (generating one
            # if necessary).
            spectrum_color = spectrum.get_color()
            if spectrum_color is not None:
                self._active_spectrum_color = None

            else:
                if self._active_spectrum_color is None:
                    self._active_spectrum_color = get_random_matplotlib_color()

                spectrum_color = self._active_spectrum_color
                spectrum.set_color(spectrum_color)

            display_info = self._add_spectrum_to_plot(spectrum, self._treeitem_active)

            # Update the tree-item for the active spectrum
            self._treeitem_active.setText(0, spectrum.get_name())

        # Update the state of the "active spectrum" tree item based on the value
        self._treeitem_active.setHidden(spectrum is None)
        self._treeitem_active.setData(0, Qt.UserRole, spectrum)
        self._act_collect_spectrum.setEnabled(spectrum is not None)
        self._draw_spectra()


    def _on_collect_spectrum(self):
        if self._app_state.get_active_spectrum() is None:
            # The "collect spectrum" button shouldn't be enabled if there is no
            # active spectrum!
            warning.warn('Shouldn\'t be able to collect spectrum when no active spectrum!')
            return

        # This will cause the app-state to emit both an "active spectrum
        # changed" event, and a "collected spectra changed" event.
        self._app_state.collect_active_spectrum()
        self._active_spectrum_color = None


    def _on_collected_spectra_changed(self, change, index):
        if change == StateChange.ITEM_ADDED:
            spectrum = self._app_state.get_collected_spectra()[index]
            treeitem = QTreeWidgetItem([spectrum.get_name()])
            treeitem.setData(0, Qt.UserRole, spectrum)

            self._treeitem_collected.insertChild(index, treeitem)
            self._treeitem_collected.setHidden(False)
            self._treeitem_collected.setExpanded(True)

            self._add_spectrum_to_plot(spectrum, treeitem)

        elif change == StateChange.ITEM_REMOVED:
            if index is not None:
                treeitem = self._treeitem_collected.takeChild(index)
                spectrum = treeitem.data(0, Qt.UserRole)
                self._remove_spectrum_from_plot(spectrum, treeitem)

            else:
                # All collected items are discarded.

                while self._treeitem_collected.childCount() > 0:
                    treeitem = self._treeitem_collected.takeChild(0)
                    spectrum = treeitem.data(0, Qt.UserRole)
                    self._remove_spectrum_from_plot(spectrum, treeitem)

            if self._treeitem_collected.childCount() == 0:
                self._treeitem_collected.setHidden(True)

        self._draw_spectra()


    def _on_tree_selection_changed(self, current, previous):
        # print(f'selection changed.  current.data = {current.data(0, Qt.UserRole)}')
        self._selected_treeview_item = current
        self._draw_spectra()


    def _on_tree_context_menu(self, pos):
        '''
        This function handles the context-menu event when the user requests a
        context menu on the tree widget.  Depending on the item chosen, the
        context menu will be populated with appropriate options.
        '''
        # Figure out which tree item was clicked on.
        treeitem = self._spectra_tree.itemAt(pos)
        if treeitem is None:
            return

        menu = QMenu(self)

        if treeitem is self._treeitem_active:
            # This is the Active spectrum.  It is always visible, so there will
            # be no show/hide option here.
            act = menu.addAction(self.tr('Collect'))
            act.triggered.connect(lambda *args : self._on_collect_spectrum())

            act = menu.addAction(self.tr('Edit...'))
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_edit_spectrum(treeitem))

            menu.addSeparator()

            act = menu.addAction(self.tr('Discard...'))
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_discard_spectrum(treeitem))

        elif treeitem is self._treeitem_collected:
            # This is the whole "Collected Spectra" group; these are unsaved
            # spectra that the user has collected.

            act = menu.addAction(self.tr('Show all spectra'))
            # act.setData(treeitem)
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_show_all_spectra(treeitem))

            act = menu.addAction(self.tr('Hide all spectra'))
            # act.setData(treeitem)
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_hide_all_spectra(treeitem))

            act = menu.addAction(self.tr('Save to file...'))
            act.triggered.connect(lambda *args :
                                  self._on_save_collected_spectra())

            menu.addSeparator()

            act = menu.addAction(self.tr('Discard all...'))
            act.triggered.connect(lambda *args :
                                  self._on_discard_collected_spectra())

        elif treeitem.parent() is None:
            # This is a spectral library

            # TODO(donnie):  Add this when we support editing spectral libraries
            # act = menu.addAction(self.tr('Save edits...'))

            act = menu.addAction(self.tr('Show all spectra'))
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_show_all_spectra(treeitem))

            act = menu.addAction(self.tr('Hide all spectra'))
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_hide_all_spectra(treeitem))

            menu.addSeparator()

            act = menu.addAction(self.tr('Unload library'))
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_unload_library(treeitem))

        else:
            # This is a specific spectrum plot (other than the active spectrum),
            # either in the collected spectra, or in a spectral library.

            spectrum = treeitem.data(0, Qt.UserRole)
            display_info = self._spectrum_display_info.get(spectrum.get_id())

            # TODO(donnie):  Show/hide option
            act = menu.addAction(self.tr('Show spectrum'))
            act.setCheckable(True)
            act.setChecked(display_info is not None)
            act.setData(spectrum)
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_toggle_spectrum_visible(treeitem))

            if not isinstance(spectrum, LibrarySpectrum):
                act = menu.addAction(self.tr('Edit...'))
                act.triggered.connect(lambda *args, treeitem=treeitem :
                                      self._on_edit_spectrum(treeitem))

                menu.addSeparator()

                act = menu.addAction(self.tr('Discard...'))
                act.triggered.connect(lambda *args, treeitem=treeitem :
                                      self._on_discard_spectrum(treeitem))

        global_pos = self._spectra_tree.mapToGlobal(pos)
        menu.exec_(global_pos)


    def _on_toggle_spectrum_visible(self, treeitem):
        '''
        This function handles the context-menu option for toggling the
        visibility of a spectrum plot.
        '''
        spectrum = treeitem.data(0, Qt.UserRole)
        display_info = self._spectrum_display_info.get(spectrum.get_id())

        # Toggle the visibility of the spectrum.
        if display_info is None:
            # Make visible
            self._add_spectrum_to_plot(spectrum, treeitem)

        else:
            # Make invisibile
            self._remove_spectrum_from_plot(spectrum, treeitem)

        self._draw_spectra()


    def _on_edit_spectrum(self, treeitem):
        '''
        This function handles the "Edit spectrum info" context-menu option.
        This should only show up on raster data-set spectra, never on library
        spectra, since they are not currently editable.
        '''
        spectrum = treeitem.data(0, Qt.UserRole)

        self._spectrum_edit_dialog.configure_ui(spectrum)
        if self._spectrum_edit_dialog.exec() != QDialog.Accepted:
            # User canceled out of the edit.
            return

        # If we got here, update the tree-view item with the spectrum's info.

        treeitem.setText(0, spectrum.get_name())

        # Is the spectrum currently being displayed?
        display_info = self._spectrum_display_info.get(spectrum.get_id())
        if display_info is not None:
            display_info.reset()
            display_info.generate_plot(self._axes, self._plot_uses_wavelengths)
            treeitem.setIcon(0, display_info.get_icon())

        # Are we showing a point on this spectrum?
        if self._click is not None and self._click.get_spectrum() is spectrum:
            self._click.remove_plot()
            self._click.generate_plot(self._axes)

        self._draw_spectra()


    def _on_discard_spectrum(self, treeitem):
        spectrum = treeitem.data(0, Qt.UserRole)

        # Get confirmation from the user.
        confirm = QMessageBox.question(self, self.tr('Discard Spectrum?'),
            self.tr('Are you sure you want to discard this spectrum?') +
            '\n\n' + spectrum.get_name())

        if confirm != QMessageBox.Yes:
            # User canceled the discard operation.
            return

        # If we got here, we are discarding the spectrum.
        if treeitem.parent() is self._treeitem_collected:
            # The spectrum is in the collected spectra.
            index = self._treeitem_collected.indexOfChild(treeitem)
            self._app_state.remove_collected_spectrum(index)
            # TODO:
            '''
            self._treeitem_collected.takeChild(index)
            if len(self._collected_spectra) == 0:
                self._treeitem_collected.setHidden(True)
            '''
        else:
            # The spectrum is the active spectrum.
            self._app_state.set_active_spectrum(None)

        # Are we showing a point on the discarded spectrum?
        if self._click is not None and self._click.get_spectrum() is spectrum:
            self._click.remove_plot()



    def _on_save_collected_spectra(self):
        supported_formats = [
            self.tr('Text files (*.txt)'),
            self.tr('All files (*)'),
        ]

        selected = QFileDialog.getSaveFileName(self,
            self.tr('Save Collected Spectra'),
            self._app_state.get_current_dir(),
            ';;'.join(supported_formats))

        if len(selected[0]) > 0:
            # Make a list of spectra to be saved.
            spectra = []
            for i in range(self._treeitem_collected.childCount()):
                treeitem = self._treeitem_collected.child(i)
                spectra.append(treeitem.data(0, Qt.UserRole))

            export_spectrum_list(selected[0], spectra)


    def _on_discard_collected_spectra(self):
        '''
        This function implements the "discard all collected spectra" context
        menu operation.
        '''
        # Get confirmation from the user.
        confirm = QMessageBox.question(self, self.tr('Discard Collected Spectra?'),
            self.tr('Are you sure you want to discard all collected spectra?'))

        if confirm != QMessageBox.Yes:
            # User canceled the discard operation.
            return

        # If we got here, we are discarding all collected spectra.  Do the
        # operation on the app-state; it will fire the appropriate event to
        # cause the UI to update properly.
        self._app_state.remove_all_collected_spectra()


    def _on_show_all_spectra(self, treeitem):
        '''
        This function implements the "show all spectra [in the group]" context
        menu operation.  It is available on the "collected spectra" group, and
        the loaded spectral library groups.
        '''
        for i in range(treeitem.childCount()):
            child_treeitem = treeitem.child(i)
            spectrum = child_treeitem.data(0, Qt.UserRole)
            display_info = self._spectrum_display_info.get(spectrum.get_id())

            # Toggle the visibility of the spectrum.
            if display_info is None:
                # Make visible
                self._add_spectrum_to_plot(spectrum, child_treeitem)

        self._draw_spectra()


    def _on_hide_all_spectra(self, treeitem):
        '''
        This function implements the "hide all spectra [in the group]" context
        menu operation.  It is available on the "collected spectra" group, and
        the loaded spectral library groups.
        '''

        for i in range(treeitem.childCount()):
            child_treeitem = treeitem.child(i)
            spectrum = child_treeitem.data(0, Qt.UserRole)
            display_info = self._spectrum_display_info.get(spectrum.get_id())

            # Toggle the visibility of the spectrum.
            if display_info is not None:
                # Make invisible
                self._remove_spectrum_from_plot(spectrum, child_treeitem)

        self._draw_spectra()


    def _on_unload_library(self, treeitem):
        '''
        This function implements the "unload spectral library" functionality.
        '''
        # Figure out which library needs to be removed, then ask the application
        # state to remove it.  This will cause a signal to be emitted,
        # indicating that the library was removed.  This widget will receive
        # that signal and update the UI appropriately.
        lib_id = treeitem.data(0, Qt.UserRole)
        self._app_state.remove_spectral_library(lib_id)


    def _draw_spectra(self):
        '''
        This helper function refreshes the matplotlib drawing canvas to reflect
        any changes that have been made to the plot.
        '''
        self._axes.relim(visible_only=True)

        # Legend:

        '''
        legend = self._axes.get_legend()
        if legend is not None:
            legend.remove()

        if self._legend_location is not None:
            # Need to add a legend.  Use the user-specified location, and also
            # the current font for the legend.
            legend_font = get_font_properties(self._font_name, self._legend_font_size)
            self._axes.legend(loc=self._legend_location, prop=legend_font)
        '''

        if self._legend_location is None:
            # Need to remove the legend
            legend = self._axes.get_legend()
            if legend is not None:
                legend.remove()

        else:
            # Need to add a legend.  Use the user-specified location, and also
            # the current font for the legend.
            legend_font = get_font_properties(self._font_name, self._legend_font_size)
            self._axes.legend(loc=self._legend_location, prop=legend_font)

        # All done!
        self._figure_canvas.draw()
