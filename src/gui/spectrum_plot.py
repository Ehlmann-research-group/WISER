from enum import Enum
import os

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import gui.generated.resources

from .spectrum_info import SpectrumInfo
from .spectrum_plot_config import SpectrumPlotConfigDialog
from .spectrum_info_editor import SpectrumInfoEditor

from .util import add_toolbar_action, get_random_matplotlib_color

from raster.envi_spectral_library import ENVISpectralLibrary
from raster.selection import Selection, SinglePixelSelection
from raster.spectra import SpectrumType, SpectrumAverageMode, calc_rect_spectrum
from raster.units import get_band_values

import matplotlib
matplotlib.use('Qt5Agg')
# TODO(donnie):  Seems to generate errors:
# matplotlib.rcParams['backend.qt5'] = 'PySide2'

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from matplotlib.backends.backend_qt5agg import FigureCanvas


class SpectrumPlot(QWidget):
    '''
    This widget provides a spectrum-plot window in the user interface.
    '''

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)

        # Initialize widget's internal state

        self._app_state = app_state

        # For user mouse clicks, these are the parameters for generating spectra

        self._default_area_avg_x = 1
        self._default_area_avg_y = 1
        self._default_average_mode = SpectrumAverageMode.MEAN

        # Display state for the "active spectrum"
        self._active_spectrum_color = None

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

        self._app_state.active_spectrum_changed.connect(self._on_active_spectrum_changed)

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


    def sizeHint(self):
        ''' The default size of the spectrum-plot widget is 400x200. '''
        return QSize(400, 200)


    def _on_configure(self):
        '''
        This event-handler gets called when the user invokes the spectrum
        configuration dialog.
        '''
        cfg_dialog = SpectrumPlotConfigDialog(self,
            default_avg_mode=self._default_average_mode,
            default_area_avg_x=self._default_area_avg_x,
            default_area_avg_y=self._default_area_avg_y)

        result = cfg_dialog.exec_()

        if result == QDialog.Accepted:
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
        treeitem_library.setData(0, Qt.UserRole, index)
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
        # Look at all the tree-widget items for the spectral libraries
        for index_ti in range(2, self._spectra_tree.topLevelItemCount()):
            treeitem = self._spectra_tree.topLevelItem(index_ti)
            data = treeitem.data(0, Qt.UserRole)
            if data == index:
                # Found the library.  Remove the tree-item for it, and also
                # clean up the internal state.
                self._spectra_tree.takeTopLevelItem(index_ti)
                del self._library_spectra[index]
                break

        # Indexes larger than the library indexes need to be adjusted to match
        # the application state.
        for index_ti in range(2, self._spectra_tree.topLevelItemCount()):
            treeitem = self._spectra_tree.topLevelItem(index_ti)
            data = treeitem.data(0, Qt.UserRole)
            if data > index:
                # Indexes larger than the library indexes need to be adjusted
                # to match the application state.
                infos = self._library_spectra.pop(data)
                data -= 1
                treeitem.setData(0, Qt.UserRole, data)
                self._library_spectra[data] = infos

        self._draw_spectra()


    def _on_active_spectrum_changed(self):
        spectrum = self._app_state.get_active_spectrum()
        if spectrum is not None:
            # There is a (possibly new) active spectrum.  Set up to display it.

            # If the active spectrum specifies a color, use that color.
            # Otherwise, use our current "active spectrum color" (generating one
            # if necessary).
            spectrum_color = spectrum.get_color()
            if spectrum_color is not None:
                self._active_spectrum_color = spectrum_color

            else:
                if self._active_spectrum_color is None:
                    self._active_spectrum_color = get_random_matplotlib_color()

                spectrum.set_color(self._active_spectrum_color)

            # Update the tree-item for the active spectrum
            self._treeitem_active.setText(0, spectrum.get_name())
            self._treeitem_active.setIcon(0, spectrum.get_icon())

        self._treeitem_active.setHidden(spectrum is None)
        self._treeitem_active.setData(0, Qt.UserRole, spectrum)
        self._act_collect_spectrum.setEnabled(spectrum is not None)
        self._draw_spectra()


    def _on_collect_spectrum(self):
        '''
        active_spectrum = self._app_state.get_active_spectrum()

        if active_spectrum is None:
            # The "collect spectrum" button shouldn't be enabled if there is no
            # active spectrum!
            warning.warn('Shouldn\'t be able to collect spectrum when no active spectrum!')
            return

        # If the active spectrum is the currently selected one, disable the
        # selection and clear our internal state.
        if self._selected_treeview_item is self._treeitem_active:
            self._treeitem_active.setSelected(False)
            self._selected_treeview_item = None

        # Create a new tree-entry for the new spectrum we are collecting

        self._app_state.collect_active_spectrum()
        self._active_spectrum_color = None

        treeitem_collected = QTreeWidgetItem([self._active_spectrum.get_name()])
        treeitem_collected.setData(0, Qt.UserRole, collected_spectrum)
        treeitem_collected.setIcon(0, collected_spectrum.get_icon())

        self._treeitem_collected.setHidden(False)
        self._treeitem_collected.setExpanded(True)

        self._treeitem_collected.addChild(treeitem_collected)

        # Select the newly collected spectrum
        self._spectra_tree.setCurrentItem(treeitem_collected, 0, QItemSelectionModel.SelectCurrent)

        self.clear_active_spectrum()
        '''
        self._app_state.collect_active_spectrum()
        self._active_spectrum_color = None


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
            act.triggered.connect(lambda *args : self._on_collect_spectrum())

            act = menu.addAction(self.tr('Edit...'))
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_edit_spectrum(treeitem))

            menu.addSeparator()

            act = menu.addAction(self.tr('Discard...'))
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_discard_spectrum(treeitem))

        elif treeitem is self._treeitem_collected:
            # This is the Collected Spectra group; these are unsaved spectra
            # that the user has collected.

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
            # act.setData(treeitem)
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_show_all_spectra(treeitem))

            act = menu.addAction(self.tr('Hide all spectra'))
            # act.setData(treeitem)
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_hide_all_spectra(treeitem))

            menu.addSeparator()

            act = menu.addAction(self.tr('Unload library'))
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_unload_library(treeitem))

        else:
            # This is a specific spectrum plot (other than the active spectrum),
            # either in the collected spectra, or in a spectral library.

            info = treeitem.data(0, Qt.UserRole)

            # TODO(donnie):  Show/hide option
            act = menu.addAction(self.tr('Show spectrum'))
            act.setCheckable(True)
            act.setChecked(info.is_visible())
            act.setData(info)
            act.triggered.connect(lambda *args, treeitem=treeitem :
                                  self._on_toggle_spectrum_visible(treeitem))

            if isinstance(info, CollectedSpectrum):
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
        info = treeitem.data(0, Qt.UserRole)

        # Toggle the visibility of the spectrum.
        new_visible = not info.is_visible()
        info.set_visible(new_visible)

        icon = QIcon()
        if new_visible:
            icon = info.get_icon()

        treeitem.setIcon(0, icon)

        self._draw_spectra()


    def _on_edit_spectrum(self, treeitem):
        info = treeitem.data(0, Qt.UserRole)

        self._spectrum_edit_dialog.configure_ui(info)
        if self._spectrum_edit_dialog.exec() != QDialog.Accepted:
            # User canceled out of the edit.
            return

        # If we got here, update the tree-view item with the spectrum's info.

        treeitem.setText(0, info.get_name())

        if info.is_visible():
            treeitem.setIcon(0, info.get_icon())

        self._draw_spectra()


    def _on_discard_spectrum(self, treeitem):
        info = treeitem.data(0, Qt.UserRole)

        # Get confirmation from the user.
        confirm = QMessageBox.question(self, self.tr('Discard Spectrum?'),
            self.tr('Are you sure you want to discard this spectrum?') +
            '\n\n' + info.get_name())

        if confirm != QMessageBox.Yes:
            # User canceled the discard operation.
            return

        # If we got here, we are discarding the spectrum.

        if treeitem.parent() is self._treeitem_collected:
            # The spectrum is in the collected spectra.
            index = self._treeitem_collected.indexOfChild(treeitem)
            del self._collected_spectra[index]
            self._treeitem_collected.takeChild(index)

            if len(self._collected_spectra) == 0:
                self._treeitem_collected.setHidden(True)
        else:
            # The spectrum is the active spectrum.
            self.clear_active_spectrum()


    def _on_save_collected_spectra(self):
        supported_formats = [
            self.tr('CSV files (*.csv)'),
            self.tr('All files (*)'),
        ]

        selected = QFileDialog.getSaveFileName(self,
            self.tr('Save Collected Spectra'),
            self._app_state.get_current_dir(),
            ';;'.join(supported_formats))

        if len(selected[0]) > 0:
            print(f'TODO(donnie) - save spectra to {selected[0]}!')


    def _on_discard_collected_spectra(self):
        # Get confirmation from the user.
        confirm = QMessageBox.question(self, self.tr('Discard Collected Spectra?'),
            self.tr('Are you sure you want to discard all collected spectra?'))

        if confirm != QMessageBox.Yes:
            # User canceled the discard operation.
            return

        # If we got here, we are discarding the collected spectra.

        self._treeitem_collected.takeChildren()
        self._treeitem_collected.setHidden(True)
        self._collected_spectra.clear()

        self._draw_spectra()


    def _on_show_all_spectra(self, treeitem):
        # print(f'action = {action}')
        # treeitem = action.data()

        if treeitem is self._treeitem_collected:
            for index, info in enumerate(self._collected_spectra):
                info.set_visible(True)
                treeitem.child(index).setIcon(0, info.get_icon())

        else:
            # The action is for a spectral library
            library_index = treeitem.data(0, Qt.UserRole)

            for index, info in enumerate(self._library_spectra[library_index]):
                info.set_visible(True)
                treeitem.child(index).setIcon(0, info.get_icon())

        self._draw_spectra()


    def _on_hide_all_spectra(self, treeitem):
        # treeitem = action.data()

        icon = QIcon()
        if treeitem is self._treeitem_collected:
            for index, info in enumerate(self._collected_spectra):
                info.set_visible(False)
                treeitem.child(index).setIcon(0, icon)

        else:
            # The action is for a spectral library
            library_index = treeitem.data(0, Qt.UserRole)

            for index, info in enumerate(self._library_spectra[library_index]):
                info.set_visible(False)
                treeitem.child(index).setIcon(0, icon)

        self._draw_spectra()


    def _on_unload_library(self, treeitem):
        # Figure out which library needs to be removed, then ask the application
        # state to remove it.  This will cause a signal to be emitted,
        # indicating that the library was removed.  This widget will receive
        # that signal and update the UI appropriately.
        library_index = treeitem.data(0, Qt.UserRole)
        print(f'Unloading spectral library at index {library_index}')
        self._app_state.remove_spectral_library(library_index)


    def _draw_spectra(self):
        self._axes.clear()

        # Build up a list of all spectra that could be displayed.  The order of
        # spectra in the list is also the order they are drawn, so add library
        # spectra first, then collected spectra, then the active spectrum, and
        # finally any highlighted spectrum.

        all_spectra = []

        # Library spectra
        for info_list in self._library_spectra.values():
            all_spectra.extend(info_list)

        # Collected spectra

        # import pprint
        # pprint.pprint(self._collected_spectra)

        all_spectra.extend(self._app_state.get_collected_spectra())

        # Active spectrum
        active_spectrum = self._app_state.get_active_spectrum()
        if active_spectrum is not None:
            all_spectra.append(active_spectrum)

        # Selected spectrum - don't forget that groups can also be selected
        selected_spectrum = None
        if self._selected_treeview_item is not None:
            data = self._selected_treeview_item.data(0, Qt.UserRole)
            if isinstance(data, SpectrumInfo):
                all_spectra.append(data)

        # Filter down spectra to only the ones that are being displayed
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
                    linewidth = 1.5

                # TODO(donnie):  Support configurable default wavelength units
                wavelengths = get_band_values(info.get_wavelengths(), u.nm)
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
                    linewidth = 1.5

                dataset = info.get_dataset()
                spectrum = info.get_spectrum()
                self._axes.plot(spectrum, linewidth=linewidth)

        self._figure_canvas.draw()
