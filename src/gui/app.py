import json
import os
import pathlib
import platform
import sys
import traceback

from typing import Dict, List, Optional, Tuple

import bugsnag

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .app_config import PixelReticleType

from .about_dialog import AboutDialog

from .rasterpane import RecenterMode

from .dockable import DockablePane

from .context_pane import ContextPane
from .main_view import MainViewWidget
from .zoom_pane import ZoomPane

from .spectrum_plot import SpectrumPlot
from .infoview import DatasetInfoView

from .util import *

from .app_config import ApplicationConfig, get_wiser_config_dir
from .app_config_dialog import AppConfigDialog
from .app_state import ApplicationState
from . import bug_reporting

import plugins

from .bandmath_dialog import BandMathDialog
import bandmath

from raster.spectra import SpectrumAverageMode
from raster.spectrum_info import SpectrumAtPoint
from raster.selection import SinglePixelSelection



class DataVisualizerApp(QMainWindow):

    def __init__(self, config_path: Optional[str] = None,
                       config: Optional[ApplicationConfig] = None):
        '''
        Initialize the data-visualization app.  This method initializes the
        model, various views, and hooks them together with the controller code.
        '''
        super().__init__(None)
        self.setWindowTitle(self.tr('Workbench for Imaging Spectroscopy Exploration and Research'))

        # Internal state

        if config_path is None:
            config_path = get_wiser_config_dir()

        self._config_path: str = config_path

        self._app_state: ApplicationState = ApplicationState(self, config=config)

        # Application Toolbars

        self._init_menus()

        self._main_toolbar = self.addToolBar(self.tr('Main'))
        self._main_toolbar.setObjectName('main_toolbar') # Needed for UI persistence
        self._init_toolbars()

        # Plugins

        self._init_plugins()

        # Status bar
        self.statusBar().showMessage(
            self.tr('Welcome to WISER - the Workbench for Imaging Spectroscopy Exploration and Research'), 10000)

        # Context pane

        self._context_pane = ContextPane(self._app_state)
        self._make_dockable_pane(self._context_pane, name='context_pane',
            title=self.tr('Context'), icon=':/icons/context-pane.svg',
            tooltip=self.tr('Show/hide the context pane'),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
                          Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea,
            area=Qt.LeftDockWidgetArea)

        # Zoom pane

        self._zoom_pane = ZoomPane(self._app_state)
        dockable = self._make_dockable_pane(self._zoom_pane, name='zoom_pane',
            title=self.tr('Zoom'), icon=':/icons/zoom-pane.svg',
            tooltip=self.tr('Show/hide the zoom pane'),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea,
            area=Qt.RightDockWidgetArea)
        dockable.hide()

        # Main raster-view

        self._main_view = MainViewWidget(self._app_state)
        self.setCentralWidget(self._main_view)

        self._image_toolbar = self._main_view.get_toolbar()
        self.addToolBar(self._image_toolbar)
        self._image_toolbar.setObjectName('image_toolbar') # Needed for UI persistence

        # Spectrum plot

        self._spectrum_plot = SpectrumPlot(self._app_state)
        dockable = self._make_dockable_pane(self._spectrum_plot, name='spectrum_plot',
            title=self.tr('Spectrum Plot'), icon=':/icons/spectrum-pane.svg',
            tooltip=self.tr('Show/hide the spectrum pane'),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea,
            area=Qt.RightDockWidgetArea)
        dockable.hide()

        # Dataset Information Window

        # TODO(donnie):  Why do we need a scroll area here?  The QTreeWidget is
        #     a scroll-area too!!
        self._dataset_info = DatasetInfoView(self._app_state)
        # scroll_area = QScrollArea()
        # scroll_area.setWidget(self.info_view)
        # scroll_area.setWidgetResizable(True)
        dockable = self._make_dockable_pane(self._dataset_info, name='dataset_info',
            title=self.tr('Dataset Info'), icon=':/icons/dataset-info.svg',
            tooltip=self.tr('Show/hide dataset information'),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea,
            area=Qt.LeftDockWidgetArea)
        dockable.hide()

        # Hook up widget events to their corresponding control functions.

        self._context_pane.click_pixel.connect(self._on_context_raster_pixel_select)
        self._context_pane.display_bands_change.connect(self._on_display_bands_change)

        self._main_view.viewport_change.connect(self._on_mainview_viewport_change)
        self._main_view.click_pixel.connect(self._on_mainview_raster_pixel_select)
        self._main_view.display_bands_change.connect(self._on_display_bands_change)
        self._main_view.create_selection.connect(self._on_create_selection)

        self._main_view.get_stretch_builder().stretch_changed.connect(self._on_stretch_changed)

        self._zoom_pane.viewport_change.connect(self._on_zoom_viewport_change)
        self._zoom_pane.click_pixel.connect(self._on_zoom_raster_pixel_select)
        self._zoom_pane.visibility_change.connect(self._on_zoom_visibility_changed)
        self._zoom_pane.display_bands_change.connect(self._on_display_bands_change)
        self._zoom_pane.create_selection.connect(self._on_create_selection)


    def _init_menus(self):

        # Configure the menus based on the OS/platform
        system_name = platform.system()

        # Application menu

        if system_name == 'Darwin':
            # macOS has an application menu that includes "about" and
            # "preferences" entries, and a "Quit XXXX" entry
            self._app_menu = self.menuBar().addMenu('WISER')

            act = self._app_menu.addAction(self.tr('About WISER'))
            act.setMenuRole(QAction.AboutRole)
            act.triggered.connect(self.show_about_dialog)

            act = self._app_menu.addAction(self.tr('Preferences...'))
            act.setMenuRole(QAction.PreferencesRole)
            act.triggered.connect(self.show_preferences)

            act = self._app_menu.addAction(self.tr('&Quit WISER'))
            act.setMenuRole(QAction.QuitRole)
            act.triggered.connect(self.quit_app)

        # File menu

        self._file_menu = self.menuBar().addMenu(self.tr('&File'))

        act = self._file_menu.addAction(self.tr('&Open...'))
        act.setShortcuts(QKeySequence.Open)
        act.setStatusTip(self.tr('Open an existing project or file'))
        act.triggered.connect(self.show_open_file_dialog)

        act = self._file_menu.addAction(self.tr('Save &project file...'))
        act.setStatusTip(self.tr('Save the current project configuration'))
        act.triggered.connect(self.show_save_project_dialog)

        self._file_menu.addSeparator()

        # TODO(donnie) - this will require a lot of testing
        # self._close_dataset_menu = self._file_menu.addMenu(self.tr('Close dataset...'))
        # act.setStatusTip(self.tr('Close an open dataset or spectral library'))
        # self._close_dataset_menu.setEnabled(False)
        #
        # self._file_menu.addSeparator()

        if system_name == 'Windows':
            act = self._file_menu.addAction(self.tr('Settings...'))
            act.setMenuRole(QAction.PreferencesRole)
            act.triggered.connect(self.show_preferences)

            self._file_menu.addSeparator()

            act = self._file_menu.addAction(self.tr('E&xit WISER'))
            act.setMenuRole(QAction.QuitRole)
            act.triggered.connect(self.quit_app)

        # View menu

        self._view_menu = self.menuBar().addMenu(self.tr('&View'))

        # Tools menu

        self._tools_menu = self.menuBar().addMenu(self.tr('&Tools'))

        act = self._tools_menu.addAction(self.tr('Band math...'))
        act.triggered.connect(self.show_bandmath_dialog)

        # Windows:  Help menu

        if platform.system() == 'Windows':
            self._help_menu = self.menuBar().addMenu(self.tr('&Help'))

            act = self._help_menu.addAction(self.tr('About WISER'))
            act.setMenuRole(QAction.AboutRole)
            act.triggered.connect(self.show_about_dialog)


    def _init_toolbars(self):
        act = add_toolbar_action(self._main_toolbar, ':/icons/open-image.svg',
            'Open image file', self)
        act.triggered.connect(self.show_open_file_dialog)

        self._main_toolbar.addSeparator()

        # If the bug-button feature flag is on, make a button that will trigger
        # an error, so that we can exercise online error reporting.
        if self._app_state.get_config('feature_flags.bug_button', default=False):
            def _raise_bug():
                raise Exception('Intentional exception for testing online bug reporting')

            act = add_toolbar_action(self._main_toolbar, ':/icons/bug.svg',
                'Generate an error!', self)
            act.triggered.connect(lambda checked=False: _raise_bug())

            self._main_toolbar.addSeparator()


    def _init_plugins(self):
        plugin_classes = self._app_state.get_config('plugins')
        for pc in plugin_classes:
            print(f'Instantiating plugin class "{pc}"')
            try:
                plugin = plugins.instantiate(pc)

            except Exception as e:
                print(f'ERROR:  Couldn\'t instantiate plugin class "{pc}"!')
                traceback.print_exc(limit=3)
                continue

            if (not isinstance(plugin, plugins.ToolsMenuPlugin) and
                not isinstance(plugin, plugins.ContextMenuPlugin) and
                not isinstance(plugin, plugins.BandMathPlugin)):
                print(f'ERROR:  Unrecognized plugin type "{pc}", skipping')
                continue

            self._app_state.add_plugin(pc, plugin)

            # Let "Tools"-menu plugins add their actions to the menu.
            if isinstance(plugin, plugins.ToolsMenuPlugin):
                plugin.add_tool_menu_items(self._tools_menu)


    def _make_dockable_pane(self, widget, name, title, icon, tooltip,
                            allowed_areas, area):

        dockable = DockablePane(widget, name, title, self._app_state,
                                icon=icon, tooltip=tooltip, parent=self)

        dockable.setAllowedAreas(allowed_areas)
        self.addDockWidget(area, dockable)

        # TODO(donnie):  Technically we don't need to get the icon and tooltip
        #     from the dockable, since we have it above.
        act = dockable.toggleViewAction()
        act.setIcon(dockable.get_icon())
        act.setToolTip(dockable.get_tooltip())

        self._view_menu.addAction(act)
        self._main_toolbar.addAction(act)

        return dockable


    def show_status_text(self, text: str, seconds: int=0):
        self.statusBar().showMessage(text, seconds * 1000)


    def quit_app(self):
        ''' User-triggered operation to exit the application. '''

        # TODO(donnie):  Ask user to save any unsaved state?  (This also means
        #     we must detect unsaved state.)

        # TODO(donnie):  Maybe save Qt state?

        # Exit WISER
        QApplication.exit(0)


    def closeEvent(self, event):
        # TODO(donnie):  Ask user to save any unsaved state?  (This also means
        #     we must detect unsaved state.)

        # TODO(donnie):  Maybe save Qt state?

        super().closeEvent(event)


    def show_about_dialog(self, evt):
        ''' Shows the "About WISER" dialog in the user interface. '''
        about = AboutDialog(self)
        about.exec()


    def show_preferences(self, evt):
        ''' Shows the WISER preferences / config dialog. '''
        config_dialog = AppConfigDialog(self._app_state, parent=self)
        if config_dialog.exec() == QDialog.Accepted:
            # Save the configuration file
            self._app_state.config().save(
                os.path.join(self._config_path, 'wiser-conf.json'))

            # The only config property that is not applied automatically is the
            # BugSnag reporting configuration.  Do that here.
            auto_notify = self._app_state.config().get('general.online_bug_reporting')
            bug_reporting.set_enabled(auto_notify)


    def show_open_file_dialog(self, evt):
        '''
        Shows the "Open File..." dialog in the user interface.  If the user
        successfully chooses a file, the open_file() method is called to
        perform the actual operation of opening the file.
        '''

        # These are all file formats that will appear in the file-open dialog
        supported_formats = [
            self.tr('ENVI raster files (*.img *.hdr)'),
            self.tr('TIFF raster files (*.tiff *.tif *.tfw)'),
            self.tr('PDS raster files (*.PDS *.IMG)'),
            self.tr('ENVI spectral libraries (*.sli *.hdr)'),
            self.tr('WISER project files (*.wiser)'),
            self.tr('All files (*)'),
        ]

        selected = QFileDialog.getOpenFileName(self,
            self.tr("Open Spectal Data File"),
            self._app_state.get_current_dir(), ';;'.join(supported_formats))
        # print(selected)

        if len(selected[0]) > 0:
            try:
                # Open the file on the application state.
                self._app_state.open_file(selected[0])
            except Exception as e:
                mbox = QMessageBox(QMessageBox.Critical,
                    self.tr('Could not open file'),
                    self.tr('The file {0} could not be opened.').format(selected[0]),
                    QMessageBox.Ok, parent=self)

                mbox.setInformativeText(str(e))
                mbox.setDetailedText(traceback.format_exc())

                mbox.exec()


    def show_save_project_dialog(self, evt):
        '''
        Shows the "Save Project..." dialog in the user interface.  If the user
        successfully chooses a file, the save_project_file() method is called to
        perform the actual operation of saving the project details.
        '''

        # These are all file formats that will appear in the file-open dialog
        supported_formats = [
            self.tr('WISER project files (*.wiser)'),
            self.tr('All files (*)'),
        ]

        selected = QFileDialog.getSaveFileName(self,
            self.tr("Open WISER Project File"),
            self._app_state.get_current_dir(), ';;'.join(supported_formats))
        # print(selected)

        if len(selected[0]) > 0:
            try:
                self.save_project_file(selected[0])
            except:
                mbox = QMessageBox(QMessageBox.Critical,
                    self.tr('Could not save project'), QMessageBox.Ok, self)

                mbox.setText(self.tr('Could not write project file.'))
                mbox.setInformativeText(file_path)

                # TODO(donnie):  Add exception-trace info here, using
                #     mbox.setDetailedText()

                mbox.exec()


    def save_project_file(self, file_path, force=False):
        '''
        Saves the entire project state to the specified file path.  This
        includes the following:

        *   Data sets that are loaded
        *   Regions of interest
        *   Qt application state including window geometry and open/close state
        '''
        # TODO(donnie):  If the project file already exists, and force is False,
        #     prompt the user about overwriting the file.

        project_info = self.generate_project_info()
        with open(file_path, 'w') as f:
            # Make the JSON output pretty so that advanced users can understand
            # it.
            json.dump(project_info, f, sort_keys=True, indent=4)

        msg = self.tr('Saved project to {}').format(file_path)
        self.statusBar().showMessage(msg, 5000)


    def generate_project_info(self):
        '''
        Generates a Python dictionary containing the current project state,
        which can then be written out as a JSON file.  This includes the
        following:

        *   Data sets that are loaded
        *   Regions of interest
        *   Qt application state including window geometry and open/close state
        '''
        project_info = {}

        # TODO(donnie):  Project description, owner, email, ...

        # Data sets
        # TODO(donnie):  This will get more sophisticated when we have multiple
        #     layers, and the like.

        project_info['datasets'] = []
        for data_set in self._app_state.get_datasets():
            ds_info = {
                'files': data_set.get_filepaths(),
            }

            # TODO(donnie):  data-set stretch, current display bands

            project_info['datasets'].append(ds_info)

        # Regions of interest

        project_info['regions_of_interest'] = []
        for (name, roi) in self._app_state.get_rois().items():
            assert name == roi.get_name()
            roi_info = roi_to_pyrep(roi)
            project_info['regions_of_interest'].append(roi_info)

        # The .toBase64() function returns a QByteArray, which we then convert
        # to a Python byte-array.  Finally, convert to Python str object to
        # save.  The base-64 encoding should be fine for UTF-8 conversion.
        project_info['qt_geometry'] = self.saveGeometry().toBase64().data().decode()
        project_info['qt_window_state'] = self.saveState().toBase64().data().decode()

        return project_info


    def load_project_file(self, file_path, force=False):
        '''
        Loads project state from the specified file path.  This includes the
        following:

        *   Data sets that are loaded
        *   Regions of interest
        *   Qt application state including window geometry and open/close state
        '''
        # TODO(donnie):  If we have in-memory-only project state, and force is
        #     False, prompt the user about loading the file.

        with open(file_path) as f:
            # Make the JSON output pretty so that advanced users can understand
            # it.
            project_info = json.load(f)
            self.apply_project_info(project_info)


    def apply_project_info(self, project_info):
        # TODO(donnie):  Surely we will also have to reset the UI widgets.
        #     Perhaps it would be better to put a clear_all() or reset()
        #     operation on the ApplicationState class, which can fire an event
        #     to views.
        # self._app_state = ApplicationState()

        for ds_info in project_info['datasets']:
            # The first file in the list is usually the one that we load.
            filename = ds_info['files'][0]
            self._app_state.open_file(filename)

            # TODO(donnie):  data-set stretch, current display bands

        # Regions of interest

        for roi_info in project_info['regions_of_interest']:
            # Reconstruct the region of interest
            roi = roi_from_pyrep(roi_info)
            self._app_state.add_roi(roi)

        # Qt window state/geometry

        s = project_info['qt_geometry']
        qba = QByteArray(bytes(s, 'utf-8'))
        self.restoreGeometry(QByteArray.fromBase64(qba))

        s = project_info['qt_window_state']
        qba = QByteArray(bytes(s, 'utf-8'))
        self.restoreState(QByteArray.fromBase64(qba))


    def save_qt_settings(self):
        '''
        Save the Qt application state (window geometry, and state of toolbars
        and dock widgets) using the platform-independent QSettings mechanism.
        This is used when the user doesn't want to use the project-settings JSON
        file to save and load settings.
        '''
        # TODO(donnie):  Store company/app name in some central constants file
        settings = QSettings('Caltech', 'WISER')
        settings.setValue('geometry', self.saveGeometry())
        settings.setValue('window-state', self.saveState())

    def load_qt_settings(self):
        '''
        Load the Qt application state (window geometry, and state of toolbars
        and dock widgets) using the platform-independent QSettings mechanism.
        This is used when the user doesn't want to use the project-settings JSON
        file to save and load settings.
        '''
        # TODO(donnie):  Store company/app name in some central constants file
        settings = QSettings('Caltech', 'WISER')
        self.restoreGeometry(settings.value('geometry'))
        self.restoreState(settings.value('window-state'))


    def show_bandmath_dialog(self):
        dialog = BandMathDialog(self._app_state)
        if dialog.exec() == QDialog.Accepted:
            expression = dialog.get_expression()
            print(f'Evaluate band math:  {expression}')

            variables = dialog.get_variable_bindings()

            # Collect functions from all plugins.
            functions = {}
            for (plugin_name, plugin) in self._app_state.get_plugins().items():
                if isinstance(plugin, plugins.BandMathPlugin):
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
                            print(f'WARNING:  Function "{k}" is defined ' +
                                  f'multiple times (last seen in plugin {name})')

                    functions.update(plugin_fns)

            try:
                (result_type, result) = bandmath.eval_bandmath_expr(expression,
                    variables, functions)

                print(f'Result of band-math evaluation is type {result_type}')
                print(f'Result is:\n{result}')

                if result_type == bandmath.VariableType.IMAGE_CUBE:
                    loader = self._app_state.get_loader()
                    new_dataset = loader.from_numpy_array(result)
                    self._app_state.add_dataset(new_dataset)

                elif result_type == bandmath.VariableType.IMAGE_BAND:
                    # Convert the image band into a 1-band image cube
                    result = result[np.newaxis, :]
                    loader = self._app_state.get_loader()
                    new_dataset = loader.from_numpy_array(result)
                    self._app_state.add_dataset(new_dataset)

                elif result_type == bandmath.VariableType.SPECTRUM:
                    # new_spectrum = bandmath.result_to_spectrum(result_type, result)
                    # self._app_state.set_active_spectrum(new_spectrum)
                    print('TODO:  create new spectrum')

            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(self, self.tr('Bandmath Evaluation Error'),
                    self.tr('Couldn\'t evaluate band-math expression') +
                    f'\n{expression}\n' + self.tr('Reason:') + f'\n{e}')
                return


    def _on_display_bands_change(self, ds_id: int, bands: Tuple, is_global: bool):
        '''
        When the user changes the display bands used in one of the raster panes,
        the pane will fire an event that the application controller can receive,
        if other raster panes also need to be updated.
        '''
        if is_global:
            self._context_pane.set_display_bands(ds_id, bands)
            self._main_view.set_display_bands(ds_id, bands)
            self._zoom_pane.set_display_bands(ds_id, bands)


    def _on_create_selection(self, selection):
        self._app_state.add_selection(selection)


    def _on_context_raster_pixel_select(self, rasterview_position, ds_point):
        '''
        When the user clicks the mouse in the context pane, the main view is
        updated to show that location in the center of the main window.
        '''
        # In the context pane, the rasterview position should always be (0, 0).
        assert rasterview_position == (0, 0)

        # Make all the views in the main image window show the point.
        self._main_view.make_point_visible(ds_point.x(), ds_point.y(),
            rasterview_pos=None)


    def _on_mainview_viewport_change(self, rasterview_position):
        '''
        When the user scrolls the viewport in the main view, the context pane
        is updated to show the visible area.
        '''
        # TODO(donnie):  Handle this!!  Need to iterate through all raster-views
        #     and draw their viewports in the context pane.  If the main view
        #     has linked scrolling enabled, we only need to draw one box though.
        if rasterview_position is None:
            return

        if self._main_view.is_multi_view() and not self._main_view.is_scrolling_linked():
            # Get a list of all visible regions from all views
            visible_region = self._main_view.get_all_visible_regions()
        else:
            rasterview = self._main_view.get_rasterview(rasterview_position)
            visible_region = rasterview.get_visible_region()

        self._context_pane.set_viewport_highlight(visible_region)


    def _on_mainview_raster_pixel_select(self, rasterview_position, ds_point):
        '''
        When the user clicks in the main view, the following things happen:
        *   The pixel is shown in the center of the zoom pane, and a selection
            reticle is shown around the pixel.
        *   The spectrum of the pixel is shown in the spectrum-plot view.

        These operations occur whether the above panes are visible or not, so
        that if they were hidden and are then shown, they will still contain the
        relevant information.
        '''
        if self._app_state.num_datasets() == 0:
            return

        # Get the dataset of the main view.  If no dataset is being displayed,
        # this is a no-op.
        ds = self._main_view.get_current_dataset(rasterview_position)
        if ds is None:
            # The clicked-on rasterview has no dataset loaded; ignore.
            return

        # App behavior varies when we are in linked mode vs. not in linked mode
        if self._main_view.is_scrolling_linked():
            # Linked scrolling:  Don't change the dataset of any other panes;
            # just show the corresponding data in those panes' datasets.

            sel = SinglePixelSelection(ds_point, None)

            self._main_view.set_pixel_highlight(sel, recenter=RecenterMode.NEVER)
            self._zoom_pane.set_pixel_highlight(sel)

            # Set the active spectrum to be from the selected dataset and pixel.
            # TODO(donnie):  If the Spectrum Plot window has a "current dataset"
            #     set, get the spectrum from there instead.

            area = (self._app_state.get_config('spectra.default_area_avg_x', as_type=int),
                    self._app_state.get_config('spectra.default_area_avg_y', as_type=int))
            mode = self._app_state.get_config('spectra.default_area_avg_mode', as_type=lambda s : SpectrumAverageMode[s])
            spectrum = SpectrumAtPoint(ds, ds_point.toTuple(), area, mode)
            self._app_state.set_active_spectrum(spectrum)


        else:
            # Non-linked scrolling:  Change the dataset of other panes before
            # updating them to show the clicked data.

            sel = SinglePixelSelection(ds_point, ds)

            self._context_pane.show_dataset(ds)

            self._main_view.set_pixel_highlight(sel, recenter=RecenterMode.NEVER)

            self._zoom_pane.show_dataset(ds)
            self._zoom_pane.set_pixel_highlight(sel)

            # Set the active spectrum to be from the selected dataset and pixel.

            area = (self._app_state.get_config('spectra.default_area_avg_x', as_type=int),
                    self._app_state.get_config('spectra.default_area_avg_y', as_type=int))
            mode = self._app_state.get_config('spectra.default_area_avg_mode', as_type=lambda s : SpectrumAverageMode[s])
            spectrum = SpectrumAtPoint(ds, ds_point.toTuple(), area, mode)
            self._app_state.set_active_spectrum(spectrum)


    def _on_stretch_changed(self, ds_id: int, bands: Tuple, stretches: List):
        '''
        Receive stretch-change events from the Stretch Builder and record them
        in the application state.  Interested widgets can register for the
        state-change events on the application state.
        '''

        # print(f'Contrast stretch changed to:')
        # for s in stretches:
        #     print(f' * {s}')

        self._app_state.set_stretches(ds_id, bands, stretches)


    def _on_zoom_visibility_changed(self, visible):
        self._update_zoom_viewport_highlight()

    def _on_zoom_viewport_change(self, rasterview_position):
        '''
        When the user scrolls the viewport in the zoom pane, the main view
        is updated to show the visible area.
        '''
        # We can ignore rasterview_position since the zoom pane always has only
        # one raster-view.
        self._update_zoom_viewport_highlight()

    def _update_zoom_viewport_highlight(self):
        visible_area = None
        if self._zoom_pane.isVisible():
            visible_area = self._zoom_pane.get_rasterview().get_visible_region()

        self._main_view.set_viewport_highlight(visible_area)


    def _on_zoom_raster_pixel_select(self, rasterview_position, ds_point):
        '''
        When the user clicks in the zoom pane, the following things happen:
        *   A selection reticle is shown around the pixel.
        *   The spectrum of the pixel is shown in the spectrum-plot view.

        These operations occur whether the above panes are visible or not, so
        that if they were hidden and are then shown, they will still contain the
        relevant information.
        '''
        # In the zoom pane, the rasterview position should always be (0, 0).
        assert rasterview_position == (0, 0)

        if self._app_state.num_datasets() == 0:
            return

        # Get the dataset of the main view.  If no dataset is being displayed,
        # this is a no-op.
        ds = self._zoom_pane.get_current_dataset()
        if ds is None:
            # The clicked-on rasterview has no dataset loaded; ignore.
            return

        # App behavior varies when we are in linked mode vs. not in linked mode
        if self._main_view.is_scrolling_linked():
            # Linked scrolling:  Don't change the dataset of any other panes;
            # just show the corresponding data in those panes' datasets.

            sel = SinglePixelSelection(ds_point, None)

            # Update the main and zoom windows to show the selected dataset and pixel.
            self._main_view.set_pixel_highlight(sel, recenter=RecenterMode.IF_NOT_VISIBLE)
            self._zoom_pane.set_pixel_highlight(sel, recenter=RecenterMode.NEVER)

            # Set the active spectrum to be from the selected dataset and pixel.
            # TODO(donnie):  Need to include the default area and average mode too!
            # TODO(donnie):  If the Spectrum Plot window has a "current dataset"
            #     set, get the spectrum from there instead.
            spectrum = SpectrumAtPoint(ds, ds_point.toTuple())
            self._app_state.set_active_spectrum(spectrum)

        else:
            # Non-linked scrolling:  Change the dataset of other panes before
            # updating them to show the clicked data.

            sel = SinglePixelSelection(ds_point, ds)

            self._context_pane.show_dataset(ds)

            # If the dataset isn't showing in the main viewing area, show it.
            # Rationale:  The visible area of the zoom-pane is also indicated
            # in the main viewing area.
            if not self._main_view.is_showing_dataset(ds):
                self._main_view.show_dataset(ds)

            self._main_view.set_pixel_highlight(sel, recenter=RecenterMode.IF_NOT_VISIBLE)

            self._zoom_pane.set_pixel_highlight(sel, recenter=RecenterMode.NEVER)

            # Set the active spectrum to be from the selected dataset and pixel.
            # TODO(donnie):  Need to include the default area and average mode too!
            spectrum = SpectrumAtPoint(ds, ds_point.toTuple())
            self._app_state.set_active_spectrum(spectrum)
