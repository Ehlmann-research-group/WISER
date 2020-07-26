import json, os, pathlib, sys, traceback
from typing import List, Tuple

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

from .app_state import ApplicationState

from .spectrum_info import SpectrumAtPoint
from raster.selection import SinglePixelSelection



class DataVisualizerApp(QMainWindow):

    def __init__(self):
        '''
        Initialize the data-visualization app.  This method initializes the
        model, various views, and hooks them together with the controller code.
        '''
        super().__init__(None)
        self.setWindowTitle(self.tr('Imaging Spectroscopy Workbench'))

        # Internal state

        self._app_state = ApplicationState(self)

        # Application Toolbars

        self._init_menus()

        self._main_toolbar = self.addToolBar(self.tr('Main'))
        self._main_toolbar.setObjectName('main_toolbar') # Needed for UI persistence
        self._init_toolbars()

        # Status bar
        self.statusBar().showMessage(
            self.tr('Welcome to the Imaging Spectroscopy Workbench'), 10000)

        # Context pane

        self._context_pane = ContextPane(self._app_state)
        self._make_dockable_pane(self._context_pane, name='context_pane',
            title=self.tr('Context'), icon=':/icons/context-pane.svg',
            tooltip=self.tr('Show/hide the context pane'),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea,
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
        # Application menu
        self._app_menu = self.menuBar().addMenu(sys.argv[0])

        act = self._app_menu.addAction(self.tr('About Imaging Spectroscopy Workbench'))
        act.setMenuRole(QAction.AboutRole)
        act.triggered.connect(self.show_about_dialog)

        # File menu

        self._file_menu = self.menuBar().addMenu(self.tr('&File'))

        act = self._file_menu.addAction(self.tr('&Open...'))
        act.setShortcuts(QKeySequence.Open)
        act.setStatusTip(self.tr('Open an existing project or file'))
        act.triggered.connect(self.show_open_file_dialog)

        act = self._file_menu.addAction(self.tr('Save &project file...'))
        act.setStatusTip(self.tr('Save the current project configuration'))
        act.triggered.connect(self.show_save_project_dialog)

        # View menu

        self._view_menu = self.menuBar().addMenu(self.tr('&View'))

        # Other menus?

        # self.window_menu = self.menuBar().addMenu(self.tr('&Window'))
        # self.help_menu = self.menuBar().addMenu(self.tr('&Help'))


    def _init_toolbars(self):
        act = add_toolbar_action(self._main_toolbar, ':/icons/open-image.svg',
            'Open image file', self)
        act.triggered.connect(self.show_open_file_dialog)

        self._main_toolbar.addSeparator()


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


    def closeEvent(self, event):
        # TODO(donnie):  Ask user to save any unsaved state?  (This also means
        #     we must detect unsaved state.)

        # TODO(donnie):  Save Qt state.

        super().closeEvent(event)


    def show_about_dialog(self, evt):
        '''
        Shows the "About the Imaging Spectroscopy Workbench" dialog in the
        user interface.
        '''
        about = AboutDialog(self)
        about.exec()


    def show_open_file_dialog(self, evt):
        '''
        Shows the "Open File..." dialog in the user interface.  If the user
        successfully chooses a file, the open_file() method is called to
        perform the actual operation of opening the file.
        '''

        # These are all file formats that will appear in the file-open dialog
        supported_formats = [
            self.tr('ISWB project files (*.iswb)'),
            self.tr('ENVI raster files (*.img *.hdr)'),
            self.tr('TIFF raster files (*.tiff *.tif *.tfw)'),
            self.tr('PDS raster files (*.PDS *.IMG)'),
            self.tr('ENVI spectral libraries (*.sli *.hdr)'),
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
                traceback.print_exc()

                mbox = QMessageBox(QMessageBox.Critical,
                    self.tr('Could not open file'),
                    self.tr('The file could not be opened.'),
                    QMessageBox.Ok, parent=self)

                # TODO(donnie):  I don't know what we might say for the
                #     informative text.
                # mbox.setInformativeText(traselected[0])

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
            self.tr('ISWB files (*.iswb)'),
            self.tr('All files (*)'),
        ]

        selected = QFileDialog.getSaveFileName(self,
            self.tr("Open ISWB Project File"),
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
        settings = QSettings('Caltech', 'Imaging Spectroscopy Workbench')
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
        settings = QSettings('Caltech', 'Imaging Spectroscopy Workbench')
        self.restoreGeometry(settings.value('geometry'))
        self.restoreState(settings.value('window-state'))


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

        # Create a single-pixel selection with the dataset and coordinate of
        # the selected pixel.
        ds = self._main_view.get_current_dataset(rasterview_position)
        sel = SinglePixelSelection(ds_point, ds)

        # Update the main and zoom windows to show the selected dataset and pixel.
        self._main_view.set_pixel_highlight(sel, recenter=RecenterMode.NEVER)
        self._zoom_pane.show_dataset(ds)
        self._zoom_pane.set_pixel_highlight(sel)

        # Set the active spectrum to be from the selected dataset and pixel.
        # TODO(donnie):  Need to include the default area and average mode too!
        spectrum = SpectrumAtPoint(ds, ds_point.toTuple())
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

        # Create a single-pixel selection with the dataset and coordinate of
        # the selected pixel.
        ds = self._main_view.get_current_dataset(rasterview_position)
        sel = SinglePixelSelection(ds_point, ds)

        # Update the main and zoom windows to show the selected dataset and pixel.
        self._main_view.set_pixel_highlight(sel, recenter=RecenterMode.IF_NOT_VISIBLE)
        self._zoom_pane.set_pixel_highlight(sel, recenter=RecenterMode.NEVER)

        # Set the active spectrum to be from the selected dataset and pixel.
        # TODO(donnie):  Need to include the default area and average mode too!
        spectrum = SpectrumAtPoint(ds, ds_point.toTuple())
        self._app_state.set_active_spectrum(spectrum)
