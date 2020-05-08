import json, os, pathlib, sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .about_dialog import AboutDialog

from .rasterpane import RecenterMode, PixelReticleType

from .dockable import DockablePane

from .context_pane import ContextPane
from .main_view import MainViewWidget
from .zoom_pane import ZoomPane

from .spectrum_plot import SpectrumPlot
from .infoview import DatasetInfoView

from .roi import RegionOfInterest, roi_to_pyrep, roi_from_pyrep
from .util import *

from raster.dataset import *
from raster.spectral_library import SpectralLibrary

from raster.gdal_dataset import GDALRasterDataLoader
from raster.envi_spectral_library import ENVISpectralLibrary


class ApplicationState(QObject):
    '''
    This class holds all application state for the visualizer.  This includes
    both model state and view state.  This is primarily so the controller can
    access everything in one place, but it also allows the programmatic
    interface to operate on both the models and the views.
    '''

    # Signal:  a data-set was added at the specified index
    dataset_added = Signal(int)

    # Signal:  the data-set at the specified index was removed
    dataset_removed = Signal(int)


    spectral_library_added = Signal(int)

    spectral_library_removed = Signal(int)


    roi_added = Signal(RegionOfInterest)

    roi_removed = Signal(RegionOfInterest)

    # TODO(donnie):  Signals for config changes and color changes!

    def __init__(self):
        super().__init__()

        self._current_dir = os.getcwd()
        self._raster_data_loader = GDALRasterDataLoader()

        # All datasets loaded in the application.
        self._datasets = []

        # All spectral libraries loaded in the application.
        self._spectral_libraries = []

        # Regions of interest in the input data sets.
        self._regions_of_interest = {}
        self._next_roi_id = 1

        # Configuration options.
        self._config = {
            'pixel-reticle-type' : PixelReticleType.SMALL_CROSS,
        }

        # Colors used for various purposes.
        self._colors = {
            'viewport-highlight' : Qt.yellow,
            'pixel-highlight' : Qt.red,
            'roi-default-color' : Qt.white,
        }


    def get_current_dir(self):
        return self._current_dir


    def open_file(self, file_path):
        '''
        Open a data or configuration file in the Imaging Spectroscopy Workbench.
        '''

        # Remember the directory of the selected file, for next file-open
        self._current_dir = os.path.dirname(file_path)

        # Is the file a project file?

        if file_path.endswith('.iswb'):
            self.load_project_file(file_path)
            return

        # Figure out if the user wants to open a raster data set or a
        # spectral library.

        if file_path.endswith('.sli') or file_path.endswith('.hdr'):
            # ENVI file, possibly a spectral library.  Find out.
            try:
                # Will this work??
                library = ENVISpectralLibrary(file_path)

                # Wow it worked!  It must be a spectral library.
                self.add_spectral_library(library)
                return

            except FileNotFoundError:
                pass
            except EnviFileFormatError:
                pass

        # Either the data doesn't look like a spectral library, or loading
        # it as a spectral library didn't work.  Load it as a regular raster
        # data file.

        raster_data = self._raster_data_loader.load(file_path)
        self.add_dataset(raster_data)


    def add_dataset(self, dataset):
        '''
        Add a dataset to the application state.  The method will fire a signal
        indicating that the dataset was added.
        '''
        if not isinstance(dataset, RasterDataSet):
            raise TypeError('dataset must be a RasterDataSet')

        index = len(self._datasets)
        self._datasets.append(dataset)

        self.dataset_added.emit(index)

    def get_dataset(self, index):
        '''
        Return the dataset at the specified index.  Standard list-access options
        are supported, such as -1 to return the last dataset.
        '''
        return self._datasets[index]

    def num_datasets(self):
        ''' Return the number of datasets in the application state. '''
        return len(self._datasets)

    def get_datasets(self):
        ''' Return a copy of the list of datasets in the application state. '''
        return list(self._datasets)

    def remove_dataset(self, index):
        '''
        Remove the specified dataset from the application state.  The method
        will fire a signal indicating that the dataset was removed.
        '''
        del self._datasets[index]
        self.dataset_removed.emit(index)


    def add_spectral_library(self, library):
        '''
        Add a spectral library to the application state.  The method will fire
        a signal indicating that the spectral library was added.
        '''
        if not isinstance(library, SpectralLibrary):
            raise TypeError('library must be a SpectralLibrary')

        index = len(self._spectral_libraries)
        self._spectral_libraries.append(library)

        self.spectral_library_added.emit(index)

    def get_spectral_library(self, index):
        '''
        Return the spectral library at the specified index.  Standard
        list-access options are supported, such as -1 to return the last
        library.
        '''
        return self._spectral_libraries[index]

    def num_spectral_libraries(self):
        '''
        Return the number of spectral libraries in the application state.
        '''
        return len(self._spectral_libraries)

    def get_spectral_libraries(self):
        '''
        Return a copy of the list of spectral libraries in the application
        state.
        '''
        return list(self._spectral_libraries)

    def remove_spectral_library(self, index):
        '''
        Remove the specified spectral library from the application state.
        The method will fire a signal indicating that the spectral library
        was removed.
        '''
        del self._spectral_libraries[index]
        self.spectral_library_removed.emit(index)


    def get_config(self, option, default=None):
        '''
        Returns the value of the specified config option.  An optional default
        value may be specified.
        '''
        return self._config.get(option, default)


    def set_config(self, option, value):
        '''
        Sets the value of the specified config option.
        '''
        self._config[option] = value


    def get_color_of(self, option):
        '''
        Returns the color of the specified config option.
        '''
        return self._colors[option]

    def set_color_of(self, option, color):
        '''
        Sets the color of the specified config option.
        '''
        self._colors[option] = color

    def make_and_add_roi(self, selection):
        # Find a unique name to assign to the region of interest
        while True:
            name = f'roi_{self._next_roi_id}'
            if name not in self._regions_of_interest:
                break

            self._next_roi_id += 1

        roi = RegionOfInterest(name, selection)
        self.add_roi(roi)

    def add_roi(self, roi):
        name = roi.get_name()
        if name in self._regions_of_interest:
            raise ValueError(
                f'A region of interest named "{name}" already exists.')

        self._regions_of_interest[name] = roi
        self.roi_added.emit(roi)

    def remove_roi(self, name):
        roi = self._regions_of_interest[name]
        del self._regions_of_interest[name]

        self.roi_removed.emit(roi)

    def get_rois(self):
        return self._regions_of_interest


class DataVisualizerApp(QMainWindow):

    def __init__(self):
        '''
        Initialize the data-visualization app.  This method initializes the
        model, various views, and hooks them together with the controller code.
        '''
        super().__init__(None)
        self.setWindowTitle(self.tr('Imaging Spectroscopy Workbench'))

        # Internal state

        self._app_state = ApplicationState()

        # Application Toolbars

        self._init_menus()

        self._main_toolbar = self.addToolBar(self.tr('Main'))
        self._main_toolbar.setObjectName('main_toolbar') # Needed for UI persistence
        self._init_toolbars()

        # Status bar
        self.statusBar().showMessage(
            self.tr('Welcome to the Imaging Spectroscopy Workbench'), 30000)

        # Context pane

        self._context_pane = ContextPane(self._app_state)
        self._make_dockable_pane(self._context_pane, name='context_pane',
            title=self.tr('Context'), icon='resources/context-pane.svg',
            tooltip=self.tr('Show/hide the context pane'),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea,
            area=Qt.LeftDockWidgetArea)

        # Zoom pane

        self._zoom_pane = ZoomPane(self._app_state)
        dockable = self._make_dockable_pane(self._zoom_pane, name='zoom_pane',
            title=self.tr('Zoom'), icon='resources/zoom-pane.svg',
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
            title=self.tr('Spectrum Plot'), icon='resources/spectrum-pane.svg',
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
            title=self.tr('Dataset Info'), icon='resources/dataset-info.svg',
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
        act = add_toolbar_action(self._main_toolbar, 'resources/open-image.svg',
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
            except:
                mbox = QMessageBox(QMessageBox.Critical,
                    self.tr('Could not open file'),
                    self.tr('The file could not be opened.'),
                    QMessageBox.Ok, parent=self)

                mbox.setInformativeText(selected[0])

                # TODO(donnie):  Add exception-trace info here, using
                #     mbox.setDetailedText()

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
        self.statusBar().showMessage(msg)


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


    def _on_display_bands_change(self, index, bands, is_global):
        '''
        When the user changes the display bands used in one of the raster panes,
        the pane will fire an event that the application controller can receive,
        if other raster panes also need to be updated.
        '''
        if is_global:
            self._context_pane.set_display_bands(index, bands)
            self._main_view.set_display_bands(index, bands)
            self._zoom_pane.set_display_bands(index, bands)


    def _on_create_selection(self, selection):
        self._app_state.add_selection(selection)


    def _on_context_raster_pixel_select(self, ds_point):
        '''
        When the user clicks the mouse in the context pane, the main view is
        updated to show that location in the center of the main window.
        '''
        self._main_view.make_point_visible(ds_point.x(), ds_point.y())


    def _on_mainview_viewport_change(self, visible_area):
        '''
        When the user scrolls the viewport in the main view, the context pane
        is updated to show the visible area.
        '''
        self._context_pane.set_viewport_highlight(visible_area)


    def _on_mainview_raster_pixel_select(self, ds_point):
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

        # Update the main view and zoom pane to show the selected pixel
        self._main_view.set_pixel_highlight(ds_point, recenter=RecenterMode.NEVER)
        self._zoom_pane.set_pixel_highlight(ds_point)

        # Show spectrum at selected pixel.  Note that we use the dataset
        # selected in the main view, since the click originated from the main
        # view.
        dataset = self._main_view.get_current_dataset()
        self._spectrum_plot.set_active_spectrum(dataset, ds_point)


    def _on_stretch_changed(self, stretches: list):
        '''
        Receive stretch-change events from the Stretch Builder and propagate
        them to all raster-views.
        '''

        print(f'Contrast stretch changed to:')
        for s in stretches:
            print(f' * {s}')

        # TODO(donnie):  This is a bit ugly.  Hook event-source directly to
        #     these sinks?  Do something else?
        self._context_pane.get_rasterview().set_stretches(stretches)
        self._main_view.get_rasterview().set_stretches(stretches)
        self._zoom_pane.get_rasterview().set_stretches(stretches)


    def _on_zoom_visibility_changed(self, visible):
        if visible:
            # Zoom pane is being shown.  Show the zoom-region highlight in the
            # main view.
            visible_region = self._zoom_pane.get_rasterview().get_visible_region()
            self._main_view.set_viewport_highlight(visible_region)

        else:
            # Zoom pane is being hidden.  Remove the zoom-region highlight from
            # the main view.
            self._main_view.set_viewport_highlight(None)


    def _on_zoom_viewport_change(self, visible_area):
        '''
        When the user scrolls the viewport in the zoom pane, the main view
        is updated to show the visible area.
        '''
        if not self._zoom_pane.isVisible():
            visible_area = None

        self._main_view.set_viewport_highlight(visible_area)


    def _on_zoom_raster_pixel_select(self, ds_point):
        '''
        When the user clicks in the zoom pane, the following things happen:
        *   A selection reticle is shown around the pixel.
        *   The spectrum of the pixel is shown in the spectrum-plot view.

        These operations occur whether the above panes are visible or not, so
        that if they were hidden and are then shown, they will still contain the
        relevant information.
        '''
        if self._app_state.num_datasets() == 0:
            return

        # Update the zoom pane to show the selected pixel
        self._main_view.set_pixel_highlight(ds_point, recenter=RecenterMode.IF_NOT_VISIBLE)
        self._zoom_pane.set_pixel_highlight(ds_point, recenter=RecenterMode.NEVER)

        # Show spectrum at selected pixel.  Note that we use the dataset
        # selected in the zoom pane, since the click originated from the zoom
        # pane.
        dataset = self._zoom_pane.get_current_dataset()
        self._spectrum_plot.set_active_spectrum(dataset, ds_point)
