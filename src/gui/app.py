import os, pathlib, sys

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

from .util import *

from raster.dataset import *


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

    # TODO(donnie):  Signals for config changes and color changes!

    def __init__(self):
        super().__init__()

        # All datasets loaded in the application.
        self._datasets = []

        # Configuration options.
        self._config = {
            'pixel-reticle-type' : PixelReticleType.SMALL_CROSS,
        }

        # Colors used for various purposes.
        self._colors = {
            'viewport-highlight' : Qt.yellow,
            'pixel-highlight' : Qt.red,
        }

        self._view_attributes = {}


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


class DataVisualizerApp(QMainWindow):

    def __init__(self, loader):
        '''
        Initialize the data-visualization app.  This method initializes the
        model, various views, and hooks them together with the controller code.

        The raster data loader to use is passed in as an argument to this class.
        '''
        super().__init__(None)
        self.setWindowTitle(self.tr('Imaging Spectroscopy Workbench'))

        if loader is None:
            raise ValueError('data loader must be specified')

        # Internal state

        self._app_state = ApplicationState()

        self.current_dir = os.getcwd()

        self.loader = loader

        # Application Toolbars

        self._init_menus()

        self._main_toolbar = self.addToolBar(self.tr('Main'))
        self._init_toolbars()

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
        self._image_toolbar = self.addToolBar(self._main_view.get_toolbar())

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

        self._context_pane.raster_pixel_select.connect(self._on_context_raster_pixel_select)
        self._context_pane.display_bands_change.connect(self._on_display_bands_change)

        self._main_view.viewport_change.connect(self._on_mainview_viewport_change)
        self._main_view.raster_pixel_select.connect(self._on_mainview_raster_pixel_select)
        self._main_view.display_bands_change.connect(self._on_display_bands_change)

        self._zoom_pane.viewport_change.connect(self._on_zoom_viewport_change)
        self._zoom_pane.raster_pixel_select.connect(self._on_zoom_raster_pixel_select)
        self._zoom_pane.visibility_change.connect(self._on_zoom_visibility_changed)
        self._zoom_pane.display_bands_change.connect(self._on_display_bands_change)


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
            self.tr('ENVI files (*.hdr *.img)'),
            self.tr('TIFF files (*.tiff *.tif *.tfw)'),
            self.tr('PDS files (*.PDS *.IMG)'),
            self.tr('All files (*)'),
        ]

        selected = QFileDialog.getOpenFileName(self,
            self.tr("Open Spectal Data File"),
            self.current_dir, ';;'.join(supported_formats))
        # print(selected)

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


    def open_file(self, file_path):
        # Remember the directory of the selected file, for next file-open
        self.current_dir = os.path.dirname(file_path)

        # Try to open the specified data-set

        raster_data = self.loader.load(file_path)
        self._app_state.add_dataset(raster_data)


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
        spectrum = dataset.get_all_bands_at(ds_point.x(), ds_point.y())
        self._spectrum_plot.set_spectrum(spectrum, dataset)


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
        spectrum = dataset.get_all_bands_at(ds_point.x(), ds_point.y())
        self._spectrum_plot.set_spectrum(spectrum, dataset)
