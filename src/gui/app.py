import os, pathlib, sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .dockable import DockablePane

from .overview_pane import OverviewPane
from .zoom_pane import ZoomPane

from .main_view import MainViewWidget

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

    # Signal:  the specified view attribute's value changed
    view_attr_changed = Signal(str)


    def __init__(self):
        super().__init__()
        self._datasets = []

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


    def set_view_attribute(self, attr_name, value):
        self._view_attributes[attr_name] = value
        self.view_attr_changed.emit(attr_name)

    def get_view_attribute(self, attr_name, default=None):
        return self._view_attributes.get(attr_name, default)


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

        self.init_menus()

        self.main_toolbar = self.addToolBar(self.tr('Main'))

        # Overview pane

        self.overview_pane = OverviewPane(self._app_state, parent=self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.overview_pane)
        act = self.overview_pane.toggleViewAction()
        self.main_toolbar.addAction(act)
        self.view_menu.addAction(act)

        # Zoom pane

        self.zoom_pane = ZoomPane(self._app_state, parent=self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.zoom_pane)
        act = self.zoom_pane.toggleViewAction()
        self.main_toolbar.addAction(act)
        self.view_menu.addAction(act)

        # Main raster-view

        self.main_view = MainViewWidget(self._app_state)
        self.setCentralWidget(self.main_view)

        # self.main_view.rasterview().viewport_change.connect(self.mainview_viewport_change)
        # self.main_view.rasterview().mouse_click.connect(self.mainview_mouse_click)

        # Spectrum plot

        self._spectrum_plot = SpectrumPlot(self._app_state)
        self._make_dockable_pane(self._spectrum_plot, name='spectrum_plot',
            title=self.tr('Spectrum Plot'), icon='resources/spectrum-pane.svg',
            tooltip=self.tr('Show/hide the spectrum pane'),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea,
            area=Qt.RightDockWidgetArea)

        # Dataset Information Window

        # TODO(donnie):  Why do we need a scroll area here?  The QTreeWidget is
        #     a scroll-area too!!
        self._dataset_info = DatasetInfoView(self._app_state)
        # scroll_area = QScrollArea()
        # scroll_area.setWidget(self.info_view)
        # scroll_area.setWidgetResizable(True)
        self._make_dockable_pane(self._dataset_info, name='dataset_info',
            title=self.tr('Dataset Info'), icon='resources/dataset-info.svg',
            tooltip=self.tr('Show/hide dataset information'),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea,
            area=Qt.LeftDockWidgetArea)


    def init_menus(self):
        # File menu

        self.file_menu = self.menuBar().addMenu(self.tr('&File'))

        act = QAction(self.tr('&Open...'), self)
        act.setShortcuts(QKeySequence.Open)
        act.setStatusTip(self.tr('Open an existing project or file'))
        act.triggered.connect(self.show_open_file_dialog)

        self.file_menu.addAction(act)

        # View menu

        self.view_menu = self.menuBar().addMenu(self.tr('&View'))

        # Other menus?

        # self.window_menu = self.menuBar().addMenu(self.tr('&Window'))
        # self.help_menu = self.menuBar().addMenu(self.tr('&Help'))


    def _make_dockable_pane(self, widget, name, title, icon, tooltip,
                            allowed_areas, area):

        dockable = DockablePane(widget, name, title, self._app_state,
                                icon=icon, tooltip=tooltip, parent=self)

        dockable.setAllowedAreas(allowed_areas)
        self.addDockWidget(area, dockable)

        act = dockable.toggleViewAction()
        self.view_menu.addAction(act)

        act = dockable.toggleViewAction()
        act.setIcon(dockable.get_icon())
        act.setToolTip(dockable.get_tooltip())
        self.main_toolbar.addAction(act)

        return dockable


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


    def summaryview_mouse_click(self, ds_point, mouse_event):
        self.main_view.rasterview().make_point_visible(ds_point.x(), ds_point.y())


    def mainview_viewport_change(self, visible_area):
        self._app_state.set_image_visible_area(visible_area)


    def mainview_mouse_click(self, ds_point, mouse_event):
        if self._model.num_datasets() == 0:
            return

        # Update the detail view
        self.detail_view.rasterview().make_point_visible(ds_point.x(), ds_point.y())
        self.detail_view.rasterview().set_current_pixel(ds_point)

        # Show spectrum at selected pixel
        dataset = self.main_view.get_current_dataset()
        spectrum = dataset.get_all_bands_at(ds_point.x(), ds_point.y(), filter_bad_values=True)
        self._spectrum_plot.set_spectrum(spectrum, dataset)


    def detailview_mouse_click(self, ds_point, mouse_event):
        if self._model.num_datasets() == 0:
            return

        # Draw selected pixel in detail view
        self.detail_view.rasterview().set_current_pixel(ds_point)

        # Show spectrum at selected pixel
        dataset = self.detail_view.get_current_dataset()
        spectrum = dataset.get_all_bands_at(ds_point.x(), ds_point.y(), filter_bad_values=True)
        self._spectrum_plot.set_spectrum(spectrum, dataset)
