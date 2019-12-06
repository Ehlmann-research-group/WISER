import os, pathlib, sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .overview_pane import OverviewPane
from .main_view import MainViewWidget
from .detail_view import DetailViewWidget
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

    # Signal:  the currently selected pixel changed
    current_pixel_changed = Signal(QPoint)

    # Signal:  the main image window's visible area changed
    image_visible_area_changed = Signal(QRect)

    def __init__(self):
        super().__init__()
        self._datasets = []

        self._image_visible_area = None

        self._current_pixel_coord = None


    def get_model(self):
        # TODO(donnie):  FIX THIS HIDEOUSNESS
        return self


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


    def set_image_visible_area(self, visible_area):
        self._image_visible_area = visible_area
        self.image_visible_area_changed.emit(visible_area)

    def get_image_visible_area(self):
        return self._image_visible_area


    def set_current_pixel(self, coord):
        if not isinstance(dataset, QPoint):
            raise TypeError('coord must be a QPoint')

        self._current_pixel_coord = coord

        self.current_pixel_changed.emit(coord)

    def get_current_pixel(self):
        return self._current_pixel_coord



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
        self._model = self._app_state

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

        # Detail raster-view

        self.detail_view = DetailViewWidget(self._model)
        # self.detail_view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        dockable = make_dockable(self.detail_view, self.tr('Detail'), self)
        dockable.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        # dockable.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dockable.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        self.addDockWidget(Qt.RightDockWidgetArea, dockable)

        self.detail_view.rasterview().viewport_change.connect(self.detailview_viewport_change)
        self.detail_view.rasterview().mouse_click.connect(self.detailview_mouse_click)

        # Main raster-view

        self.main_view = MainViewWidget(self._model)
        self.setCentralWidget(self.main_view)

        self.main_view.rasterview().viewport_change.connect(self.mainview_viewport_change)
        self.main_view.rasterview().mouse_click.connect(self.mainview_mouse_click)

        # Spectrum plot

        self.spectrum_plot = SpectrumPlot()
        dockable = make_dockable(self.spectrum_plot, self.tr('Spectral Plot'), self)
        dockable.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dockable)

        # Dataset Information Window

        # TODO(donnie):  Why do we need a scroll area here?  The QTreeWidget is
        #     a scroll-area too!!
        self.info_view = DatasetInfoView(self._model)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.info_view)
        scroll_area.setWidgetResizable(True)
        dockable = make_dockable(scroll_area, self.tr('Dataset Information'), self)
        dockable.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dockable)


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

    def init_toolbar(self):
        # Main application toolbar
        toolbar = QToolBar(self.tr('ISWB'), parent=self)

        self._cbox_dataset = QComboBox(parent=self)
        self._cbox_dataset.setEditable(False)
        self._cbox_dataset.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        toolbar.addWidget(self._cbox_dataset)

        self._cbox_dataset.activated.connect(self.change_dataset)

        self._act_fit_to_window = toolbar.addAction(
            QIcon('resources/zoom-to-fit.svg'),
            self.tr('Fit image to window'))
        self._act_fit_to_window.setCheckable(True)
        self._act_fit_to_window.setChecked(True)

        self._act_fit_to_window.triggered.connect(self.toggle_fit_to_window)



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
        self._model.add_dataset(raster_data)

        # Force an update in the UI
        # TODO(donnie):  This needs to migrate to the main-view class!
        self.main_view.rasterview()._emit_viewport_change()


    def summaryview_mouse_click(self, ds_point, mouse_event):
        self.main_view.rasterview().make_point_visible(ds_point.x(), ds_point.y())


    def mainview_viewport_change(self, visible_area):
        self._app_state.set_image_visible_area(visible_area)

        # TODO:  Need to figure out the appropriate zoom-pane behavior here.
        # center = visible_area.center()
        # self.detail_view.rasterview().make_point_visible(center.x(), center.y())


    def mainview_mouse_click(self, ds_point, mouse_event):
        if self._model.num_datasets() == 0:
            return

        # Update the detail view
        self.detail_view.rasterview().make_point_visible(ds_point.x(), ds_point.y())
        self.detail_view.rasterview().set_current_pixel(ds_point)

        # Show spectrum at selected pixel
        dataset = self.main_view.get_current_dataset()
        spectrum = dataset.get_all_bands_at(ds_point.x(), ds_point.y(), filter_bad_values=True)
        self.spectrum_plot.set_spectrum(spectrum, dataset)


    def detailview_viewport_change(self, visible_area):
        self.main_view.rasterview().set_visible_area(visible_area)

    def detailview_mouse_click(self, ds_point, mouse_event):
        if self._model.num_datasets() == 0:
            return

        # Draw selected pixel in detail view
        self.detail_view.rasterview().set_current_pixel(ds_point)

        # Show spectrum at selected pixel
        dataset = self.detail_view.get_current_dataset()
        spectrum = dataset.get_all_bands_at(ds_point.x(), ds_point.y(), filter_bad_values=True)
        self.spectrum_plot.set_spectrum(spectrum, dataset)
