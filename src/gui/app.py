import os, pathlib, sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .model import AppModel

from .summary_view import SummaryViewWidget
from .main_view import MainViewWidget
from .detail_view import DetailViewWidget
from .spectrum_plot import SpectrumPlot

from .util import *

# TODO(donnie):  This class shouldn't know about the GDAL data loader
from raster.dataset import *
from raster.gdal_dataset import GDALRasterDataLoader


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

        self._model = AppModel()

        self.current_dir = os.getcwd()

        self.loader = loader

        # Toolbars

        # self.init_toolbar()

        # Summary raster-view

        self.summary_view = SummaryViewWidget(self._model)
        # self.summary_view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        dockable = make_dockable(self.summary_view, self.tr('Summary'), self)
        # dockable.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        # dockable.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dockable.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        self.addDockWidget(Qt.LeftDockWidgetArea, dockable)

        self.summary_view.rasterview().mouse_click.connect(self.summaryview_mouse_click)

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

        act = QAction(self.tr('Show &Context Window'), self)
        act.setStatusTip(self.tr('Show the overview/context image window'))
        # act.triggered.connect(self.show_context_window)

        act = QAction(self.tr('Show &Zoom Window'), self)
        act.setStatusTip(self.tr('Show the zoom image window'))
        # act.triggered.connect(self.show_zoom_window)

        act = QAction(self.tr('Show &Spectrum Window'), self)
        act.setStatusTip(self.tr('Show the spectrum-plot window'))
        # act.triggered.connect(self.show_spectrum_window)

        # Other menus?

        # self.window_menu = self.menuBar().addMenu(self.tr('&Window'))
        # self.help_menu = self.menuBar().addMenu(self.tr('&Help'))


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
        # print(f'Viewport changed:  {visible_area}')
        self.summary_view.rasterview().set_visible_area(visible_area)

        center = visible_area.center()
        self.detail_view.rasterview().make_point_visible(center.x(), center.y())

    def mainview_mouse_click(self, ds_point, mouse_event):
        self.detail_view.rasterview().make_point_visible(ds_point.x(), ds_point.y())

        # Show spectrum at selected pixel
        dataset = self.main_view.get_current_dataset()
        spectrum = dataset.get_all_bands_at(ds_point.x(), ds_point.y(), filter_bad_values=True)
        self.spectrum_plot.set_spectrum(spectrum)


    def detailview_viewport_change(self, visible_area):
        self.main_view.rasterview().set_visible_area(visible_area)

    def detailview_mouse_click(self, ds_point, mouse_event):
        # Show spectrum at selected pixel
        dataset = self.detail_view.get_current_dataset()
        spectrum = dataset.get_all_bands_at(ds_point.x(), ds_point.y(), filter_bad_values=True)
        self.spectrum_plot.set_spectrum(spectrum)
