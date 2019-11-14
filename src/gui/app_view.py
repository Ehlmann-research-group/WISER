import os, pathlib, sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .model import AppModel

from .summary_view import SummaryViewWidget
from .main_view import MainViewWidget
from .detail_view import DetailViewWidget

from .util import *

# TODO(donnie):  This class shouldn't know about the GDAL data loader
from raster.dataset import *
from raster.gdal_dataset import GDALRasterDataLoader


class DataVisualizerApp(QMainWindow):

    def __init__(self):
        super().__init__(None)
        self.setWindowTitle(self.tr('Imaging Spectroscopy Workbench'))

        # Internal state

        self._model = AppModel()
        self.current_dir = os.getcwd()

        # TODO(donnie):  This class shouldn't know about the GDAL data loader
        self.loader = GDALRasterDataLoader()

        # Toolbars

        # self.init_toolbar()

        # Summary raster-view

        self.summary_view = SummaryViewWidget(self._model)

        dockable = QDockWidget(self.tr('Summary'), parent=self)
        dockable.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dockable.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dockable.setWidget(self.summary_view)

        self.addDockWidget(Qt.LeftDockWidgetArea, dockable)

        # Detail raster-view

        self.detail_view = DetailViewWidget(self._model)

        dockable = QDockWidget(self.tr('Detail'), parent=self)
        dockable.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dockable.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dockable.setWidget(self.detail_view)

        self.addDockWidget(Qt.RightDockWidgetArea, dockable)

        # Main raster-view

        self.main_view = MainViewWidget(self._model)
        self.setCentralWidget(self.main_view)


    def init_menus(self):
        self.file_menu = self.menuBar().addMenu(self.tr('&File'))
        # self.window_menu = self.menuBar().addMenu(self.tr('&Window'))
        # self.help_menu = self.menuBar().addMenu(self.tr('&Help'))

        act = QAction(self.tr('&Open...'), self)
        act.setShortcuts(QKeySequence.Open)
        act.setStatusTip(self.tr("Open an existing project or file"))
        act.triggered.connect(self.show_open_file_dialog)

        self.file_menu.addAction(act)


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
