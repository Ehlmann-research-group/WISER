import os, pathlib, sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *




class DataVisualizerApp(QMainWindow):

    def __init__(self):
        super().__init__(None)
        self.setWindowTitle('Data Visualizer App')

        self.current_dir = os.getcwd()

    def init_menus(self):
        print('Initializing menus.')

        self.file_menu = self.menuBar().addMenu(self.tr('&File'))
        # self.window_menu = self.menuBar().addMenu(self.tr('&Window'))
        # self.help_menu = self.menuBar().addMenu(self.tr('&Help'))

        act = QAction(self.tr('&Open...'), self)
        act.setShortcuts(QKeySequence.Open)
        act.setStatusTip(self.tr("Open an existing project or file"))
        act.triggered.connect(self.open_file)

        self.file_menu.addAction(act)


    def open_file(self, evt):
        print('Open file...')

        supported_formats = [
            self.tr('ENVI files (*.hdr *.img)'),
            self.tr('TIFF files (*.tiff *.tif *.tfw)'),
            self.tr('PDS files (*.PDS *.IMG)'),
            self.tr('All files (*)'),
        ]

        selected = QFileDialog.getOpenFileName(self, self.tr("Open Spectal Data File"),
            self.current_dir, ';;'.join(supported_formats))
        print(selected)

        if len(selected[0]) > 0:
            # Remember the directory of the selected file, for next file-open
            self.current_dir = os.path.dirname(selected[0])


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ui = DataVisualizerApp()
    ui.init_menus()
    ui.show()

    sys.exit(app.exec_())
