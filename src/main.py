import sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from gui.mainview import DataVisualizerApp


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ui = DataVisualizerApp()
    ui.init_menus()

    # Set the initial window size to be 70% of the screen size.
    screen_size = app.screens()[0].size()
    ui.resize(screen_size * 0.7)

    ui.show()

    sys.exit(app.exec_())
