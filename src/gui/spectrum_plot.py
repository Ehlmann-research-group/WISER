from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import matplotlib
matplotlib.use('Qt5Agg')
# TODO(donnie):  Seems to generate errors:
# matplotlib.rcParams['backend.qt5'] = 'PySide2'

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvas



class SpectrumPlot(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # self._model.dataset_added.connect(self.add_dataset)
        # self._model.dataset_removed.connect(self.remove_dataset)

        # Set up Matplotlib

        # plt.ion()   # Turn on interactive plotting

        self.figure, self.axes = plt.subplots()
        self.figure_canvas = FigureCanvas(self.figure)

        # self.axes.set_autoscalex_on(True)
        # self.axes.set_autoscaley_on(False)
        # self.axes.set_ylim((0, 1))

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addWidget(self.figure_canvas)
        self.setLayout(layout)

        self.spectral_series = []


    def sizeHint(self):
        ''' The default size of the spectrum-plot widget is 200x200. '''
        return QSize(200, 200)


    ''' # TODO(donnie):  What do we need to have for these operations?
    def add_dataset(self, index):
        dataset = self._model.get_dataset(index)
        file_path = dataset.get_filepath()

        self._cbox_dataset.insertItem(index, os.path.basename(file_path))

        if self._model.num_datasets() == 1:
            # We finally have a dataset!
            self._dataset_index = 0
            self.update_image()


    def remove_dataset(self, index):
        self._cbox_dataset.removeItem(index)

        num = self._model.num_datasets()

        if num == 0 or self._dataset_index == index:
            self._dataset_index = min(self._dataset_index, num - 1)
            if self._dataset_index == -1:
                self._dataset_index = None

            self.update_image()
    '''


    def clear(self):
        self.spectral_series.clear()
        self.axes.clear()
        self.figure_canvas.draw()


    def add_spectrum(self, spectrum, band_info=None):
        # TODO(donnie):  Assert that band_info and spectrum have the same sizes
        # TODO(donnie):  How to handle missing wavelength info?
        # TODO(donnie):  How to handle bad band info?

        self.spectral_series.append(spectrum)

        self.axes.plot(spectrum, linewidth=0.5, scalex=True, scaley=False)
        self.figure_canvas.draw()

    def set_spectrum(self, spectrum, band_info=None):
        self.clear()
        self.add_spectrum(spectrum)
