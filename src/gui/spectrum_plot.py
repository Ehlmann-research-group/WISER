import sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.rcParams['backend.qt5'] = 'PySide2'

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvas


class SpectrumPlot(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # plt.ion()   # Turn on interactive plotting

        self.figure, self.axes = plt.subplots()
        self.figure_canvas = FigureCanvas(self.figure)

        self.axes.set_autoscalex_on(True)
        self.axes.set_autoscaley_on(False)
        self.axes.set_ylim((0, 1))

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addWidget(self.figure_canvas)
        self.setLayout(layout)

        self.spectral_series = []

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
