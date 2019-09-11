import sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from PySide2.QtCharts import QtCharts


class SpectrumPlot(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.chart = QtCharts.QChart()
        # TODO(donnie):  Chart animations get pretty annoying, should leave this
        #     OFF.
        # self.chart.setAnimationOptions(QtCharts.QChart.AllAnimations)
        self.chart.legend().setVisible(False)

        '''
        self.x_axis = QtCharts.QValueAxis()
        self.x_axis.setTickInterval(100)

        self.y_axis = QtCharts.QValueAxis()
        self.y_axis.setRange(0, 1)
        self.y_axis.setTickInterval(0.1)

        self.chart.setAxisX(self.x_axis)
        self.chart.setAxisY(self.y_axis)
        '''

        self.chart_view = QtCharts.QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addWidget(self.chart_view)
        self.setLayout(layout)

        self.spectralSeries = []

    def clear(self):
        self.spectralSeries.clear()
        self.chart.removeAllSeries()

    def add_spectrum(self, spectrum):
        series = QtCharts.QLineSeries()

        # TODO(donnie):  Assert that band_info and spectrum have the same sizes
        # TODO(donnie):  How to handle missing wavelength info?
        # TODO(donnie):  How to handle bad band info?

        for i in range(len(spectrum)):
            series.append(i, spectrum[i])

        self.spectralSeries.append(series)

        self.chart.addSeries(series)

    def set_spectrum(self, spectrum):
        self.clear()
        self.add_spectrum(spectrum)
