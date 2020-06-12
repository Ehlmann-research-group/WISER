# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'spectrum_plot_config.ui',
# licensing of 'spectrum_plot_config.ui' applies.
#
# Created: Sun May 10 09:32:38 2020
#      by: pyside2-uic  running on PySide2 5.13.2
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_SpectrumPlotConfig(object):
    def setupUi(self, SpectrumPlotConfig):
        SpectrumPlotConfig.setObjectName("SpectrumPlotConfig")
        SpectrumPlotConfig.resize(357, 215)
        self.buttonBox = QtWidgets.QDialogButtonBox(SpectrumPlotConfig)
        self.buttonBox.setGeometry(QtCore.QRect(10, 180, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayoutWidget = QtWidgets.QWidget(SpectrumPlotConfig)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 9, 341, 161))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.grid_layout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setObjectName("grid_layout")
        self.lineedit_area_avg_y = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineedit_area_avg_y.setObjectName("lineedit_area_avg_y")
        self.grid_layout.addWidget(self.lineedit_area_avg_y, 2, 2, 1, 1)
        self.label_placeholder = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_placeholder.setObjectName("label_placeholder")
        self.grid_layout.addWidget(self.label_placeholder, 3, 0, 1, 3)
        self.lineedit_area_avg_x = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineedit_area_avg_x.setObjectName("lineedit_area_avg_x")
        self.grid_layout.addWidget(self.lineedit_area_avg_x, 1, 2, 1, 1)
        self.label_area_avg_y = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_area_avg_y.setObjectName("label_area_avg_y")
        self.grid_layout.addWidget(self.label_area_avg_y, 2, 1, 1, 1)
        self.combobox_avg_mode = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.combobox_avg_mode.setObjectName("combobox_avg_mode")
        self.grid_layout.addWidget(self.combobox_avg_mode, 0, 1, 1, 2)
        self.label_area_avg = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_area_avg.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_area_avg.setObjectName("label_area_avg")
        self.grid_layout.addWidget(self.label_area_avg, 1, 0, 1, 1)
        self.label_area_avg_x = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_area_avg_x.setObjectName("label_area_avg_x")
        self.grid_layout.addWidget(self.label_area_avg_x, 1, 1, 1, 1)
        self.label_avg_mode = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_avg_mode.setObjectName("label_avg_mode")
        self.grid_layout.addWidget(self.label_avg_mode, 0, 0, 1, 1)
        self.grid_layout.setColumnStretch(0, 1)
        self.label_area_avg_y.setBuddy(self.lineedit_area_avg_y)
        self.label_area_avg_x.setBuddy(self.lineedit_area_avg_x)
        self.label_avg_mode.setBuddy(self.combobox_avg_mode)

        self.retranslateUi(SpectrumPlotConfig)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), SpectrumPlotConfig.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), SpectrumPlotConfig.reject)
        QtCore.QMetaObject.connectSlotsByName(SpectrumPlotConfig)
        SpectrumPlotConfig.setTabOrder(self.combobox_avg_mode, self.lineedit_area_avg_x)
        SpectrumPlotConfig.setTabOrder(self.lineedit_area_avg_x, self.lineedit_area_avg_y)

    def retranslateUi(self, SpectrumPlotConfig):
        SpectrumPlotConfig.setWindowTitle(QtWidgets.QApplication.translate("SpectrumPlotConfig", "Spectrum Plot Configuration", None, -1))
        self.label_placeholder.setText(QtWidgets.QApplication.translate("SpectrumPlotConfig", "TODO:  Spectrum plot configuration (tick marks, etc.)", None, -1))
        self.label_area_avg_y.setText(QtWidgets.QApplication.translate("SpectrumPlotConfig", "Y", None, -1))
        self.label_area_avg.setText(QtWidgets.QApplication.translate("SpectrumPlotConfig", "Default area-average size", None, -1))
        self.label_area_avg_x.setText(QtWidgets.QApplication.translate("SpectrumPlotConfig", "X", None, -1))
        self.label_avg_mode.setText(QtWidgets.QApplication.translate("SpectrumPlotConfig", "Default area-average mode", None, -1))

