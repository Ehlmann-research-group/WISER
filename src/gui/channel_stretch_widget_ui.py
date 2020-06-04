# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'channel_stretch_widget.ui',
# licensing of 'channel_stretch_widget.ui' applies.
#
# Created: Fri May  1 14:10:04 2020
#      by: pyside2-uic  running on PySide2 5.13.2
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_ChannelStretchWidget(object):
    def setupUi(self, ChannelStretchWidget):
        ChannelStretchWidget.setObjectName("ChannelStretchWidget")
        ChannelStretchWidget.resize(570, 240)
        ChannelStretchWidget.setMinimumSize(QtCore.QSize(570, 240))
        self.groupbox_channel = QtWidgets.QGroupBox(ChannelStretchWidget)
        self.groupbox_channel.setGeometry(QtCore.QRect(0, 0, 571, 241))
        self.groupbox_channel.setObjectName("groupbox_channel")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupbox_channel)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 30, 551, 201))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.grid_layout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.grid_layout.setSpacing(4)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setObjectName("grid_layout")
        self.lineedit_stretch_high = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineedit_stretch_high.setObjectName("lineedit_stretch_high")
        self.grid_layout.addWidget(self.lineedit_stretch_high, 6, 1, 1, 1)
        self.lineedit_stretch_low = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineedit_stretch_low.setObjectName("lineedit_stretch_low")
        self.grid_layout.addWidget(self.lineedit_stretch_low, 5, 1, 1, 1)
        self.button_apply_bounds = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_apply_bounds.setObjectName("button_apply_bounds")
        self.grid_layout.addWidget(self.button_apply_bounds, 2, 1, 1, 1)
        self.slider_stretch_low = QtWidgets.QSlider(self.gridLayoutWidget)
        self.slider_stretch_low.setMaximum(100)
        self.slider_stretch_low.setOrientation(QtCore.Qt.Horizontal)
        self.slider_stretch_low.setInvertedAppearance(False)
        self.slider_stretch_low.setObjectName("slider_stretch_low")
        self.grid_layout.addWidget(self.slider_stretch_low, 5, 2, 1, 1)
        self.label_stretch_high = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_stretch_high.setObjectName("label_stretch_high")
        self.grid_layout.addWidget(self.label_stretch_high, 6, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.grid_layout.addItem(spacerItem, 4, 0, 1, 2)
        self.histogram_widget = QtWidgets.QWidget(self.gridLayoutWidget)
        self.histogram_widget.setObjectName("histogram_widget")
        self.grid_layout.addWidget(self.histogram_widget, 0, 2, 5, 1)
        self.slider_stretch_high = QtWidgets.QSlider(self.gridLayoutWidget)
        self.slider_stretch_high.setOrientation(QtCore.Qt.Horizontal)
        self.slider_stretch_high.setObjectName("slider_stretch_high")
        self.grid_layout.addWidget(self.slider_stretch_high, 6, 2, 1, 1)
        self.button_reset_bounds = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_reset_bounds.setObjectName("button_reset_bounds")
        self.grid_layout.addWidget(self.button_reset_bounds, 3, 1, 1, 1)
        self.lineedit_max_bound = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineedit_max_bound.setObjectName("lineedit_max_bound")
        self.grid_layout.addWidget(self.lineedit_max_bound, 1, 1, 1, 1)
        self.label_stretch_low = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_stretch_low.setObjectName("label_stretch_low")
        self.grid_layout.addWidget(self.label_stretch_low, 5, 0, 1, 1)
        self.lineedit_min_bound = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineedit_min_bound.setObjectName("lineedit_min_bound")
        self.grid_layout.addWidget(self.lineedit_min_bound, 0, 1, 1, 1)
        self.label_max_bound = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_max_bound.setObjectName("label_max_bound")
        self.grid_layout.addWidget(self.label_max_bound, 1, 0, 1, 1)
        self.label_min_bound = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_min_bound.setObjectName("label_min_bound")
        self.grid_layout.addWidget(self.label_min_bound, 0, 0, 1, 1)
        self.grid_layout.setColumnStretch(2, 10)
        self.grid_layout.setRowStretch(4, 10)
        self.label_max_bound.setBuddy(self.lineedit_max_bound)
        self.label_min_bound.setBuddy(self.lineedit_min_bound)

        self.retranslateUi(ChannelStretchWidget)
        QtCore.QMetaObject.connectSlotsByName(ChannelStretchWidget)

    def retranslateUi(self, ChannelStretchWidget):
        ChannelStretchWidget.setWindowTitle(QtWidgets.QApplication.translate("ChannelStretchWidget", "Channel Stretch Widget", None, -1))
        self.groupbox_channel.setTitle(QtWidgets.QApplication.translate("ChannelStretchWidget", "Channel Name", None, -1))
        self.button_apply_bounds.setToolTip(QtWidgets.QApplication.translate("ChannelStretchWidget", "Filter band data using min/max bounds before computing histogram", None, -1))
        self.button_apply_bounds.setText(QtWidgets.QApplication.translate("ChannelStretchWidget", "Apply", None, -1))
        self.slider_stretch_low.setToolTip(QtWidgets.QApplication.translate("ChannelStretchWidget", "Drag to adjust low boundary of stretch", None, -1))
        self.label_stretch_high.setText(QtWidgets.QApplication.translate("ChannelStretchWidget", "Stretch High", None, -1))
        self.slider_stretch_high.setToolTip(QtWidgets.QApplication.translate("ChannelStretchWidget", "Drag to adjust high boundary of stretch", None, -1))
        self.button_reset_bounds.setToolTip(QtWidgets.QApplication.translate("ChannelStretchWidget", "Reset min/max bounds to min/max values from band data", None, -1))
        self.button_reset_bounds.setText(QtWidgets.QApplication.translate("ChannelStretchWidget", "Reset", None, -1))
        self.label_stretch_low.setText(QtWidgets.QApplication.translate("ChannelStretchWidget", "Stretch Low", None, -1))
        self.label_max_bound.setText(QtWidgets.QApplication.translate("ChannelStretchWidget", "Maximum", None, -1))
        self.label_min_bound.setText(QtWidgets.QApplication.translate("ChannelStretchWidget", "Minimum", None, -1))

