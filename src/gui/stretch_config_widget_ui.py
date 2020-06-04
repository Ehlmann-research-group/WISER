# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'stretch_config_widget.ui',
# licensing of 'stretch_config_widget.ui' applies.
#
# Created: Fri May  8 13:08:53 2020
#      by: pyside2-uic  running on PySide2 5.13.2
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_StretchConfigWidget(object):
    def setupUi(self, StretchConfigWidget):
        StretchConfigWidget.setObjectName("StretchConfigWidget")
        StretchConfigWidget.resize(570, 160)
        StretchConfigWidget.setMinimumSize(QtCore.QSize(570, 160))
        self.gridLayoutWidget = QtWidgets.QWidget(StretchConfigWidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 571, 161))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.grid_layout_widget = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.grid_layout_widget.setContentsMargins(0, 0, 0, 0)
        self.grid_layout_widget.setObjectName("grid_layout_widget")
        self.groupbox_stretch = QtWidgets.QGroupBox(self.gridLayoutWidget)
        self.groupbox_stretch.setObjectName("groupbox_stretch")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.groupbox_stretch)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 30, 211, 121))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.grid_layout_stretch = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.grid_layout_stretch.setSpacing(4)
        self.grid_layout_stretch.setContentsMargins(0, 0, 0, 0)
        self.grid_layout_stretch.setObjectName("grid_layout_stretch")
        self.button_linear_2_5 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.button_linear_2_5.setObjectName("button_linear_2_5")
        self.grid_layout_stretch.addWidget(self.button_linear_2_5, 2, 1, 1, 1)
        self.rb_stretch_linear = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.rb_stretch_linear.setObjectName("rb_stretch_linear")
        self.grid_layout_stretch.addWidget(self.rb_stretch_linear, 1, 0, 1, 3)
        self.rb_stretch_equalize = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.rb_stretch_equalize.setObjectName("rb_stretch_equalize")
        self.grid_layout_stretch.addWidget(self.rb_stretch_equalize, 4, 0, 1, 3)
        self.button_linear_5_0 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.button_linear_5_0.setObjectName("button_linear_5_0")
        self.grid_layout_stretch.addWidget(self.button_linear_5_0, 3, 1, 1, 1)
        self.rb_stretch_none = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.rb_stretch_none.setObjectName("rb_stretch_none")
        self.grid_layout_stretch.addWidget(self.rb_stretch_none, 0, 0, 1, 3)
        self.grid_layout_stretch.setColumnMinimumWidth(0, 10)
        self.grid_layout_widget.addWidget(self.groupbox_stretch, 0, 1, 1, 1)
        self.groupbox_conditioner = QtWidgets.QGroupBox(self.gridLayoutWidget)
        self.groupbox_conditioner.setObjectName("groupbox_conditioner")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.groupbox_conditioner)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(10, 30, 171, 71))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.grid_layout_cond = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.grid_layout_cond.setSpacing(4)
        self.grid_layout_cond.setContentsMargins(0, 0, 0, 0)
        self.grid_layout_cond.setObjectName("grid_layout_cond")
        self.rb_cond_log = QtWidgets.QRadioButton(self.gridLayoutWidget_3)
        self.rb_cond_log.setObjectName("rb_cond_log")
        self.grid_layout_cond.addWidget(self.rb_cond_log, 2, 0, 1, 1)
        self.rb_cond_sqrt = QtWidgets.QRadioButton(self.gridLayoutWidget_3)
        self.rb_cond_sqrt.setObjectName("rb_cond_sqrt")
        self.grid_layout_cond.addWidget(self.rb_cond_sqrt, 1, 0, 1, 1)
        self.rb_cond_none = QtWidgets.QRadioButton(self.gridLayoutWidget_3)
        self.rb_cond_none.setObjectName("rb_cond_none")
        self.grid_layout_cond.addWidget(self.rb_cond_none, 0, 0, 1, 1)
        self.grid_layout_widget.addWidget(self.groupbox_conditioner, 0, 2, 1, 1)

        self.retranslateUi(StretchConfigWidget)
        QtCore.QMetaObject.connectSlotsByName(StretchConfigWidget)

    def retranslateUi(self, StretchConfigWidget):
        StretchConfigWidget.setWindowTitle(QtWidgets.QApplication.translate("StretchConfigWidget", "Form", None, -1))
        self.groupbox_stretch.setTitle(QtWidgets.QApplication.translate("StretchConfigWidget", "Stretch", None, -1))
        self.button_linear_2_5.setToolTip(QtWidgets.QApplication.translate("StretchConfigWidget", "Apply a 2.5% linear stretch to all channels", None, -1))
        self.button_linear_2_5.setText(QtWidgets.QApplication.translate("StretchConfigWidget", "2.5% linear", None, -1))
        self.rb_stretch_linear.setToolTip(QtWidgets.QApplication.translate("StretchConfigWidget", "Apply a linear stretch", None, -1))
        self.rb_stretch_linear.setText(QtWidgets.QApplication.translate("StretchConfigWidget", "Linear Stretch", None, -1))
        self.rb_stretch_equalize.setToolTip(QtWidgets.QApplication.translate("StretchConfigWidget", "Apply an equalization stretch", None, -1))
        self.rb_stretch_equalize.setText(QtWidgets.QApplication.translate("StretchConfigWidget", "Equalize Stretch", None, -1))
        self.button_linear_5_0.setToolTip(QtWidgets.QApplication.translate("StretchConfigWidget", "Apply a 5% linear stretch to all channels", None, -1))
        self.button_linear_5_0.setText(QtWidgets.QApplication.translate("StretchConfigWidget", "5% linear", None, -1))
        self.rb_stretch_none.setToolTip(QtWidgets.QApplication.translate("StretchConfigWidget", "No contrast stretch will be applied", None, -1))
        self.rb_stretch_none.setText(QtWidgets.QApplication.translate("StretchConfigWidget", "Full Linear Stretch", None, -1))
        self.groupbox_conditioner.setTitle(QtWidgets.QApplication.translate("StretchConfigWidget", "Conditioner", None, -1))
        self.rb_cond_log.setToolTip(QtWidgets.QApplication.translate("StretchConfigWidget", "Apply a logarithmic conditioner before applying stretch", None, -1))
        self.rb_cond_log.setText(QtWidgets.QApplication.translate("StretchConfigWidget", "Logarithmic", None, -1))
        self.rb_cond_sqrt.setToolTip(QtWidgets.QApplication.translate("StretchConfigWidget", "Apply a square-root conditioner before applying stretch", None, -1))
        self.rb_cond_sqrt.setText(QtWidgets.QApplication.translate("StretchConfigWidget", "Square root", None, -1))
        self.rb_cond_none.setToolTip(QtWidgets.QApplication.translate("StretchConfigWidget", "Use no conditioner before applying stretch", None, -1))
        self.rb_cond_none.setText(QtWidgets.QApplication.translate("StretchConfigWidget", "None", None, -1))

