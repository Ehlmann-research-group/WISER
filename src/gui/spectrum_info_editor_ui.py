# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'spectrum_info_editor.ui'
##
## Created by: Qt User Interface Compiler version 5.14.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *


class Ui_SpectrumInfoEditor(object):
    def setupUi(self, SpectrumInfoEditor):
        if not SpectrumInfoEditor.objectName():
            SpectrumInfoEditor.setObjectName(u"SpectrumInfoEditor")
        SpectrumInfoEditor.resize(333, 336)
        SpectrumInfoEditor.setModal(True)
        self.buttonBox = QDialogButtonBox(SpectrumInfoEditor)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setGeometry(QRect(10, 300, 311, 32))
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.gridLayoutWidget = QWidget(SpectrumInfoEditor)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(10, 10, 316, 291))
        self.grid_layout = QGridLayout(self.gridLayoutWidget)
        self.grid_layout.setObjectName(u"grid_layout")
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.label_location = QLabel(self.gridLayoutWidget)
        self.label_location.setObjectName(u"label_location")

        self.grid_layout.addWidget(self.label_location, 4, 0, 1, 1)

        self.lineedit_area_avg_y = QLineEdit(self.gridLayoutWidget)
        self.lineedit_area_avg_y.setObjectName(u"lineedit_area_avg_y")

        self.grid_layout.addWidget(self.lineedit_area_avg_y, 7, 2, 1, 1)

        self.lineedit_dataset = QLineEdit(self.gridLayoutWidget)
        self.lineedit_dataset.setObjectName(u"lineedit_dataset")
        self.lineedit_dataset.setReadOnly(True)

        self.grid_layout.addWidget(self.lineedit_dataset, 3, 1, 1, 2)

        self.label_dataset = QLabel(self.gridLayoutWidget)
        self.label_dataset.setObjectName(u"label_dataset")

        self.grid_layout.addWidget(self.label_dataset, 3, 0, 1, 1)

        self.label_spectrum_type = QLabel(self.gridLayoutWidget)
        self.label_spectrum_type.setObjectName(u"label_spectrum_type")

        self.grid_layout.addWidget(self.label_spectrum_type, 2, 0, 1, 1)

        self.combobox_avg_mode = QComboBox(self.gridLayoutWidget)
        self.combobox_avg_mode.addItem("")
        self.combobox_avg_mode.addItem("")
        self.combobox_avg_mode.setObjectName(u"combobox_avg_mode")

        self.grid_layout.addWidget(self.combobox_avg_mode, 5, 1, 1, 2)

        self.label_plot_color = QLabel(self.gridLayoutWidget)
        self.label_plot_color.setObjectName(u"label_plot_color")

        self.grid_layout.addWidget(self.label_plot_color, 8, 0, 1, 1)

        self.label_area_avg_y = QLabel(self.gridLayoutWidget)
        self.label_area_avg_y.setObjectName(u"label_area_avg_y")

        self.grid_layout.addWidget(self.label_area_avg_y, 7, 1, 1, 1)

        self.lineedit_location = QLineEdit(self.gridLayoutWidget)
        self.lineedit_location.setObjectName(u"lineedit_location")
        self.lineedit_location.setReadOnly(True)

        self.grid_layout.addWidget(self.lineedit_location, 4, 1, 1, 2)

        self.label_area_avg_x = QLabel(self.gridLayoutWidget)
        self.label_area_avg_x.setObjectName(u"label_area_avg_x")

        self.grid_layout.addWidget(self.label_area_avg_x, 6, 1, 1, 1)

        self.grid_layout_plot_color = QGridLayout()
        self.grid_layout_plot_color.setSpacing(0)
        self.grid_layout_plot_color.setObjectName(u"grid_layout_plot_color")
        self.lineedit_plot_color = QLineEdit(self.gridLayoutWidget)
        self.lineedit_plot_color.setObjectName(u"lineedit_plot_color")

        self.grid_layout_plot_color.addWidget(self.lineedit_plot_color, 0, 0, 1, 1)

        self.button_plot_color = QPushButton(self.gridLayoutWidget)
        self.button_plot_color.setObjectName(u"button_plot_color")
        self.button_plot_color.setMinimumSize(QSize(32, 32))

        self.grid_layout_plot_color.addWidget(self.button_plot_color, 0, 1, 1, 1)

        self.grid_layout_plot_color.setColumnStretch(0, 10)

        self.grid_layout.addLayout(self.grid_layout_plot_color, 8, 1, 1, 2)

        self.label_avg_mode = QLabel(self.gridLayoutWidget)
        self.label_avg_mode.setObjectName(u"label_avg_mode")

        self.grid_layout.addWidget(self.label_avg_mode, 5, 0, 1, 1)

        self.lineedit_area_avg_x = QLineEdit(self.gridLayoutWidget)
        self.lineedit_area_avg_x.setObjectName(u"lineedit_area_avg_x")

        self.grid_layout.addWidget(self.lineedit_area_avg_x, 6, 2, 1, 1)

        self.label_description = QLabel(self.gridLayoutWidget)
        self.label_description.setObjectName(u"label_description")

        self.grid_layout.addWidget(self.label_description, 0, 0, 1, 1)

        self.lineedit_name = QLineEdit(self.gridLayoutWidget)
        self.lineedit_name.setObjectName(u"lineedit_name")

        self.grid_layout.addWidget(self.lineedit_name, 1, 0, 1, 3)

        self.label_area_avg = QLabel(self.gridLayoutWidget)
        self.label_area_avg.setObjectName(u"label_area_avg")
        self.label_area_avg.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.grid_layout.addWidget(self.label_area_avg, 6, 0, 1, 1)

        self.lineedit_spectrum_type = QLineEdit(self.gridLayoutWidget)
        self.lineedit_spectrum_type.setObjectName(u"lineedit_spectrum_type")
        self.lineedit_spectrum_type.setReadOnly(True)

        self.grid_layout.addWidget(self.lineedit_spectrum_type, 2, 1, 1, 2)

        self.grid_layout.setColumnStretch(0, 10)
#if QT_CONFIG(shortcut)
        self.label_location.setBuddy(self.lineedit_location)
        self.label_dataset.setBuddy(self.lineedit_dataset)
        self.label_plot_color.setBuddy(self.lineedit_plot_color)
        self.label_area_avg_y.setBuddy(self.lineedit_area_avg_y)
        self.label_area_avg_x.setBuddy(self.lineedit_area_avg_x)
        self.label_avg_mode.setBuddy(self.combobox_avg_mode)
        self.label_description.setBuddy(self.lineedit_name)
#endif // QT_CONFIG(shortcut)
        QWidget.setTabOrder(self.lineedit_name, self.lineedit_spectrum_type)
        QWidget.setTabOrder(self.lineedit_spectrum_type, self.lineedit_dataset)
        QWidget.setTabOrder(self.lineedit_dataset, self.lineedit_location)
        QWidget.setTabOrder(self.lineedit_location, self.combobox_avg_mode)
        QWidget.setTabOrder(self.combobox_avg_mode, self.lineedit_area_avg_x)
        QWidget.setTabOrder(self.lineedit_area_avg_x, self.lineedit_area_avg_y)
        QWidget.setTabOrder(self.lineedit_area_avg_y, self.lineedit_plot_color)
        QWidget.setTabOrder(self.lineedit_plot_color, self.button_plot_color)

        self.retranslateUi(SpectrumInfoEditor)
        self.buttonBox.accepted.connect(SpectrumInfoEditor.accept)
        self.buttonBox.rejected.connect(SpectrumInfoEditor.reject)

        QMetaObject.connectSlotsByName(SpectrumInfoEditor)
    # setupUi

    def retranslateUi(self, SpectrumInfoEditor):
        SpectrumInfoEditor.setWindowTitle(QCoreApplication.translate("SpectrumInfoEditor", u"Spectrum Information", None))
        self.label_location.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Location", None))
        self.label_dataset.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Data set", None))
        self.label_spectrum_type.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Spectrum type", None))
        self.combobox_avg_mode.setItemText(0, QCoreApplication.translate("SpectrumInfoEditor", u"Mean", None))
        self.combobox_avg_mode.setItemText(1, QCoreApplication.translate("SpectrumInfoEditor", u"Median", None))

        self.label_plot_color.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Plot color", None))
        self.label_area_avg_y.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Y", None))
        self.label_area_avg_x.setText(QCoreApplication.translate("SpectrumInfoEditor", u"X", None))
        self.button_plot_color.setText(QCoreApplication.translate("SpectrumInfoEditor", u"...", None))
        self.label_avg_mode.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Average mode", None))
        self.label_description.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Name:", None))
        self.label_area_avg.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Area-average size", None))
    # retranslateUi

