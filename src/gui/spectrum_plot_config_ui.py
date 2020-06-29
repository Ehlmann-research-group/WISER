# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'spectrum_plot_config.ui'
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


class Ui_SpectrumPlotConfig(object):
    def setupUi(self, SpectrumPlotConfig):
        if not SpectrumPlotConfig.objectName():
            SpectrumPlotConfig.setObjectName(u"SpectrumPlotConfig")
        SpectrumPlotConfig.resize(357, 215)
        self.buttonBox = QDialogButtonBox(SpectrumPlotConfig)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setGeometry(QRect(10, 180, 341, 32))
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.gridLayoutWidget = QWidget(SpectrumPlotConfig)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(9, 9, 341, 161))
        self.grid_layout = QGridLayout(self.gridLayoutWidget)
        self.grid_layout.setObjectName(u"grid_layout")
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.lineedit_area_avg_y = QLineEdit(self.gridLayoutWidget)
        self.lineedit_area_avg_y.setObjectName(u"lineedit_area_avg_y")

        self.grid_layout.addWidget(self.lineedit_area_avg_y, 2, 2, 1, 1)

        self.label_placeholder = QLabel(self.gridLayoutWidget)
        self.label_placeholder.setObjectName(u"label_placeholder")

        self.grid_layout.addWidget(self.label_placeholder, 3, 0, 1, 3)

        self.lineedit_area_avg_x = QLineEdit(self.gridLayoutWidget)
        self.lineedit_area_avg_x.setObjectName(u"lineedit_area_avg_x")

        self.grid_layout.addWidget(self.lineedit_area_avg_x, 1, 2, 1, 1)

        self.label_area_avg_y = QLabel(self.gridLayoutWidget)
        self.label_area_avg_y.setObjectName(u"label_area_avg_y")

        self.grid_layout.addWidget(self.label_area_avg_y, 2, 1, 1, 1)

        self.combobox_avg_mode = QComboBox(self.gridLayoutWidget)
        self.combobox_avg_mode.setObjectName(u"combobox_avg_mode")

        self.grid_layout.addWidget(self.combobox_avg_mode, 0, 1, 1, 2)

        self.label_area_avg = QLabel(self.gridLayoutWidget)
        self.label_area_avg.setObjectName(u"label_area_avg")
        self.label_area_avg.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.grid_layout.addWidget(self.label_area_avg, 1, 0, 1, 1)

        self.label_area_avg_x = QLabel(self.gridLayoutWidget)
        self.label_area_avg_x.setObjectName(u"label_area_avg_x")

        self.grid_layout.addWidget(self.label_area_avg_x, 1, 1, 1, 1)

        self.label_avg_mode = QLabel(self.gridLayoutWidget)
        self.label_avg_mode.setObjectName(u"label_avg_mode")

        self.grid_layout.addWidget(self.label_avg_mode, 0, 0, 1, 1)

        self.grid_layout.setColumnStretch(0, 1)
#if QT_CONFIG(shortcut)
        self.label_area_avg_y.setBuddy(self.lineedit_area_avg_y)
        self.label_area_avg_x.setBuddy(self.lineedit_area_avg_x)
        self.label_avg_mode.setBuddy(self.combobox_avg_mode)
#endif // QT_CONFIG(shortcut)
        QWidget.setTabOrder(self.combobox_avg_mode, self.lineedit_area_avg_x)
        QWidget.setTabOrder(self.lineedit_area_avg_x, self.lineedit_area_avg_y)

        self.retranslateUi(SpectrumPlotConfig)
        self.buttonBox.accepted.connect(SpectrumPlotConfig.accept)
        self.buttonBox.rejected.connect(SpectrumPlotConfig.reject)

        QMetaObject.connectSlotsByName(SpectrumPlotConfig)
    # setupUi

    def retranslateUi(self, SpectrumPlotConfig):
        SpectrumPlotConfig.setWindowTitle(QCoreApplication.translate("SpectrumPlotConfig", u"Spectrum Plot Configuration", None))
        self.label_placeholder.setText(QCoreApplication.translate("SpectrumPlotConfig", u"TODO:  Spectrum plot configuration (tick marks, etc.)", None))
        self.label_area_avg_y.setText(QCoreApplication.translate("SpectrumPlotConfig", u"Y", None))
        self.label_area_avg.setText(QCoreApplication.translate("SpectrumPlotConfig", u"Default area-average size", None))
        self.label_area_avg_x.setText(QCoreApplication.translate("SpectrumPlotConfig", u"X", None))
        self.label_avg_mode.setText(QCoreApplication.translate("SpectrumPlotConfig", u"Default area-average mode", None))
    # retranslateUi

