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
        SpectrumInfoEditor.resize(419, 329)
        SpectrumInfoEditor.setModal(True)
        self.gridLayout = QGridLayout(SpectrumInfoEditor)
        self.gridLayout.setObjectName(u"gridLayout")
        self.lineedit_area_avg_y = QLineEdit(SpectrumInfoEditor)
        self.lineedit_area_avg_y.setObjectName(u"lineedit_area_avg_y")

        self.gridLayout.addWidget(self.lineedit_area_avg_y, 6, 2, 1, 1)

        self.label_location = QLabel(SpectrumInfoEditor)
        self.label_location.setObjectName(u"label_location")

        self.gridLayout.addWidget(self.label_location, 3, 0, 1, 2)

        self.lineedit_dataset = QLineEdit(SpectrumInfoEditor)
        self.lineedit_dataset.setObjectName(u"lineedit_dataset")
        self.lineedit_dataset.setReadOnly(True)

        self.gridLayout.addWidget(self.lineedit_dataset, 2, 2, 1, 1)

        self.lineedit_area_avg_x = QLineEdit(SpectrumInfoEditor)
        self.lineedit_area_avg_x.setObjectName(u"lineedit_area_avg_x")

        self.gridLayout.addWidget(self.lineedit_area_avg_x, 5, 2, 1, 1)

        self.label_area_avg = QLabel(SpectrumInfoEditor)
        self.label_area_avg.setObjectName(u"label_area_avg")
        self.label_area_avg.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_area_avg, 5, 0, 1, 1)

        self.label_area_avg_x = QLabel(SpectrumInfoEditor)
        self.label_area_avg_x.setObjectName(u"label_area_avg_x")

        self.gridLayout.addWidget(self.label_area_avg_x, 5, 1, 1, 1)

        self.label_description = QLabel(SpectrumInfoEditor)
        self.label_description.setObjectName(u"label_description")

        self.gridLayout.addWidget(self.label_description, 0, 0, 1, 2)

        self.label_dataset = QLabel(SpectrumInfoEditor)
        self.label_dataset.setObjectName(u"label_dataset")

        self.gridLayout.addWidget(self.label_dataset, 2, 0, 1, 2)

        self.lineedit_location = QLineEdit(SpectrumInfoEditor)
        self.lineedit_location.setObjectName(u"lineedit_location")
        self.lineedit_location.setReadOnly(True)

        self.gridLayout.addWidget(self.lineedit_location, 3, 2, 1, 1)

        self.label_avg_mode = QLabel(SpectrumInfoEditor)
        self.label_avg_mode.setObjectName(u"label_avg_mode")

        self.gridLayout.addWidget(self.label_avg_mode, 4, 0, 1, 2)

        self.label_spectrum_type = QLabel(SpectrumInfoEditor)
        self.label_spectrum_type.setObjectName(u"label_spectrum_type")

        self.gridLayout.addWidget(self.label_spectrum_type, 1, 0, 1, 2)

        self.lineedit_spectrum_type = QLineEdit(SpectrumInfoEditor)
        self.lineedit_spectrum_type.setObjectName(u"lineedit_spectrum_type")
        self.lineedit_spectrum_type.setReadOnly(True)

        self.gridLayout.addWidget(self.lineedit_spectrum_type, 1, 2, 1, 1)

        self.lineedit_name = QLineEdit(SpectrumInfoEditor)
        self.lineedit_name.setObjectName(u"lineedit_name")

        self.gridLayout.addWidget(self.lineedit_name, 0, 2, 1, 1)

        self.buttonBox = QDialogButtonBox(SpectrumInfoEditor)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.gridLayout.addWidget(self.buttonBox, 9, 0, 1, 3)

        self.combobox_avg_mode = QComboBox(SpectrumInfoEditor)
        self.combobox_avg_mode.addItem("")
        self.combobox_avg_mode.addItem("")
        self.combobox_avg_mode.setObjectName(u"combobox_avg_mode")

        self.gridLayout.addWidget(self.combobox_avg_mode, 4, 2, 1, 1)

        self.widget = QWidget(SpectrumInfoEditor)
        self.widget.setObjectName(u"widget")
        self.gridLayout_2 = QGridLayout(self.widget)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.lineedit_plot_color = QLineEdit(self.widget)
        self.lineedit_plot_color.setObjectName(u"lineedit_plot_color")

        self.gridLayout_2.addWidget(self.lineedit_plot_color, 0, 0, 1, 1)

        self.button_plot_color = QPushButton(self.widget)
        self.button_plot_color.setObjectName(u"button_plot_color")
        self.button_plot_color.setMinimumSize(QSize(32, 32))

        self.gridLayout_2.addWidget(self.button_plot_color, 0, 1, 1, 1)

        self.gridLayout_2.setColumnStretch(0, 1)

        self.gridLayout.addWidget(self.widget, 7, 2, 1, 1)

        self.label_area_avg_y = QLabel(SpectrumInfoEditor)
        self.label_area_avg_y.setObjectName(u"label_area_avg_y")

        self.gridLayout.addWidget(self.label_area_avg_y, 6, 1, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 8, 0, 1, 2)

        self.label_plot_color = QLabel(SpectrumInfoEditor)
        self.label_plot_color.setObjectName(u"label_plot_color")

        self.gridLayout.addWidget(self.label_plot_color, 7, 0, 1, 2)

#if QT_CONFIG(shortcut)
        self.label_location.setBuddy(self.lineedit_location)
        self.label_area_avg_x.setBuddy(self.lineedit_area_avg_x)
        self.label_description.setBuddy(self.lineedit_name)
        self.label_dataset.setBuddy(self.lineedit_dataset)
        self.label_avg_mode.setBuddy(self.combobox_avg_mode)
        self.label_spectrum_type.setBuddy(self.lineedit_spectrum_type)
        self.label_area_avg_y.setBuddy(self.lineedit_area_avg_y)
        self.label_plot_color.setBuddy(self.lineedit_plot_color)
#endif // QT_CONFIG(shortcut)

        self.retranslateUi(SpectrumInfoEditor)
        self.buttonBox.accepted.connect(SpectrumInfoEditor.accept)
        self.buttonBox.rejected.connect(SpectrumInfoEditor.reject)

        QMetaObject.connectSlotsByName(SpectrumInfoEditor)
    # setupUi

    def retranslateUi(self, SpectrumInfoEditor):
        SpectrumInfoEditor.setWindowTitle(QCoreApplication.translate("SpectrumInfoEditor", u"Spectrum Information", None))
        self.label_location.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Location:", None))
        self.label_area_avg.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Area-average size", None))
        self.label_area_avg_x.setText(QCoreApplication.translate("SpectrumInfoEditor", u"X", None))
        self.label_description.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Name:", None))
        self.label_dataset.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Data set:", None))
        self.label_avg_mode.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Average mode:", None))
        self.label_spectrum_type.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Spectrum type:", None))
        self.combobox_avg_mode.setItemText(0, QCoreApplication.translate("SpectrumInfoEditor", u"Mean", None))
        self.combobox_avg_mode.setItemText(1, QCoreApplication.translate("SpectrumInfoEditor", u"Median", None))

        self.button_plot_color.setText(QCoreApplication.translate("SpectrumInfoEditor", u"...", None))
        self.label_area_avg_y.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Y", None))
        self.label_plot_color.setText(QCoreApplication.translate("SpectrumInfoEditor", u"Plot color:", None))
    # retranslateUi

