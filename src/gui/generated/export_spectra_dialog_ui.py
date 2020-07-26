# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'export_spectra_dialog.ui'
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


class Ui_ExportSpectraDialog(object):
    def setupUi(self, ExportSpectraDialog):
        if not ExportSpectraDialog.objectName():
            ExportSpectraDialog.setObjectName(u"ExportSpectraDialog")
        ExportSpectraDialog.resize(400, 300)
        self.gridLayout = QGridLayout(ExportSpectraDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_2 = QLabel(ExportSpectraDialog)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 3, 1, 1, 1)

        self.comboBox = QComboBox(ExportSpectraDialog)
        self.comboBox.setObjectName(u"comboBox")

        self.gridLayout.addWidget(self.comboBox, 3, 2, 1, 1)

        self.label_3 = QLabel(ExportSpectraDialog)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 1, 1, 2)

        self.widget = QWidget(ExportSpectraDialog)
        self.widget.setObjectName(u"widget")
        self.gridLayout_2 = QGridLayout(self.widget)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.lineEdit = QLineEdit(self.widget)
        self.lineEdit.setObjectName(u"lineEdit")

        self.gridLayout_2.addWidget(self.lineEdit, 1, 0, 1, 1)

        self.pushButton = QPushButton(self.widget)
        self.pushButton.setObjectName(u"pushButton")

        self.gridLayout_2.addWidget(self.pushButton, 1, 1, 1, 1)

        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)

        self.gridLayout_2.setColumnStretch(0, 1)

        self.gridLayout.addWidget(self.widget, 2, 1, 1, 2)

        self.label_5 = QLabel(ExportSpectraDialog)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.gridLayout.addWidget(self.label_5, 6, 1, 1, 2)

        self.comboBox_2 = QComboBox(ExportSpectraDialog)
        self.comboBox_2.setObjectName(u"comboBox_2")

        self.gridLayout.addWidget(self.comboBox_2, 5, 2, 1, 1)

        self.label_4 = QLabel(ExportSpectraDialog)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 5, 1, 1, 1)

        self.buttonBox = QDialogButtonBox(ExportSpectraDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.gridLayout.addWidget(self.buttonBox, 7, 1, 1, 2)

        self.gridLayout.setRowStretch(6, 1)

        self.retranslateUi(ExportSpectraDialog)
        self.buttonBox.accepted.connect(ExportSpectraDialog.accept)
        self.buttonBox.rejected.connect(ExportSpectraDialog.reject)

        QMetaObject.connectSlotsByName(ExportSpectraDialog)
    # setupUi

    def retranslateUi(self, ExportSpectraDialog):
        ExportSpectraDialog.setWindowTitle(QCoreApplication.translate("ExportSpectraDialog", u"Export Spectra", None))
        self.label_2.setText(QCoreApplication.translate("ExportSpectraDialog", u"Wavelength units:", None))
        self.label_3.setText(QCoreApplication.translate("ExportSpectraDialog", u"TODO:  Description of what is being output goes here.  Example:\n"
"\n"
"Saving 155 spectra with one set of wavelengths.", None))
        self.pushButton.setText(QCoreApplication.translate("ExportSpectraDialog", u"...", None))
        self.label.setText(QCoreApplication.translate("ExportSpectraDialog", u"Output filename:", None))
        self.label_5.setText(QCoreApplication.translate("ExportSpectraDialog", u"TODO:  Descriptive image of output format will go here!", None))
        self.label_4.setText(QCoreApplication.translate("ExportSpectraDialog", u"Output format:", None))
    # retranslateUi

