# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'export_image.ui'
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


class Ui_ExportImageDialog(object):
    def setupUi(self, ExportImageDialog):
        if not ExportImageDialog.objectName():
            ExportImageDialog.setObjectName(u"ExportImageDialog")
        ExportImageDialog.resize(283, 265)
        self.gridLayout = QGridLayout(ExportImageDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.lbl_image_format = QLabel(ExportImageDialog)
        self.lbl_image_format.setObjectName(u"lbl_image_format")

        self.gridLayout.addWidget(self.lbl_image_format, 4, 0, 1, 1)

        self.cbox_image_format = QComboBox(ExportImageDialog)
        self.cbox_image_format.setObjectName(u"cbox_image_format")

        self.gridLayout.addWidget(self.cbox_image_format, 4, 1, 1, 1)

        self.stack_image_config = QStackedWidget(ExportImageDialog)
        self.stack_image_config.setObjectName(u"stack_image_config")
        self.page_png = QWidget()
        self.page_png.setObjectName(u"page_png")
        self.verticalLayout = QVBoxLayout(self.page_png)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lbl_png = QLabel(self.page_png)
        self.lbl_png.setObjectName(u"lbl_png")
        self.lbl_png.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lbl_png)

        self.stack_image_config.addWidget(self.page_png)
        self.page_tiff = QWidget()
        self.page_tiff.setObjectName(u"page_tiff")
        self.verticalLayout_3 = QVBoxLayout(self.page_tiff)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gbox_tiff = QGroupBox(self.page_tiff)
        self.gbox_tiff.setObjectName(u"gbox_tiff")
        self.gridLayout_4 = QGridLayout(self.gbox_tiff)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.lbl_tiff_quality = QLabel(self.gbox_tiff)
        self.lbl_tiff_quality.setObjectName(u"lbl_tiff_quality")

        self.gridLayout_4.addWidget(self.lbl_tiff_quality, 1, 0, 1, 1)

        self.lbl_tiff_compression = QLabel(self.gbox_tiff)
        self.lbl_tiff_compression.setObjectName(u"lbl_tiff_compression")

        self.gridLayout_4.addWidget(self.lbl_tiff_compression, 0, 0, 1, 1)

        self.cbox_tiff_compression = QComboBox(self.gbox_tiff)
        self.cbox_tiff_compression.setObjectName(u"cbox_tiff_compression")

        self.gridLayout_4.addWidget(self.cbox_tiff_compression, 0, 1, 1, 1)

        self.sbox_tiff_quality = QSpinBox(self.gbox_tiff)
        self.sbox_tiff_quality.setObjectName(u"sbox_tiff_quality")
        self.sbox_tiff_quality.setMaximum(95)

        self.gridLayout_4.addWidget(self.sbox_tiff_quality, 1, 1, 1, 1)

        self.gridLayout_4.setColumnStretch(1, 1)

        self.verticalLayout_3.addWidget(self.gbox_tiff)

        self.stack_image_config.addWidget(self.page_tiff)
        self.page_jpeg = QWidget()
        self.page_jpeg.setObjectName(u"page_jpeg")
        self.verticalLayout_2 = QVBoxLayout(self.page_jpeg)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gbox_jpeg = QGroupBox(self.page_jpeg)
        self.gbox_jpeg.setObjectName(u"gbox_jpeg")
        self.gridLayout_3 = QGridLayout(self.gbox_jpeg)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.sbox_jpeg_quality = QSpinBox(self.gbox_jpeg)
        self.sbox_jpeg_quality.setObjectName(u"sbox_jpeg_quality")
        self.sbox_jpeg_quality.setMaximum(95)

        self.gridLayout_3.addWidget(self.sbox_jpeg_quality, 0, 1, 1, 1)

        self.lbl_jpeg_quality = QLabel(self.gbox_jpeg)
        self.lbl_jpeg_quality.setObjectName(u"lbl_jpeg_quality")

        self.gridLayout_3.addWidget(self.lbl_jpeg_quality, 0, 0, 1, 1)

        self.gridLayout_3.setColumnStretch(1, 1)

        self.verticalLayout_2.addWidget(self.gbox_jpeg)

        self.stack_image_config.addWidget(self.page_jpeg)

        self.gridLayout.addWidget(self.stack_image_config, 6, 0, 1, 2)

        self.widget = QWidget(ExportImageDialog)
        self.widget.setObjectName(u"widget")
        self.gridLayout_5 = QGridLayout(self.widget)
        self.gridLayout_5.setSpacing(0)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.ledit_filename = QLineEdit(self.widget)
        self.ledit_filename.setObjectName(u"ledit_filename")

        self.gridLayout_5.addWidget(self.ledit_filename, 1, 0, 1, 1)

        self.btn_filename = QPushButton(self.widget)
        self.btn_filename.setObjectName(u"btn_filename")

        self.gridLayout_5.addWidget(self.btn_filename, 1, 1, 1, 1)

        self.lbl_filename = QLabel(self.widget)
        self.lbl_filename.setObjectName(u"lbl_filename")

        self.gridLayout_5.addWidget(self.lbl_filename, 0, 0, 1, 2)

        self.gridLayout_5.setColumnStretch(0, 1)

        self.gridLayout.addWidget(self.widget, 3, 0, 1, 2)

        self.buttonBox = QDialogButtonBox(ExportImageDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Save)

        self.gridLayout.addWidget(self.buttonBox, 7, 0, 1, 2)

        self.gridLayout.setColumnStretch(1, 1)
#if QT_CONFIG(shortcut)
        self.lbl_image_format.setBuddy(self.cbox_image_format)
        self.lbl_tiff_quality.setBuddy(self.sbox_tiff_quality)
        self.lbl_tiff_compression.setBuddy(self.cbox_tiff_compression)
        self.lbl_jpeg_quality.setBuddy(self.sbox_jpeg_quality)
        self.lbl_filename.setBuddy(self.ledit_filename)
#endif // QT_CONFIG(shortcut)
        QWidget.setTabOrder(self.ledit_filename, self.btn_filename)
        QWidget.setTabOrder(self.btn_filename, self.cbox_image_format)
        QWidget.setTabOrder(self.cbox_image_format, self.cbox_tiff_compression)
        QWidget.setTabOrder(self.cbox_tiff_compression, self.sbox_tiff_quality)
        QWidget.setTabOrder(self.sbox_tiff_quality, self.sbox_jpeg_quality)

        self.retranslateUi(ExportImageDialog)
        self.buttonBox.accepted.connect(ExportImageDialog.accept)
        self.buttonBox.rejected.connect(ExportImageDialog.reject)

        self.stack_image_config.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(ExportImageDialog)
    # setupUi

    def retranslateUi(self, ExportImageDialog):
        ExportImageDialog.setWindowTitle(QCoreApplication.translate("ExportImageDialog", u"Export Image", None))
        self.lbl_image_format.setText(QCoreApplication.translate("ExportImageDialog", u"Image format", None))
        self.lbl_png.setText(QCoreApplication.translate("ExportImageDialog", u"No configuration required\n"
"for the PNG image format", None))
        self.gbox_tiff.setTitle(QCoreApplication.translate("ExportImageDialog", u"TIFF Settings", None))
        self.lbl_tiff_quality.setText(QCoreApplication.translate("ExportImageDialog", u"Image Quality", None))
        self.lbl_tiff_compression.setText(QCoreApplication.translate("ExportImageDialog", u"Compression", None))
        self.gbox_jpeg.setTitle(QCoreApplication.translate("ExportImageDialog", u"JPEG Settings", None))
        self.lbl_jpeg_quality.setText(QCoreApplication.translate("ExportImageDialog", u"Image Quality", None))
        self.btn_filename.setText(QCoreApplication.translate("ExportImageDialog", u"...", None))
        self.lbl_filename.setText(QCoreApplication.translate("ExportImageDialog", u"Filename", None))
    # retranslateUi

