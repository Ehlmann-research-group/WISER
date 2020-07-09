# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'band_chooser.ui'
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


class Ui_BandChooserDialog(object):
    def setupUi(self, BandChooserDialog):
        if not BandChooserDialog.objectName():
            BandChooserDialog.setObjectName(u"BandChooserDialog")
        BandChooserDialog.resize(350, 368)
        self.gridLayout = QGridLayout(BandChooserDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.buttonBox = QDialogButtonBox(BandChooserDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.gridLayout.addWidget(self.buttonBox, 3, 1, 1, 1)

        self.chk_apply_globally = QCheckBox(BandChooserDialog)
        self.chk_apply_globally.setObjectName(u"chk_apply_globally")

        self.gridLayout.addWidget(self.chk_apply_globally, 3, 0, 1, 1)

        self.gbox_detail = QGroupBox(BandChooserDialog)
        self.gbox_detail.setObjectName(u"gbox_detail")
        self.verticalLayout_2 = QVBoxLayout(self.gbox_detail)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.config_stack = QStackedWidget(self.gbox_detail)
        self.config_stack.setObjectName(u"config_stack")
        self.config_rgb = QWidget()
        self.config_rgb.setObjectName(u"config_rgb")
        self.gridLayout_2 = QGridLayout(self.config_rgb)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.cbox_red_band = QComboBox(self.config_rgb)
        self.cbox_red_band.setObjectName(u"cbox_red_band")

        self.gridLayout_2.addWidget(self.cbox_red_band, 0, 1, 1, 1)

        self.lbl_blue_band = QLabel(self.config_rgb)
        self.lbl_blue_band.setObjectName(u"lbl_blue_band")

        self.gridLayout_2.addWidget(self.lbl_blue_band, 2, 0, 1, 1)

        self.cbox_green_band = QComboBox(self.config_rgb)
        self.cbox_green_band.setObjectName(u"cbox_green_band")

        self.gridLayout_2.addWidget(self.cbox_green_band, 1, 1, 1, 1)

        self.cbox_blue_band = QComboBox(self.config_rgb)
        self.cbox_blue_band.setObjectName(u"cbox_blue_band")

        self.gridLayout_2.addWidget(self.cbox_blue_band, 2, 1, 1, 1)

        self.lbl_red_band = QLabel(self.config_rgb)
        self.lbl_red_band.setObjectName(u"lbl_red_band")

        self.gridLayout_2.addWidget(self.lbl_red_band, 0, 0, 1, 1)

        self.lbl_green_band = QLabel(self.config_rgb)
        self.lbl_green_band.setObjectName(u"lbl_green_band")

        self.gridLayout_2.addWidget(self.lbl_green_band, 1, 0, 1, 1)

        self.btn_rgb_choose_defaults = QPushButton(self.config_rgb)
        self.btn_rgb_choose_defaults.setObjectName(u"btn_rgb_choose_defaults")

        self.gridLayout_2.addWidget(self.btn_rgb_choose_defaults, 3, 0, 1, 2)

        self.btn_rgb_choose_visible = QPushButton(self.config_rgb)
        self.btn_rgb_choose_visible.setObjectName(u"btn_rgb_choose_visible")

        self.gridLayout_2.addWidget(self.btn_rgb_choose_visible, 4, 0, 1, 2)

        self.gridLayout_2.setColumnStretch(1, 1)
        self.config_stack.addWidget(self.config_rgb)
        self.config_grayscale = QWidget()
        self.config_grayscale.setObjectName(u"config_grayscale")
        self.gridLayout_3 = QGridLayout(self.config_grayscale)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.lbl_gray_band = QLabel(self.config_grayscale)
        self.lbl_gray_band.setObjectName(u"lbl_gray_band")

        self.gridLayout_3.addWidget(self.lbl_gray_band, 0, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer, 2, 0, 1, 1)

        self.cbox_gray_band = QComboBox(self.config_grayscale)
        self.cbox_gray_band.setObjectName(u"cbox_gray_band")

        self.gridLayout_3.addWidget(self.cbox_gray_band, 0, 1, 1, 1)

        self.btn_gray_choose_defaults = QPushButton(self.config_grayscale)
        self.btn_gray_choose_defaults.setObjectName(u"btn_gray_choose_defaults")

        self.gridLayout_3.addWidget(self.btn_gray_choose_defaults, 1, 0, 1, 2)

        self.gridLayout_3.setColumnStretch(1, 1)
        self.config_stack.addWidget(self.config_grayscale)

        self.verticalLayout_2.addWidget(self.config_stack)


        self.gridLayout.addWidget(self.gbox_detail, 1, 0, 1, 2)

        self.gbox_general = QGroupBox(BandChooserDialog)
        self.gbox_general.setObjectName(u"gbox_general")
        self.verticalLayout = QVBoxLayout(self.gbox_general)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.rbtn_rgb = QRadioButton(self.gbox_general)
        self.rbtn_rgb.setObjectName(u"rbtn_rgb")

        self.verticalLayout.addWidget(self.rbtn_rgb)

        self.rbtn_grayscale = QRadioButton(self.gbox_general)
        self.rbtn_grayscale.setObjectName(u"rbtn_grayscale")

        self.verticalLayout.addWidget(self.rbtn_grayscale)


        self.gridLayout.addWidget(self.gbox_general, 0, 0, 1, 2)

        self.gridLayout.setColumnStretch(0, 1)
#if QT_CONFIG(shortcut)
        self.lbl_blue_band.setBuddy(self.cbox_blue_band)
        self.lbl_red_band.setBuddy(self.cbox_red_band)
        self.lbl_green_band.setBuddy(self.cbox_green_band)
        self.lbl_gray_band.setBuddy(self.cbox_gray_band)
#endif // QT_CONFIG(shortcut)
        QWidget.setTabOrder(self.rbtn_rgb, self.rbtn_grayscale)
        QWidget.setTabOrder(self.rbtn_grayscale, self.cbox_red_band)
        QWidget.setTabOrder(self.cbox_red_band, self.cbox_green_band)
        QWidget.setTabOrder(self.cbox_green_band, self.cbox_blue_band)
        QWidget.setTabOrder(self.cbox_blue_band, self.btn_rgb_choose_defaults)
        QWidget.setTabOrder(self.btn_rgb_choose_defaults, self.btn_rgb_choose_visible)
        QWidget.setTabOrder(self.btn_rgb_choose_visible, self.cbox_gray_band)
        QWidget.setTabOrder(self.cbox_gray_band, self.btn_gray_choose_defaults)
        QWidget.setTabOrder(self.btn_gray_choose_defaults, self.chk_apply_globally)

        self.retranslateUi(BandChooserDialog)
        self.buttonBox.accepted.connect(BandChooserDialog.accept)
        self.buttonBox.rejected.connect(BandChooserDialog.reject)

        self.config_stack.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(BandChooserDialog)
    # setupUi

    def retranslateUi(self, BandChooserDialog):
        BandChooserDialog.setWindowTitle(QCoreApplication.translate("BandChooserDialog", u"Band Chooser", None))
        self.chk_apply_globally.setText(QCoreApplication.translate("BandChooserDialog", u"&Apply to all views", None))
        self.gbox_detail.setTitle(QCoreApplication.translate("BandChooserDialog", u"Detail", None))
        self.lbl_blue_band.setText(QCoreApplication.translate("BandChooserDialog", u"Blue Band", None))
        self.lbl_red_band.setText(QCoreApplication.translate("BandChooserDialog", u"Red Band", None))
        self.lbl_green_band.setText(QCoreApplication.translate("BandChooserDialog", u"Green Band", None))
        self.btn_rgb_choose_defaults.setText(QCoreApplication.translate("BandChooserDialog", u"Choose &Default Bands", None))
        self.btn_rgb_choose_visible.setText(QCoreApplication.translate("BandChooserDialog", u"Choose &Visible-Light Bands", None))
        self.lbl_gray_band.setText(QCoreApplication.translate("BandChooserDialog", u"Grayscale Band", None))
        self.btn_gray_choose_defaults.setText(QCoreApplication.translate("BandChooserDialog", u"Choose &Default Band", None))
        self.gbox_general.setTitle(QCoreApplication.translate("BandChooserDialog", u"General", None))
        self.rbtn_rgb.setText(QCoreApplication.translate("BandChooserDialog", u"&RGB", None))
        self.rbtn_grayscale.setText(QCoreApplication.translate("BandChooserDialog", u"&Grayscale", None))
    # retranslateUi

