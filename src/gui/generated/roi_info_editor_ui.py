# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'roi_info_editor.ui'
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


class Ui_ROIInfoEditor(object):
    def setupUi(self, ROIInfoEditor):
        if not ROIInfoEditor.objectName():
            ROIInfoEditor.setObjectName(u"ROIInfoEditor")
        ROIInfoEditor.resize(366, 227)
        self.gridLayout = QGridLayout(ROIInfoEditor)
        self.gridLayout.setObjectName(u"gridLayout")
        self.lineedit_name = QLineEdit(ROIInfoEditor)
        self.lineedit_name.setObjectName(u"lineedit_name")

        self.gridLayout.addWidget(self.lineedit_name, 0, 1, 1, 1)

        self.lbl_desc = QLabel(ROIInfoEditor)
        self.lbl_desc.setObjectName(u"lbl_desc")
        self.lbl_desc.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.gridLayout.addWidget(self.lbl_desc, 2, 0, 1, 1)

        self.lbl_name = QLabel(ROIInfoEditor)
        self.lbl_name.setObjectName(u"lbl_name")

        self.gridLayout.addWidget(self.lbl_name, 0, 0, 1, 1)

        self.lbl_color = QLabel(ROIInfoEditor)
        self.lbl_color.setObjectName(u"lbl_color")

        self.gridLayout.addWidget(self.lbl_color, 1, 0, 1, 1)

        self.widget = QWidget(ROIInfoEditor)
        self.widget.setObjectName(u"widget")
        self.gridLayout_2 = QGridLayout(self.widget)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.btn_color_chooser = QPushButton(self.widget)
        self.btn_color_chooser.setObjectName(u"btn_color_chooser")

        self.gridLayout_2.addWidget(self.btn_color_chooser, 0, 1, 1, 1)

        self.lineedit_color = QLineEdit(self.widget)
        self.lineedit_color.setObjectName(u"lineedit_color")

        self.gridLayout_2.addWidget(self.lineedit_color, 0, 0, 1, 1)

        self.gridLayout_2.setColumnStretch(0, 1)

        self.gridLayout.addWidget(self.widget, 1, 1, 1, 1)

        self.buttonBox = QDialogButtonBox(ROIInfoEditor)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.gridLayout.addWidget(self.buttonBox, 3, 0, 1, 2)

        self.textedit_desc = QPlainTextEdit(ROIInfoEditor)
        self.textedit_desc.setObjectName(u"textedit_desc")

        self.gridLayout.addWidget(self.textedit_desc, 2, 1, 1, 1)

#if QT_CONFIG(shortcut)
        self.lbl_desc.setBuddy(self.textedit_desc)
        self.lbl_name.setBuddy(self.lineedit_name)
        self.lbl_color.setBuddy(self.lineedit_color)
#endif // QT_CONFIG(shortcut)
        QWidget.setTabOrder(self.lineedit_name, self.lineedit_color)
        QWidget.setTabOrder(self.lineedit_color, self.btn_color_chooser)
        QWidget.setTabOrder(self.btn_color_chooser, self.textedit_desc)

        self.retranslateUi(ROIInfoEditor)
        self.buttonBox.accepted.connect(ROIInfoEditor.accept)
        self.buttonBox.rejected.connect(ROIInfoEditor.reject)

        QMetaObject.connectSlotsByName(ROIInfoEditor)
    # setupUi

    def retranslateUi(self, ROIInfoEditor):
        ROIInfoEditor.setWindowTitle(QCoreApplication.translate("ROIInfoEditor", u"Region of Interest - Information", None))
        self.lbl_desc.setText(QCoreApplication.translate("ROIInfoEditor", u"Description:", None))
        self.lbl_name.setText(QCoreApplication.translate("ROIInfoEditor", u"Name:", None))
        self.lbl_color.setText(QCoreApplication.translate("ROIInfoEditor", u"Color:", None))
#if QT_CONFIG(tooltip)
        self.btn_color_chooser.setToolTip(QCoreApplication.translate("ROIInfoEditor", u"Show Color Chooser dialog", None))
#endif // QT_CONFIG(tooltip)
        self.btn_color_chooser.setText(QCoreApplication.translate("ROIInfoEditor", u"...", None))
    # retranslateUi

