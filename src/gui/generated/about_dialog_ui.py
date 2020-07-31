# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'about_dialog.ui'
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


class Ui_AboutDialog(object):
    def setupUi(self, AboutDialog):
        if not AboutDialog.objectName():
            AboutDialog.setObjectName(u"AboutDialog")
        AboutDialog.resize(611, 387)
        self.gridLayout = QGridLayout(AboutDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.textEdit = QTextEdit(AboutDialog)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setReadOnly(True)

        self.gridLayout.addWidget(self.textEdit, 3, 0, 1, 2)

        self.label_version = QLabel(AboutDialog)
        self.label_version.setObjectName(u"label_version")

        self.gridLayout.addWidget(self.label_version, 1, 0, 1, 2)

        self.buttonBox = QDialogButtonBox(AboutDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Ok)

        self.gridLayout.addWidget(self.buttonBox, 4, 1, 1, 1)

        self.btn_system_info = QPushButton(AboutDialog)
        self.btn_system_info.setObjectName(u"btn_system_info")

        self.gridLayout.addWidget(self.btn_system_info, 4, 0, 1, 1)

        self.label = QLabel(AboutDialog)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setTextFormat(Qt.AutoText)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)

        self.label_copyright = QLabel(AboutDialog)
        self.label_copyright.setObjectName(u"label_copyright")

        self.gridLayout.addWidget(self.label_copyright, 2, 0, 1, 1)


        self.retranslateUi(AboutDialog)
        self.buttonBox.accepted.connect(AboutDialog.accept)
        self.buttonBox.rejected.connect(AboutDialog.reject)

        QMetaObject.connectSlotsByName(AboutDialog)
    # setupUi

    def retranslateUi(self, AboutDialog):
        AboutDialog.setWindowTitle(QCoreApplication.translate("AboutDialog", u"About WISER", None))
        self.textEdit.setHtml(QCoreApplication.translate("AboutDialog", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'.AppleSystemUIFont'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">TODO: Text about the software, its website, where it came from, libraries used, other resources.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Bug reports are welcome!</span>  When submitting a bug report, please include the information reported by the &quot;System Info&quot; dialog below.  This will help us identify issues that may be related to platform and library versions.  Thanks!</p></body></html>", None))
        self.label_version.setText(QCoreApplication.translate("AboutDialog", u"TODO:  Version!", None))
        self.btn_system_info.setText(QCoreApplication.translate("AboutDialog", u"System Info", None))
        self.label.setText(QCoreApplication.translate("AboutDialog", u"WISER:  The Workbench for Imaging Spectroscopy Exploration and Research", None))
        self.label_copyright.setText(QCoreApplication.translate("AboutDialog", u"TODO:  Copyright!", None))
    # retranslateUi

