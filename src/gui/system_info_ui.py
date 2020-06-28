# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'system_info.ui'
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


class Ui_SystemInfoDialog(object):
    def setupUi(self, SystemInfoDialog):
        if not SystemInfoDialog.objectName():
            SystemInfoDialog.setObjectName(u"SystemInfoDialog")
        SystemInfoDialog.resize(400, 300)
        SystemInfoDialog.setSizeGripEnabled(True)
        SystemInfoDialog.setModal(True)
        self.gridLayout = QGridLayout(SystemInfoDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.te_system_info = QPlainTextEdit(SystemInfoDialog)
        self.te_system_info.setObjectName(u"te_system_info")
        self.te_system_info.setReadOnly(True)

        self.gridLayout.addWidget(self.te_system_info, 0, 0, 1, 1)

        self.buttonBox = QDialogButtonBox(SystemInfoDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Close)

        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 1)

        self.btn_copy_to_clipboard = QPushButton(SystemInfoDialog)
        self.btn_copy_to_clipboard.setObjectName(u"btn_copy_to_clipboard")

        self.gridLayout.addWidget(self.btn_copy_to_clipboard, 1, 0, 1, 1)


        self.retranslateUi(SystemInfoDialog)
        self.buttonBox.accepted.connect(SystemInfoDialog.accept)
        self.buttonBox.rejected.connect(SystemInfoDialog.reject)

        QMetaObject.connectSlotsByName(SystemInfoDialog)
    # setupUi

    def retranslateUi(self, SystemInfoDialog):
        SystemInfoDialog.setWindowTitle(QCoreApplication.translate("SystemInfoDialog", u"System Information", None))
        self.btn_copy_to_clipboard.setText(QCoreApplication.translate("SystemInfoDialog", u"Copy to Clipboard", None))
    # retranslateUi

