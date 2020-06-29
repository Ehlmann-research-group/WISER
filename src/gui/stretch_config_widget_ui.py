# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'stretch_config_widget.ui'
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


class Ui_StretchConfigWidget(object):
    def setupUi(self, StretchConfigWidget):
        if not StretchConfigWidget.objectName():
            StretchConfigWidget.setObjectName(u"StretchConfigWidget")
        StretchConfigWidget.resize(570, 160)
        StretchConfigWidget.setMinimumSize(QSize(570, 160))
        self.gridLayoutWidget = QWidget(StretchConfigWidget)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(0, 0, 571, 161))
        self.grid_layout_widget = QGridLayout(self.gridLayoutWidget)
        self.grid_layout_widget.setObjectName(u"grid_layout_widget")
        self.grid_layout_widget.setHorizontalSpacing(-1)
        self.grid_layout_widget.setContentsMargins(0, 0, 0, 0)
        self.groupbox_stretch = QGroupBox(self.gridLayoutWidget)
        self.groupbox_stretch.setObjectName(u"groupbox_stretch")
        self.gridLayoutWidget_2 = QWidget(self.groupbox_stretch)
        self.gridLayoutWidget_2.setObjectName(u"gridLayoutWidget_2")
        self.gridLayoutWidget_2.setGeometry(QRect(10, 30, 211, 121))
        self.grid_layout_stretch = QGridLayout(self.gridLayoutWidget_2)
        self.grid_layout_stretch.setSpacing(4)
        self.grid_layout_stretch.setObjectName(u"grid_layout_stretch")
        self.grid_layout_stretch.setContentsMargins(0, 0, 0, 0)
        self.button_linear_2_5 = QPushButton(self.gridLayoutWidget_2)
        self.button_linear_2_5.setObjectName(u"button_linear_2_5")

        self.grid_layout_stretch.addWidget(self.button_linear_2_5, 2, 1, 1, 1)

        self.rb_stretch_linear = QRadioButton(self.gridLayoutWidget_2)
        self.rb_stretch_linear.setObjectName(u"rb_stretch_linear")

        self.grid_layout_stretch.addWidget(self.rb_stretch_linear, 1, 0, 1, 3)

        self.rb_stretch_equalize = QRadioButton(self.gridLayoutWidget_2)
        self.rb_stretch_equalize.setObjectName(u"rb_stretch_equalize")

        self.grid_layout_stretch.addWidget(self.rb_stretch_equalize, 4, 0, 1, 3)

        self.button_linear_5_0 = QPushButton(self.gridLayoutWidget_2)
        self.button_linear_5_0.setObjectName(u"button_linear_5_0")

        self.grid_layout_stretch.addWidget(self.button_linear_5_0, 3, 1, 1, 1)

        self.rb_stretch_none = QRadioButton(self.gridLayoutWidget_2)
        self.rb_stretch_none.setObjectName(u"rb_stretch_none")

        self.grid_layout_stretch.addWidget(self.rb_stretch_none, 0, 0, 1, 3)

        self.grid_layout_stretch.setColumnMinimumWidth(0, 10)

        self.grid_layout_widget.addWidget(self.groupbox_stretch, 0, 1, 1, 1)

        self.groupbox_conditioner = QGroupBox(self.gridLayoutWidget)
        self.groupbox_conditioner.setObjectName(u"groupbox_conditioner")
        self.gridLayoutWidget_3 = QWidget(self.groupbox_conditioner)
        self.gridLayoutWidget_3.setObjectName(u"gridLayoutWidget_3")
        self.gridLayoutWidget_3.setGeometry(QRect(10, 30, 171, 71))
        self.grid_layout_cond = QGridLayout(self.gridLayoutWidget_3)
        self.grid_layout_cond.setSpacing(4)
        self.grid_layout_cond.setObjectName(u"grid_layout_cond")
        self.grid_layout_cond.setContentsMargins(0, 0, 0, 0)
        self.rb_cond_log = QRadioButton(self.gridLayoutWidget_3)
        self.rb_cond_log.setObjectName(u"rb_cond_log")

        self.grid_layout_cond.addWidget(self.rb_cond_log, 2, 0, 1, 1)

        self.rb_cond_sqrt = QRadioButton(self.gridLayoutWidget_3)
        self.rb_cond_sqrt.setObjectName(u"rb_cond_sqrt")

        self.grid_layout_cond.addWidget(self.rb_cond_sqrt, 1, 0, 1, 1)

        self.rb_cond_none = QRadioButton(self.gridLayoutWidget_3)
        self.rb_cond_none.setObjectName(u"rb_cond_none")

        self.grid_layout_cond.addWidget(self.rb_cond_none, 0, 0, 1, 1)


        self.grid_layout_widget.addWidget(self.groupbox_conditioner, 0, 2, 1, 1)


        self.retranslateUi(StretchConfigWidget)

        QMetaObject.connectSlotsByName(StretchConfigWidget)
    # setupUi

    def retranslateUi(self, StretchConfigWidget):
        StretchConfigWidget.setWindowTitle(QCoreApplication.translate("StretchConfigWidget", u"Form", None))
        self.groupbox_stretch.setTitle(QCoreApplication.translate("StretchConfigWidget", u"Stretch", None))
#if QT_CONFIG(tooltip)
        self.button_linear_2_5.setToolTip(QCoreApplication.translate("StretchConfigWidget", u"Apply a 2.5% linear stretch to all channels", None))
#endif // QT_CONFIG(tooltip)
        self.button_linear_2_5.setText(QCoreApplication.translate("StretchConfigWidget", u"2.5% linear", None))
#if QT_CONFIG(tooltip)
        self.rb_stretch_linear.setToolTip(QCoreApplication.translate("StretchConfigWidget", u"Apply a linear stretch", None))
#endif // QT_CONFIG(tooltip)
        self.rb_stretch_linear.setText(QCoreApplication.translate("StretchConfigWidget", u"Linear Stretch", None))
#if QT_CONFIG(tooltip)
        self.rb_stretch_equalize.setToolTip(QCoreApplication.translate("StretchConfigWidget", u"Apply an equalization stretch", None))
#endif // QT_CONFIG(tooltip)
        self.rb_stretch_equalize.setText(QCoreApplication.translate("StretchConfigWidget", u"Equalize Stretch", None))
#if QT_CONFIG(tooltip)
        self.button_linear_5_0.setToolTip(QCoreApplication.translate("StretchConfigWidget", u"Apply a 5% linear stretch to all channels", None))
#endif // QT_CONFIG(tooltip)
        self.button_linear_5_0.setText(QCoreApplication.translate("StretchConfigWidget", u"5% linear", None))
#if QT_CONFIG(tooltip)
        self.rb_stretch_none.setToolTip(QCoreApplication.translate("StretchConfigWidget", u"No contrast stretch will be applied", None))
#endif // QT_CONFIG(tooltip)
        self.rb_stretch_none.setText(QCoreApplication.translate("StretchConfigWidget", u"Full Linear Stretch", None))
        self.groupbox_conditioner.setTitle(QCoreApplication.translate("StretchConfigWidget", u"Conditioner", None))
#if QT_CONFIG(tooltip)
        self.rb_cond_log.setToolTip(QCoreApplication.translate("StretchConfigWidget", u"Apply a logarithmic conditioner before applying stretch", None))
#endif // QT_CONFIG(tooltip)
        self.rb_cond_log.setText(QCoreApplication.translate("StretchConfigWidget", u"Logarithmic", None))
#if QT_CONFIG(tooltip)
        self.rb_cond_sqrt.setToolTip(QCoreApplication.translate("StretchConfigWidget", u"Apply a square-root conditioner before applying stretch", None))
#endif // QT_CONFIG(tooltip)
        self.rb_cond_sqrt.setText(QCoreApplication.translate("StretchConfigWidget", u"Square root", None))
#if QT_CONFIG(tooltip)
        self.rb_cond_none.setToolTip(QCoreApplication.translate("StretchConfigWidget", u"Use no conditioner before applying stretch", None))
#endif // QT_CONFIG(tooltip)
        self.rb_cond_none.setText(QCoreApplication.translate("StretchConfigWidget", u"None", None))
    # retranslateUi

