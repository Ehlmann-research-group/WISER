# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'channel_stretch_widget.ui'
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


class Ui_ChannelStretchWidget(object):
    def setupUi(self, ChannelStretchWidget):
        if not ChannelStretchWidget.objectName():
            ChannelStretchWidget.setObjectName(u"ChannelStretchWidget")
        ChannelStretchWidget.resize(570, 240)
        ChannelStretchWidget.setMinimumSize(QSize(570, 240))
        self.groupbox_channel = QGroupBox(ChannelStretchWidget)
        self.groupbox_channel.setObjectName(u"groupbox_channel")
        self.groupbox_channel.setGeometry(QRect(0, 0, 571, 241))
        self.gridLayoutWidget = QWidget(self.groupbox_channel)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(10, 30, 551, 201))
        self.grid_layout = QGridLayout(self.gridLayoutWidget)
        self.grid_layout.setSpacing(4)
        self.grid_layout.setObjectName(u"grid_layout")
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.lineedit_stretch_high = QLineEdit(self.gridLayoutWidget)
        self.lineedit_stretch_high.setObjectName(u"lineedit_stretch_high")

        self.grid_layout.addWidget(self.lineedit_stretch_high, 6, 1, 1, 1)

        self.lineedit_stretch_low = QLineEdit(self.gridLayoutWidget)
        self.lineedit_stretch_low.setObjectName(u"lineedit_stretch_low")

        self.grid_layout.addWidget(self.lineedit_stretch_low, 5, 1, 1, 1)

        self.button_apply_bounds = QPushButton(self.gridLayoutWidget)
        self.button_apply_bounds.setObjectName(u"button_apply_bounds")

        self.grid_layout.addWidget(self.button_apply_bounds, 2, 1, 1, 1)

        self.slider_stretch_low = QSlider(self.gridLayoutWidget)
        self.slider_stretch_low.setObjectName(u"slider_stretch_low")
        self.slider_stretch_low.setMaximum(100)
        self.slider_stretch_low.setOrientation(Qt.Horizontal)
        self.slider_stretch_low.setInvertedAppearance(False)

        self.grid_layout.addWidget(self.slider_stretch_low, 5, 2, 1, 1)

        self.label_stretch_high = QLabel(self.gridLayoutWidget)
        self.label_stretch_high.setObjectName(u"label_stretch_high")

        self.grid_layout.addWidget(self.label_stretch_high, 6, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.grid_layout.addItem(self.verticalSpacer, 4, 0, 1, 2)

        self.histogram_widget = QWidget(self.gridLayoutWidget)
        self.histogram_widget.setObjectName(u"histogram_widget")

        self.grid_layout.addWidget(self.histogram_widget, 0, 2, 5, 1)

        self.slider_stretch_high = QSlider(self.gridLayoutWidget)
        self.slider_stretch_high.setObjectName(u"slider_stretch_high")
        self.slider_stretch_high.setOrientation(Qt.Horizontal)

        self.grid_layout.addWidget(self.slider_stretch_high, 6, 2, 1, 1)

        self.button_reset_bounds = QPushButton(self.gridLayoutWidget)
        self.button_reset_bounds.setObjectName(u"button_reset_bounds")

        self.grid_layout.addWidget(self.button_reset_bounds, 3, 1, 1, 1)

        self.lineedit_max_bound = QLineEdit(self.gridLayoutWidget)
        self.lineedit_max_bound.setObjectName(u"lineedit_max_bound")

        self.grid_layout.addWidget(self.lineedit_max_bound, 1, 1, 1, 1)

        self.label_stretch_low = QLabel(self.gridLayoutWidget)
        self.label_stretch_low.setObjectName(u"label_stretch_low")

        self.grid_layout.addWidget(self.label_stretch_low, 5, 0, 1, 1)

        self.lineedit_min_bound = QLineEdit(self.gridLayoutWidget)
        self.lineedit_min_bound.setObjectName(u"lineedit_min_bound")

        self.grid_layout.addWidget(self.lineedit_min_bound, 0, 1, 1, 1)

        self.label_max_bound = QLabel(self.gridLayoutWidget)
        self.label_max_bound.setObjectName(u"label_max_bound")

        self.grid_layout.addWidget(self.label_max_bound, 1, 0, 1, 1)

        self.label_min_bound = QLabel(self.gridLayoutWidget)
        self.label_min_bound.setObjectName(u"label_min_bound")

        self.grid_layout.addWidget(self.label_min_bound, 0, 0, 1, 1)

        self.grid_layout.setRowStretch(4, 10)
        self.grid_layout.setColumnStretch(2, 10)
#if QT_CONFIG(shortcut)
        self.label_max_bound.setBuddy(self.lineedit_max_bound)
        self.label_min_bound.setBuddy(self.lineedit_min_bound)
#endif // QT_CONFIG(shortcut)

        self.retranslateUi(ChannelStretchWidget)

        QMetaObject.connectSlotsByName(ChannelStretchWidget)
    # setupUi

    def retranslateUi(self, ChannelStretchWidget):
        ChannelStretchWidget.setWindowTitle(QCoreApplication.translate("ChannelStretchWidget", u"Channel Stretch Widget", None))
        self.groupbox_channel.setTitle(QCoreApplication.translate("ChannelStretchWidget", u"Channel Name", None))
#if QT_CONFIG(tooltip)
        self.button_apply_bounds.setToolTip(QCoreApplication.translate("ChannelStretchWidget", u"Filter band data using min/max bounds before computing histogram", None))
#endif // QT_CONFIG(tooltip)
        self.button_apply_bounds.setText(QCoreApplication.translate("ChannelStretchWidget", u"Apply", None))
#if QT_CONFIG(tooltip)
        self.slider_stretch_low.setToolTip(QCoreApplication.translate("ChannelStretchWidget", u"Drag to adjust low boundary of stretch", None))
#endif // QT_CONFIG(tooltip)
        self.label_stretch_high.setText(QCoreApplication.translate("ChannelStretchWidget", u"Stretch High", None))
#if QT_CONFIG(tooltip)
        self.slider_stretch_high.setToolTip(QCoreApplication.translate("ChannelStretchWidget", u"Drag to adjust high boundary of stretch", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.button_reset_bounds.setToolTip(QCoreApplication.translate("ChannelStretchWidget", u"Reset min/max bounds to min/max values from band data", None))
#endif // QT_CONFIG(tooltip)
        self.button_reset_bounds.setText(QCoreApplication.translate("ChannelStretchWidget", u"Reset", None))
        self.label_stretch_low.setText(QCoreApplication.translate("ChannelStretchWidget", u"Stretch Low", None))
        self.label_max_bound.setText(QCoreApplication.translate("ChannelStretchWidget", u"Maximum", None))
        self.label_min_bound.setText(QCoreApplication.translate("ChannelStretchWidget", u"Minimum", None))
    # retranslateUi

