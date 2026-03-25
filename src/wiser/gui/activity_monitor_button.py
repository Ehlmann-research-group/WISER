from __future__ import annotations

from typing import Optional

from PySide2.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide2.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide2.QtWidgets import QToolButton, QWidget

from .activity_monitor import ActivityMonitorDialog


class ActivityMonitorButton(QToolButton):
    def __init__(
        self,
        activity_monitor: ActivityMonitorDialog,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._activity_monitor = activity_monitor
        self._has_active_tasks = False
        self._spinner_angle = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(80)
        self._spinner_timer.timeout.connect(self._advance_spinner)

        self.setAutoRaise(True)
        self.setIconSize(QSize(24, 24))
        self.setToolTip(self.tr("Show activity monitor"))
        self.setIcon(self._build_icon())
        self.clicked.connect(self.show_activity_monitor)

        self._activity_monitor.active_task_count_changed.connect(self._on_active_task_count_changed)
        self._on_active_task_count_changed(self._activity_monitor.get_active_task_count())

    def show_activity_monitor(self) -> None:
        self._activity_monitor.show()
        self._activity_monitor.raise_()
        self._activity_monitor.activateWindow()

    def _advance_spinner(self) -> None:
        self._spinner_angle = (self._spinner_angle + 30) % 360
        self.setIcon(self._build_icon())

    def _on_active_task_count_changed(self, active_task_count: int) -> None:
        self._has_active_tasks = active_task_count > 0

        if self._has_active_tasks:
            self.setToolTip(self.tr(f"Show activity monitor ({active_task_count} active task(s))"))
            if not self._spinner_timer.isActive():
                self._spinner_timer.start()
        else:
            self.setToolTip(self.tr("Show activity monitor"))
            self._spinner_timer.stop()
            self._spinner_angle = 0

        self.setIcon(self._build_icon())

    def _build_icon(self) -> QIcon:
        size = 24
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)

        panel_rect = QRectF(2.0, 3.0, 16.0, 18.0)
        painter.setPen(QPen(QColor("#1F1F1F"), 1.4))
        painter.setBrush(QColor("#EFEFEF"))
        painter.drawRect(panel_rect)

        painter.fillRect(QRectF(4.5, 6.0, 11.0, 2.0), QColor("#3B3B3B"))
        for y in (10.0, 13.5, 17.0):
            painter.fillRect(QRectF(4.5, y, 8.8, 1.5), QColor("#8A8A8A"))

        if self._has_active_tasks:
            self._paint_spinner_badge(painter)

        painter.end()
        return QIcon(pixmap)

    def _paint_spinner_badge(self, painter: QPainter) -> None:
        center = QPointF(18.0, 18.0)
        outer_radius = 4.6
        inner_radius = 2.1

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#F5F5F5"))
        painter.drawEllipse(center, outer_radius + 1.1, outer_radius + 1.1)

        for index in range(12):
            alpha = int(70 + (185 * ((index + 1) / 12.0)))
            painter.save()
            painter.translate(center)
            painter.rotate(self._spinner_angle - (index * 30))
            painter.setBrush(QColor(35, 35, 35, alpha))
            painter.drawRoundedRect(
                QRectF(inner_radius, -0.65, outer_radius - inner_radius, 1.3),
                0.6,
                0.6,
            )
            painter.restore()
