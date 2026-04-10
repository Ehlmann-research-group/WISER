"""
Frameless startup window shown while WISER loads.

Uses a normal top-level window (not QSplashScreen) so the taskbar shows an entry
on Windows. Supports optional log output when startup fails and a close control
to cancel launch.
"""

from __future__ import annotations

import logging
import traceback
from typing import Optional

from PySide2.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide2.QtGui import QFont, QIcon, QPixmap
from PySide2.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class WiserGuiImportThread(QThread):
    """
    Imports wiser.gui.app on a background thread so the event loop stays responsive.

    QWidget construction must still run on the GUI thread; only the import runs here.
    """

    succeeded = Signal(object)
    failed = Signal(str)

    def run(self) -> None:
        try:
            import wiser.gui.app as app_module
        except BaseException:
            self.failed.emit(traceback.format_exc())
            return
        self.succeeded.emit(app_module)


# Shown under the logo until startup finishes or fails.
DEFAULT_LOADING_MESSAGE = "WISER may take longer to load the\nfirst time after a fresh install."


class _AspectFitPixmapLabel(QLabel):
    """
    QLabel that rescales a pixmap to fit the current label size without clipping.

    ``setScaledContents(True)`` is *not* used: it stretches the pixmap to the full
    label rectangle and ignores aspect ratio. Here we rescale with
    ``Qt.KeepAspectRatio`` on each resize so the image fits inside and stays centered.

    ``min_size`` and ``preferred_size`` are square **edge lengths** (one number, not a
    ``QSize``) used only for ``minimumSizeHint`` / ``sizeHint`` so the layout knows
    how much space to reserve for a roughly square logo.
    """

    def __init__(
        self,
        source: QPixmap,
        *,
        preferred_size: int,
        min_size: int = 64,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._source = QPixmap(source)
        self._preferred_size = max(1, preferred_size)
        self._min_size = max(1, min_size)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def minimumSizeHint(self) -> QSize:
        m = self._min_size
        return QSize(m, m)

    def sizeHint(self) -> QSize:
        s = self._preferred_size
        return QSize(s, s)

    def _refresh_pixmap(self) -> None:
        if self._source.isNull():
            return
        # QLabel draws the pixmap inside contentsRect(), not the full geometry.
        # Using width()/height() after setContentsMargins() scales too large → slight clip.
        r = self.contentsRect()
        w = max(1, r.width())
        h = max(1, r.height())
        dpr = self.devicePixelRatioF()
        scaled = self._source.scaled(
            int(w * dpr),
            int(h * dpr),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(dpr)
        super().setPixmap(scaled)


class StartupSplash(QWidget):
    """
    Borderless window with logo, status text, optional logs, and a close button.

    Call :meth:`finish_and_show_main` after the main window is ready, or
    :meth:`set_startup_failed` if construction fails.
    """

    def __init__(
        self,
        pixmap: QPixmap,
        icon: QIcon,
        *,
        loading_message: str = DEFAULT_LOADING_MESSAGE,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._cancelled = False
        self._finished = False
        self._failed = False

        self._setup_window(icon)
        self._build_ui(pixmap, loading_message)

    _LOGO_SIZE = 300
    _WINDOW_WIDTH = 400
    _WINDOW_HEIGHT = 520

    def _setup_window(self, icon: QIcon) -> None:
        # Normal window (not Qt.SplashScreen) so the shell shows a taskbar entry on Windows.
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        # WA_StyledBackground is required for Qt to actually paint border/bg on a plain QWidget.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowTitle("WISER")
        self.setWindowIcon(icon)
        self.setFixedSize(self._WINDOW_WIDTH, self._WINDOW_HEIGHT)
        self.setStyleSheet(
            """
            StartupSplash {
                background-color: #f7f7f7;
                border: 2px solid #b0b0b0;
                border-radius: 6px;
            }
            """
        )

        # Drop shadow gives depth without relying on OS window decorations.
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(Qt.gray)
        self.setGraphicsEffect(shadow)

    def _build_ui(self, pixmap: QPixmap, loading_message: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 12)
        root.setSpacing(0)

        # ── Header row: close button pinned to top-right ────────────────────
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addStretch()
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setToolTip("Cancel startup and exit")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(
            """
            QPushButton {
                border: none;
                border-radius: 4px;
                font-size: 18px;
                color: #777;
                background: transparent;
            }
            QPushButton:hover { background-color: #e0e0e0; color: #111; }
            QPushButton:pressed { background-color: #d0d0d0; }
            """
        )
        close_btn.clicked.connect(self._on_close_clicked)
        header.addWidget(close_btn)
        root.addLayout(header)

        # ── Logo (row stretches; pixmap scales to fit label, aspect preserved) ─
        logo = _AspectFitPixmapLabel(pixmap, preferred_size=self._LOGO_SIZE)
        logo.setContentsMargins(0, 6, 0, 8)
        root.addWidget(logo, stretch=1)

        # ── Status / hint text ───────────────────────────────────────────────
        self._message_label = QLabel(loading_message)
        self._message_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self._message_label.setWordWrap(True)
        msg_font = QFont(self._message_label.font())
        msg_font.setPointSize(max(msg_font.pointSize(), 10))
        self._message_label.setFont(msg_font)
        self._message_label.setStyleSheet("color: #333;")
        root.addWidget(self._message_label)

        root.addSpacing(12)

        # ── Divider ──────────────────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        divider.setStyleSheet("color: #ccc;")
        root.addWidget(divider)

        root.addSpacing(8)

        # ── Logs ─────────────────────────────────────────────────────────────
        logs_title = QLabel("Logs")
        logs_title_font = QFont(logs_title.font())
        logs_title_font.setBold(True)
        logs_title.setFont(logs_title_font)
        logs_title.setStyleSheet("color: #444;")
        root.addWidget(logs_title)

        root.addSpacing(4)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setPlaceholderText("Startup messages appear here if something goes wrong.")
        # Preferred (not Expanding): avoid consuming all leftover height; scroll inside.
        self._log_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._log_view.setMinimumHeight(40)
        self._log_view.setMaximumHeight(80)
        self._log_view.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self._log_view.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 9pt;
            }
            """
        )
        root.addWidget(self._log_view)

    def _on_close_clicked(self) -> None:
        self.close()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._center_on_primary_screen()

    def _center_on_primary_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        fg = self.frameGeometry()
        fg.moveCenter(screen.availableGeometry().center())
        self.move(fg.topLeft())

    def append_log(self, text: str) -> None:
        """Append text to the log pane (does not imply failure)."""
        self._log_view.appendPlainText(text.rstrip())

    def set_startup_failed(self, message: str) -> None:
        """Show a failure state: message in logs and a visible error hint."""
        self._finished = True
        self._failed = True
        self.append_log(message)
        self._message_label.setText("WISER could not start. See Logs below for details.")
        self._message_label.setStyleSheet("color: #a40000; font-weight: bold;")
        self._log_view.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #fff8f8;
                border: 1px solid #c04040;
                border-radius: 3px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 9pt;
            }
            """
        )
        logger.error("Startup failed; details written to splash log view.")

    def finish_and_show_main(self, main_window: QMainWindow) -> None:
        """Hide this splash and show the main application window."""
        if self._cancelled:
            return
        self._finished = True
        self.hide()
        self.deleteLater()
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()

    def is_cancelled(self) -> bool:
        return self._cancelled

    def is_finished(self) -> bool:
        """True after success (:meth:`finish_and_show_main`) or :meth:`set_startup_failed`."""
        return self._finished

    def closeEvent(self, event) -> None:
        app = QApplication.instance()
        if not self._finished:
            self._cancelled = True
            logger.info("Startup cancelled by user from splash window.")
        super().closeEvent(event)
        # After a failed startup, closing the splash should exit the process.
        # During load, cancel should exit. (On success we hide() rather than close().)
        if self._cancelled or self._failed:
            QTimer.singleShot(0, lambda: app.quit() if app is not None else None)
