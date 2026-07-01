"""
A minimal non-modal progress dialog (title + description + progress bar).

:class:`ProgressDialog` is shown *non-blocking* (via ``show()``, not ``exec()``) so
the event loop keeps running and the bar updates in real time. It is deliberately
*not* window-modal: input suppression is done by the owning task, which disables only
the single window the work belongs to (leaving the rest of the app interactive)
rather than blocking the whole window hierarchy the way ``Qt.WindowModal`` would. The
dialog itself is dumb: it renders whatever :meth:`set_progress` is told. The
orchestration that feeds it (and the Activity Monitor, and the disable/re-enable of
the owning window) lives in :mod:`wiser.gui.progress_task`.

Closing it (the window's close button or Escape) does not dismiss it directly;
instead it emits :attr:`cancel_requested` so the owning task can cancel the work and
tear down. The owner closes the dialog via :meth:`finish` once the task ends (whether
it completed or was cancelled).
"""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout, QWidget

# The progress bar runs on 0..PROGRESS_RESOLUTION so a 0..1 fraction maps to a smooth
# integer bar (and reads as a percentage by default).
PROGRESS_RESOLUTION = 1000


class ProgressDialog(QDialog):
    """Non-modal dialog with a title, a description line, and a progress bar."""

    # Emitted when the user tries to close the dialog (close button or Escape) while a
    # task is running; the owning task listens for this to cancel the work.
    cancel_requested = Signal()

    def __init__(
        self,
        parent: Optional[QWidget],
        title: str,
        description: str = "",
    ) -> None:
        super().__init__(parent)
        self._finished = False
        # Guards against emitting cancel_requested more than once per run.
        self._cancel_requested = False

        self.setWindowTitle(title)
        # Non-modal: the owning task (run_with_progress) suppresses input by disabling
        # just the specific window the work belongs to, so the rest of the app stays
        # interactive. Shown via show() so the event loop keeps updating the bar.
        self.setWindowModality(Qt.NonModal)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)

        self._title_label = QLabel(title, self)
        self._title_label.setStyleSheet("font-weight: 600;")

        self._description_label = QLabel(description, self)
        self._description_label.setWordWrap(True)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, PROGRESS_RESOLUTION)
        self._progress_bar.setValue(0)

        layout.addWidget(self._title_label)
        layout.addWidget(self._description_label)
        layout.addWidget(self._progress_bar)

    def set_progress(self, fraction: float, message: str = "") -> None:
        """Set the bar from a 0..1 fraction and, if given, update the description."""
        clamped = max(0.0, min(1.0, fraction))
        self._progress_bar.setValue(int(round(clamped * PROGRESS_RESOLUTION)))
        if message:
            self._description_label.setText(message)

    def finish(self) -> None:
        """Mark the task done and dismiss the dialog (bypasses the close guards)."""
        self._finished = True
        self.accept()

    # -- cancellation (close button / Escape while a task runs) ----------------

    def _request_cancel(self) -> None:
        """Ask the owning task to cancel; the owner closes the dialog via finish()."""
        if self._finished or self._cancel_requested:
            return
        self._cancel_requested = True
        self.cancel_requested.emit()

    def reject(self) -> None:
        # Escape triggers reject(). Once finished, close normally; otherwise treat it
        # as a cancel request and leave the dialog up for the owner to dismiss.
        if self._finished:
            super().reject()
        else:
            self._request_cancel()

    def closeEvent(self, event) -> None:
        # Same for the window close button: request cancellation and keep the dialog
        # visible until the owner tears the task down and calls finish().
        if self._finished:
            super().closeEvent(event)
        else:
            self._request_cancel()
            event.ignore()
