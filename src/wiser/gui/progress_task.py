"""
Run any scheduler-submitted function with live progress in a blocking modal.

:func:`run_with_progress` is the reusable entry point: give it a function ``fn`` that
accepts a ``progress`` keyword (a :class:`~wiser.utils.progress.ProgressReporter`),
and it will

  * show a :class:`~wiser.gui.progress_dialog.ProgressDialog` over ``parent``,
  * register an Activity Monitor row,
  * submit ``fn`` to the work scheduler's thread pool with a reporter injected, and
  * drive both the dialog and the Activity Monitor from the reporter, invoking
    ``on_success`` / ``on_error`` on the GUI thread when the task finishes.

The reporter's sink is a thread-safe Qt signal emit, so ``fn`` may report progress
from the worker thread while the slots that touch widgets run on the GUI thread.

This is the thread-scoped path (see ``FOLLOWUP_io_work_units.md``); it does not use
the work-unit RAM accounting.
"""

from concurrent.futures import Future
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget

from wiser.utils.primitives import PriorityClass
from wiser.utils.progress import ProgressReporter
from wiser.utils.task_system import ProgressUpdate

from .progress_dialog import PROGRESS_RESOLUTION, ProgressDialog


class _ProgressTaskRunner(QObject):
    """
    Owns one in-flight task: marshals progress and completion from the worker thread
    to the GUI thread, updates the dialog + Activity Monitor row, and cleans up.

    Parented to the task's parent widget so it outlives the background work
    regardless of when the caller drops its reference.
    """

    # Emitted from the worker thread; queued onto the GUI thread.
    progressed = Signal(float, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        parent: Optional[QObject],
        app_services: Any,
        dialog: ProgressDialog,
        activity_id: int,
        on_success: Optional[Callable[[Any], None]],
        on_error: Optional[Callable[[str], None]],
        block_window: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_services = app_services
        self._dialog = dialog
        self._activity_id = activity_id
        self._on_success = on_success
        self._on_error = on_error
        # The window whose input we suppress while the task runs (disabled on start,
        # re-enabled on completion). None means nothing is blocked.
        self._block_window = block_window

        self.progressed.connect(self._on_progress)
        self.succeeded.connect(self._on_succeeded)
        self.failed.connect(self._on_failed)

    # -- sink + future callback (called from the worker thread) ---------------

    def report(self, fraction: float, message: str) -> None:
        """Progress sink handed to the :class:`ProgressReporter` (worker thread)."""
        self.progressed.emit(fraction, message)

    def on_future_done(self, future: Future) -> None:
        """Future done-callback (worker/pool thread): funnel result to the GUI thread."""
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user via `failed`
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)

    # -- GUI-thread slots -----------------------------------------------------

    def _on_progress(self, fraction: float, message: str) -> None:
        self._dialog.set_progress(fraction, message)
        numerator = int(round(max(0.0, min(1.0, fraction)) * PROGRESS_RESOLUTION))
        update = ProgressUpdate(current_iteration=numerator, total_iterations=PROGRESS_RESOLUTION)
        self._app_services.activity_monitor.progress_update.emit((self._activity_id, update))

    def _on_succeeded(self, result: object) -> None:
        self._app_services.activity_monitor.set_task_finished(self._activity_id)
        self._dialog.finish()
        if self._on_success is not None:
            self._on_success(result)
        self._cleanup()

    def _on_failed(self, message: str) -> None:
        self._app_services.activity_monitor.set_task_failed(self._activity_id, message)
        self._dialog.finish()
        if self._on_error is not None:
            self._on_error(message)
        self._cleanup()

    def _cleanup(self) -> None:
        # Re-enable the window we suppressed so the user can interact with it again.
        if self._block_window is not None:
            self._block_window.setEnabled(True)
        self._dialog.deleteLater()
        self.deleteLater()


def run_with_progress(
    app_services: Any,
    block_window: Any,
    title: str,
    fn: Callable[..., Any],
    *args: Any,
    on_success: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    description: str = "",
    meta: Optional[Dict[str, str]] = None,
    priority: PriorityClass = PriorityClass.BACKGROUND,
    **kwargs: Any,
) -> _ProgressTaskRunner:
    """
    Run ``fn(*args, progress=reporter, **kwargs)`` on the work scheduler while showing a
    progress dialog and a mirrored Activity Monitor row, calling ``on_success`` /
    ``on_error`` on the GUI thread when it finishes.

    Input is suppressed by disabling ``block_window`` for the duration of the task (and
    re-enabling it on completion), so *only* that one window is frozen while the rest of
    the app stays usable. This is deliberately not ``Qt.WindowModal``: that would also
    freeze every ancestor window, which for the mosaic would wrongly lock all of WISER.

    ``fn`` must accept a ``progress`` keyword (a :class:`ProgressReporter`). Returns the
    runner that owns the dialog until completion; the caller may ignore it (it is kept
    alive by its Qt parent and the future callback).
    """
    # The window we disable while the task runs.
    target = block_window.window() if isinstance(block_window, QWidget) else None

    # Pick who the progress dialog is parented to. It must NOT be parented *under*
    # `target`: Qt propagates setEnabled(False) to child windows, so a dialog owned by
    # the window we disable would be disabled (unclickable) too.
    owner = target.parentWidget() if target is not None else None
    dialog_parent = owner.window() if owner is not None else target

    dialog = ProgressDialog(dialog_parent, title=title, description=description)
    activity_id = app_services.activity_monitor.register_task(
        title=title, meta=meta or {}, cancel_callback=None
    )
    runner = _ProgressTaskRunner(
        dialog_parent, app_services, dialog, activity_id, on_success, on_error, target
    )
    reporter = ProgressReporter(sink=runner.report)

    if target is not None:
        target.setEnabled(False)
    dialog.show()
    dialog.raise_()
    future = app_services.scheduler.submit_thread(priority, fn, *args, progress=reporter, **kwargs)
    future.add_done_callback(runner.on_future_done)
    return runner
