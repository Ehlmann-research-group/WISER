"""
One progress dialog for a render that spans every raster pane.

A stretch change or a dataset open re-renders the main view, the context pane
and the zoom pane one after another, all on the GUI thread.  On a large image
that takes seconds, so the change runs under a :class:`RenderProgress`: each
``RasterView`` render it reaches reports its work in steps (a band read, a row
strip of the stretch or of the RGB packing, a stage of a joint stretch) to the
one dialog, and checks between steps whether the user cancelled.

The dialog appears only once the work has taken about ``MINIMUM_DURATION_MS``,
so small images never flash it.  Once shown, the dialog processes events at each
step, which keeps the application repainting and lets the Cancel button work.

Cancelling is all-or-nothing: every ``RasterView`` that a render under the
progress touched is restored to its state from before the change, the
render-cache entries those renders added are removed, and the code
that opened the progress (for example ``ApplicationState.set_stretches``) undoes
its own state change, so no pane shows a partial or mismatched image.
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QProgressDialog

# Work shorter than this never shows the dialog.
MINIMUM_DURATION_MS = 500


class RenderCancelled(Exception):
    """Raised at a render step after the user cancelled the active render."""


class _ProgressDialog(QProgressDialog):
    """A progress dialog that Escape cannot dismiss when it has no Cancel button."""

    def __init__(self, label: str, cancel_text: Optional[str], parent):
        super().__init__(label, cancel_text, 0, 0, parent)
        self._cancellable = cancel_text is not None

    def reject(self) -> None:
        if self._cancellable:
            super().reject()


class RenderProgress:
    """
    The progress of one render that may span several raster views.  Create it
    through :func:`render_progress`; a ``RasterView`` finds the active one with
    :meth:`active`.
    """

    _active: Optional["RenderProgress"] = None

    def __init__(self, label: str, cancellable: bool):
        self._cancellable = cancellable
        self._cancelled = False
        self._total = 0
        self._done = 0
        # id(view) -> (view, state captured before this render changed it)
        self._snapshots: Dict[int, Any] = {}
        # (cache, key) of each render-cache entry a render under this progress added
        self._cache_items: List[Tuple[Any, int]] = []

        cancel_text = QApplication.translate("RenderProgress", "Cancel") if cancellable else None
        self._dialog = _ProgressDialog(label, cancel_text, QApplication.activeWindow())
        self._dialog.setWindowModality(Qt.ApplicationModal)
        self._dialog.setMinimumDuration(MINIMUM_DURATION_MS)
        self._dialog.setAutoClose(False)
        self._dialog.setAutoReset(False)
        if not cancellable:
            self._dialog.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self._dialog.setValue(0)

    @classmethod
    def active(cls) -> Optional["RenderProgress"]:
        """The render progress currently open, or ``None``."""
        return cls._active

    def track(self, view) -> None:
        """
        Record ``view``'s render state before the first change this progress
        makes to it, so a cancel can restore it.  ``view`` must provide
        ``_get_render_state()`` and ``_restore_render_state(state)``.
        """
        if id(view) not in self._snapshots:
            self._snapshots[id(view)] = (view, view._get_render_state())

    def track_cache_item(self, cache, key: int) -> None:
        """Record a render-cache entry this render added, so a cancel can remove it."""
        self._cache_items.append((cache, key))

    def add_steps(self, count: int) -> None:
        """Add ``count`` steps to the total work."""
        self._total += count
        self._dialog.setMaximum(self._total)

    def check(self) -> None:
        """Raise :class:`RenderCancelled` if the user has cancelled."""
        if self._cancelled:
            raise RenderCancelled()
        if self._cancellable and self._dialog.wasCanceled():
            self._cancelled = True
            raise RenderCancelled()

    def step(self) -> None:
        """
        Mark one step done, then raise :class:`RenderCancelled` if the user has
        cancelled.
        """
        self._done += 1
        self._dialog.setValue(min(self._done, self._total))
        self.check()

    def was_cancelled(self) -> bool:
        return self._cancelled or (self._cancellable and self._dialog.wasCanceled())

    def _finish(self) -> None:
        if self.was_cancelled():
            self._cancelled = True
            for view, state in reversed(list(self._snapshots.values())):
                view._restore_render_state(state)
            for cache, key in self._cache_items:
                cache.remove_cache_item(key)
        self._dialog.reset()
        self._dialog.hide()
        self._dialog.deleteLater()


@contextmanager
def render_progress(label: str, cancellable: bool) -> Iterator[RenderProgress]:
    """
    Run the body as one render with one progress dialog.  When a render progress
    is already open, the body joins it and ``label`` and ``cancellable`` are
    ignored.  With ``cancellable`` False the dialog has no Cancel button and the
    render always runs to completion.  On leaving, a cancelled render restores
    every view it tracked;
    check :meth:`RenderProgress.was_cancelled` to undo other state the body
    changed.
    """
    outer = RenderProgress.active()
    if outer is not None:
        yield outer
        return

    progress = RenderProgress(label, cancellable)
    RenderProgress._active = progress
    try:
        yield progress
    finally:
        RenderProgress._active = None
        progress._finish()
