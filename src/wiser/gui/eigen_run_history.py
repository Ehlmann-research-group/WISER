"""Shared base class for PCA/MNF (and future eigen-decomposition) run history.

PCA and MNF each have their own concrete record type so the two histories are
never accidentally conflated, but the manager logic — store records, emit
``records_changed`` on add/remove/dataset_removed, expose liveness/status of
the input dataset — is identical between them.  This module factors that
shared logic into :class:`EigenRunHistoryManagerBase` so the per-task-type
managers (see :mod:`wiser.gui.pca_history` and :mod:`wiser.gui.mnf_history`)
are tiny shells declaring their record type.
"""

from __future__ import annotations

from typing import Generic, List, Protocol, TYPE_CHECKING, TypeVar

from PySide2.QtCore import QObject, Signal

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState


class _HasRunIdAndInputDataset(Protocol):
    """The minimal shape a record must have for the base manager to use it."""

    run_id: int
    input_dataset_id: int


R = TypeVar("R", bound=_HasRunIdAndInputDataset)


class EigenRunHistoryManagerBase(QObject, Generic[R]):
    """Generic in-memory list of completed eigen-decomposition runs.

    Subclass and bind ``R`` to your concrete record type.  The subclass
    typically doesn't need to override anything — just inherit and
    instantiate.
    """

    records_changed = Signal()

    def __init__(self, app_state: "ApplicationState") -> None:
        super().__init__()
        self._app_state = app_state
        self._records: List[R] = []
        # Any dataset removal could flip liveness for any record; we just
        # re-emit unconditionally and let the view re-render.  Re-rendering
        # N rows is cheap for typical N.
        app_state.dataset_removed.connect(self._on_dataset_removed)

    # ----- mutation -----

    def add_record(self, record: R) -> None:
        self._records.append(record)
        self.records_changed.emit()

    def remove_record(self, run_id: int) -> None:
        before = len(self._records)
        self._records = [r for r in self._records if r.run_id != run_id]
        if len(self._records) != before:
            self.records_changed.emit()

    # ----- read -----

    def get_records(self) -> List[R]:
        return list(self._records)

    def is_input_alive(self, record: R) -> bool:
        try:
            self._app_state.get_dataset(record.input_dataset_id)
            return True
        except KeyError:
            return False

    def get_status_text(self, record: R) -> str:
        return "Available" if self.is_input_alive(record) else "Input dataset closed"

    # ----- signal slots -----

    def _on_dataset_removed(self, _removed_id: int) -> None:
        self.records_changed.emit()
