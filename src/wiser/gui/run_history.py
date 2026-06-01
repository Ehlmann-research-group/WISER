"""Shared base classes for per-task-type completed-run histories and viewers.

PCA, MNF, and linear unmixing each have their own concrete record type so the
histories are never accidentally conflated, but the manager logic — store
records, emit ``records_changed`` on add/remove/dataset_removed, expose
liveness/status of the input dataset — is identical between them.  This
module factors that shared logic into :class:`RunHistoryManagerBase` so the
per-task-type managers (see :mod:`wiser.gui.permanent_plugins.pca_plugin`,
:mod:`wiser.gui.mnf`, :mod:`wiser.gui.linear_unmixing`) are tiny shells
declaring their record type and overriding only what differs.

The eigenvalue-based past-runs viewer (PCA + MNF) also lives here as
:class:`EigenScreeRunHistoryDialog`; linear unmixing has its own viewer
because its rows show endmember spectra rather than a scree plot.
"""

from __future__ import annotations

import datetime
from typing import Generic, List, Protocol, TYPE_CHECKING, TypeVar

import numpy as np
from PySide2.QtCore import QObject, Qt, Signal, Slot
from PySide2.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from wiser.gui.scree_plot import build_scree_plot_figure

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState


class _HasRunIdAndInputDataset(Protocol):
    """The minimal shape a record must have for the base manager to use it."""

    run_id: int
    input_dataset_id: int


R = TypeVar("R", bound=_HasRunIdAndInputDataset)


class _EigenRunRecord(Protocol):
    """Shape a record must have for :class:`EigenScreeRunHistoryDialog`."""

    run_id: int
    timestamp: datetime.datetime
    input_dataset_id: int
    input_dataset_name_snapshot: str
    num_components_chosen: int
    max_components_available: int
    eigenvalues: np.ndarray


ER = TypeVar("ER", bound=_EigenRunRecord)


class RunHistoryManagerBase(QObject, Generic[R]):
    """Generic in-memory list of completed task runs.

    Subclass and bind ``R`` to your concrete record type.  The subclass
    typically doesn't need to override anything — just inherit and
    instantiate.  Override :meth:`get_status_text` (and add accessors like
    ``is_output_alive``) if your task tracks more liveness than just the
    input dataset.
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


# Column indices for the eigen-scree past-runs table.  Shared between the
# active + closed tables.
_RUN_COL_RUN = 0
_RUN_COL_TIME = 1
_RUN_COL_INPUT = 2
_RUN_COL_NUM_USED = 3
_RUN_COL_MAX_AVAIL = 4
_RUN_COL_STATUS = 5
_RUN_COL_VIEW = 6
_RUN_COL_DELETE = 7
_RUN_COL_COUNT = 8

# Unicode arrows for the Closed Runs toggle button (mirrors activity monitor
# and the linear-unmix history dialog).
_ARROW_EXPANDED = "▼"
_ARROW_COLLAPSED = "▶"


class EigenScreeRunHistoryDialog(QDialog, Generic[ER]):
    """Non-modal viewer for past runs that produced an eigenvalue spectrum.

    Used by PCA and MNF; structurally mirrors
    :class:`wiser.gui.linear_unmixing.LinearUnmixingHistoryDialog` (active +
    collapsible-closed tables), but the View button opens a scree plot of the
    captured eigenvalues rather than re-displaying spectra.

    Subclasses customize:
      * :attr:`task_label` — short name used in the window title and the
        scree-plot title (e.g. ``"PCA"``, ``"MNF"``).

    The manager is passed in and must be a
    :class:`RunHistoryManagerBase` instance whose records satisfy
    :class:`_EigenRunRecord` (i.e. carry the ``eigenvalues``,
    ``num_components_chosen``, etc. fields).
    """

    # Subclasses override.  Used in the window title and scree-plot title.
    task_label: str = "Run"

    def __init__(
        self,
        app_state: "ApplicationState",
        manager: RunHistoryManagerBase[ER],
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr(f"{self.task_label} — Past Runs"))
        self.setModal(False)
        self.resize(800, 500)

        self._app_state = app_state
        self._manager = manager
        self._closed_expanded = False

        outer = QVBoxLayout(self)

        outer.addWidget(QLabel(self.tr("Active Runs"), self))
        self._tbl_active = self._make_table()
        outer.addWidget(self._tbl_active)

        self._btn_toggle_closed = QPushButton(self)
        self._btn_toggle_closed.clicked.connect(self._toggle_closed_section)
        outer.addWidget(self._btn_toggle_closed)

        self._tbl_closed = self._make_table()
        self._tbl_closed.setVisible(self._closed_expanded)
        outer.addWidget(self._tbl_closed)

        btn_close = QPushButton(self.tr("Close"), self)
        btn_close.clicked.connect(self.close)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(btn_close)
        outer.addLayout(bottom)

        self._manager.records_changed.connect(self._rebuild_tables)
        self._rebuild_tables()

    # ----- helpers -----

    def _make_table(self) -> QTableWidget:
        table = QTableWidget(self)
        table.setColumnCount(_RUN_COL_COUNT)
        table.setHorizontalHeaderLabels(
            [
                self.tr("Run"),
                self.tr("Time"),
                self.tr("Input dataset"),
                self.tr("# Used"),
                self.tr("# Available"),
                self.tr("Status"),
                self.tr(""),
                self.tr(""),
            ]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        header = table.horizontalHeader()
        header.setSectionResizeMode(_RUN_COL_INPUT, QHeaderView.Stretch)
        header.setSectionResizeMode(_RUN_COL_STATUS, QHeaderView.Stretch)
        for col in (
            _RUN_COL_RUN,
            _RUN_COL_TIME,
            _RUN_COL_NUM_USED,
            _RUN_COL_MAX_AVAIL,
            _RUN_COL_VIEW,
            _RUN_COL_DELETE,
        ):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        return table

    def _toggle_closed_section(self) -> None:
        self._closed_expanded = not self._closed_expanded
        self._tbl_closed.setVisible(self._closed_expanded)
        self._sync_closed_toggle_button()

    def _sync_closed_toggle_button(self) -> None:
        arrow = _ARROW_EXPANDED if self._closed_expanded else _ARROW_COLLAPSED
        count = self._tbl_closed.rowCount()
        self._btn_toggle_closed.setText(self.tr(f"Closed Runs ({count}) {arrow}"))

    # ----- rendering -----

    @Slot()
    def _rebuild_tables(self) -> None:
        self._tbl_active.setRowCount(0)
        self._tbl_closed.setRowCount(0)
        for record in self._manager.get_records():
            if self._manager.is_input_alive(record):
                self._append_row(self._tbl_active, record, alive=True)
            else:
                self._append_row(self._tbl_closed, record, alive=False)
        self._sync_closed_toggle_button()

    def _append_row(self, table: QTableWidget, record: ER, *, alive: bool) -> None:
        row = table.rowCount()
        table.insertRow(row)

        table.setItem(row, _RUN_COL_RUN, QTableWidgetItem(str(record.run_id)))
        table.setItem(
            row,
            _RUN_COL_TIME,
            QTableWidgetItem(record.timestamp.strftime("%Y-%m-%d %H:%M:%S")),
        )
        table.setItem(
            row,
            _RUN_COL_INPUT,
            self._make_dataset_item(
                record.input_dataset_id,
                record.input_dataset_name_snapshot,
                alive=alive,
            ),
        )
        table.setItem(
            row,
            _RUN_COL_NUM_USED,
            QTableWidgetItem(str(record.num_components_chosen)),
        )
        table.setItem(
            row,
            _RUN_COL_MAX_AVAIL,
            QTableWidgetItem(str(record.max_components_available)),
        )
        table.setItem(
            row,
            _RUN_COL_STATUS,
            QTableWidgetItem(self._manager.get_status_text(record)),
        )

        # Lambda binding captures rid by default arg to avoid the
        # late-binding-loop-variable pitfall.
        rid = record.run_id
        btn_view = QPushButton(self.tr("View"), table)
        btn_view.clicked.connect(lambda checked=False, r=rid: self._on_view_clicked(r))
        table.setCellWidget(row, _RUN_COL_VIEW, btn_view)

        btn_delete = QPushButton(self.tr("Delete"), table)
        btn_delete.clicked.connect(lambda checked=False, r=rid: self._manager.remove_record(r))
        table.setCellWidget(row, _RUN_COL_DELETE, btn_delete)

    def _make_dataset_item(
        self,
        dataset_id: int,
        snapshot_name: str,
        *,
        alive: bool,
    ) -> QTableWidgetItem:
        if alive:
            ds = self._app_state.get_dataset(dataset_id)
            return QTableWidgetItem(ds.get_name() or snapshot_name)
        item = QTableWidgetItem(f"{snapshot_name} (closed)")
        font = item.font()
        font.setItalic(True)
        item.setFont(font)
        item.setForeground(Qt.gray)
        return item

    # ----- View click -----

    def _on_view_clicked(self, run_id: int) -> None:
        # Liveness can change between the click and the slot firing; re-fetch
        # the record fresh rather than capturing it in the lambda.
        record = next(
            (r for r in self._manager.get_records() if r.run_id == run_id),
            None,
        )
        if record is None:
            return

        title = (
            f"{self.task_label} Run {record.run_id} — Scree Plot " f"({record.input_dataset_name_snapshot})"
        )
        description = (
            f"{self.task_label} run {record.run_id} on "
            f"'{record.input_dataset_name_snapshot}' — "
            f"{record.num_components_chosen} of {record.max_components_available} "
            f"components kept."
        )
        figure, axes = build_scree_plot_figure(record.eigenvalues, title=title)
        self._app_state.show_matplotlib_display_widget(
            figure=figure,
            axes=axes,
            window_title=title,
            description=description,
        )
