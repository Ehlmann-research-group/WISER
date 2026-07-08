"""Dependency-aware Save-Project dialog (issue #626).

Presents the only real decision when saving a project: which in-memory
("RAM-backed") datasets to include.  File-backed datasets are referenced by path
for free; everything else cascades from the dataset choice, so the dialog shows a
live consequence list -- the dataset-backed spectra that will freeze to a
snapshot and the stretches that will be dropped -- recomputed as the user toggles
a dataset.  The chosen :class:`~wiser.project.resolver.DependencyResolver` is
handed to :func:`~wiser.project.orchestrate.save_project`.

The dialog is a thin view over :mod:`wiser.project.save_plan`; all the
dependency logic (and its tests) live there.
"""

from typing import TYPE_CHECKING, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from wiser.project.resolver import SavePolicy
from wiser.project.save_plan import resolver_for_selection, save_plan, savable_dataset_roots

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.project.resolver import DependencyResolver


class SaveProjectDialog(QDialog):
    def __init__(self, app_state: "ApplicationState", parent=None):
        super().__init__(parent)
        self._app_state = app_state
        self._ram_datasets, self._file_datasets = savable_dataset_roots(app_state)
        self.setWindowTitle(self.tr("Save Project"))

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                self.tr(
                    "Choose which in-memory datasets to include. File-backed datasets are "
                    "referenced automatically, and everything else follows from your choice."
                )
            )
        )

        self._roots_table = QTableWidget(len(self._ram_datasets), 2, self)
        self._roots_table.setHorizontalHeaderLabels([self.tr("Include"), self.tr("In-memory dataset")])
        self._roots_table.verticalHeader().setVisible(False)
        self._roots_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._roots_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for row, dataset in enumerate(self._ram_datasets):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Checked)
            self._roots_table.setItem(row, 0, check)
            label = dataset.get_name() or self.tr("dataset {0}").format(dataset.get_id())
            self._roots_table.setItem(row, 1, QTableWidgetItem(label))
        # Connect after populating so seeding the checkboxes doesn't fire refresh.
        self._roots_table.itemChanged.connect(self._refresh_consequences)
        layout.addWidget(self._roots_table)

        if self._file_datasets:
            layout.addWidget(
                QLabel(
                    self.tr("{0} file-backed dataset(s) will be referenced (not copied).").format(
                        len(self._file_datasets)
                    )
                )
            )

        layout.addWidget(QLabel(self.tr("Consequences of your selection:")))
        self._consequences = QTableWidget(0, 2, self)
        self._consequences.setHorizontalHeaderLabels([self.tr("Item"), self.tr("Result")])
        self._consequences.verticalHeader().setVisible(False)
        self._consequences.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._consequences.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self._consequences)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh_consequences()

    def _excluded_dataset_ids(self) -> List[int]:
        excluded = []
        for row, dataset in enumerate(self._ram_datasets):
            item = self._roots_table.item(row, 0)
            if item is not None and item.checkState() != Qt.Checked:
                excluded.append(dataset.get_id())
        return excluded

    def _refresh_consequences(self, *args) -> None:
        resolver = resolver_for_selection(self._app_state, self._excluded_dataset_ids())
        affected = [
            row for row in save_plan(self._app_state, resolver) if row["policy"] != SavePolicy.FAITHFUL.value
        ]
        self._consequences.setRowCount(len(affected))
        for i, row in enumerate(affected):
            self._consequences.setItem(i, 0, QTableWidgetItem(row["item"]))
            self._consequences.setItem(i, 1, QTableWidgetItem(_policy_text(row["policy"])))

    def get_resolver(self) -> "DependencyResolver":
        """The resolver for the user's current selection (call after ``exec()``)."""
        return resolver_for_selection(self._app_state, self._excluded_dataset_ids())


def _policy_text(policy: str) -> str:
    if policy == SavePolicy.SNAPSHOT.value:
        return "frozen to a snapshot"
    if policy == SavePolicy.DROP.value:
        return "dropped"
    return policy
