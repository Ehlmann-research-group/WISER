"""Reusable controller for a QTableWidget that holds a curated list of spectra.

Encapsulates the behaviour previously inlined in :class:`LinearUnmixingDialog`'s
endmember table: configuring the table, adding/removing rows (with dedupe and a
per-row trash button), keeping rows in sync with application-state removals of
collected spectra and spectral libraries, and the "Add Collected Spectrum" /
"Import Spectrum" entry flows.

Both the linear-unmixing endmember table and the k-means manual-init table drive
their tables through this controller.  Consumers may pass ``add_filter`` to
validate or transform a batch of spectra before any rows are created — k-means
uses it to drop band-count mismatches and warn — while a ``None`` filter adds
every spectrum (linear unmixing).
"""

import os
from typing import Any, Callable, List, Optional

from PySide2.QtCore import QObject, Qt
from PySide2.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHeaderView,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from wiser.gui.app_state import ApplicationState
from wiser.gui.import_spectra_text import ImportSpectraTextDialog
from wiser.gui.util import GenericMultiSelectDialog, StateChange, build_trash_button
from wiser.raster.spectral_library import ListSpectralLibrary
from wiser.raster.spectrum import Spectrum


class SpectraTableController(QObject):
    """Drive a two-column QTableWidget of spectra (name + remove button).

    Args:
        table: The (already-created) table widget to manage.  Its contents and
            configuration are fully owned by this controller.
        app_state: Running application state, used to resolve collected spectra
            and spectral libraries and to receive removal signals.
        parent_widget: The dialog that owns the table; used as the Qt parent for
            child dialogs (multi-select, file picker, import) so they centre on it.
        name_column_label: Header text for the spectrum-name column.
        remove_tooltip: Tooltip for each row's remove button.
        remove_fallback: Text shown on the remove button when no trash icon is
            available.
        add_filter: Optional hook applied to a batch of spectra before rows are
            created.  It receives the full batch and returns the subset to add
            (and may show its own warnings).  ``None`` adds every spectrum.
    """

    _NAME_COL = 0
    _REMOVE_COL = 1

    def __init__(
        self,
        table: QTableWidget,
        app_state: ApplicationState,
        parent_widget: QWidget,
        *,
        name_column_label: str = "Spectrum",
        remove_tooltip: str = "Remove",
        remove_fallback: str = "Remove",
        add_filter: Optional[Callable[[List[Spectrum]], List[Spectrum]]] = None,
    ) -> None:
        super().__init__(parent_widget)
        self._table = table
        self._app_state = app_state
        self._parent = parent_widget
        self._remove_tooltip = remove_tooltip
        self._remove_fallback = remove_fallback
        self._add_filter = add_filter

        self._configure_table(name_column_label)

        # Keep the table in sync with the app's spectrum store: if a collected
        # spectrum is discarded, or a spectral library is removed, drop any rows
        # referencing them. (No name-change signal exists, so we can't reactively
        # update display names — they'll just stay stale until the row is removed.)
        app_state.collected_spectra_changed.connect(self._on_collected_spectra_changed)
        app_state.spectral_library_removed.connect(self._on_spectral_library_removed)

    # ------------------------------------------------------------------
    # Table configuration
    # ------------------------------------------------------------------

    def _configure_table(self, name_column_label: str) -> None:
        table = self._table
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels([name_column_label, ""])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        header = table.horizontalHeader()
        # Name column takes all available horizontal space; remove button only
        # claims what it needs so it stays compact.
        header.setSectionResizeMode(self._NAME_COL, QHeaderView.Stretch)
        header.setSectionResizeMode(self._REMOVE_COL, QHeaderView.ResizeToContents)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_spectra(self, specs: List[Spectrum]) -> None:
        """Add a batch of spectra as rows, after the optional ``add_filter``.

        All add paths (collected, imported, programmatic) funnel through here so
        the filter sees the whole batch at once.
        """
        to_add = self._add_filter(specs) if self._add_filter is not None else specs
        for spec in to_add:
            self._add_row(spec)

    def collect_spectra(self) -> List[Spectrum]:
        """Return the ordered list of spectra currently in the table.

        Spectra are re-fetched from the application state by their stored ID so
        the returned objects reflect the live spectrum, not a stale copy.

        Raises:
            KeyError: If a collected spectrum or spectral library has been removed
                from the app state since it was added to the table.
        """
        table = self._table
        spectra: List[Spectrum] = []
        for row in range(table.rowCount()):
            item = table.item(row, self._NAME_COL)
            if item is None:
                continue
            spec_id = item.data(Qt.UserRole)
            if isinstance(spec_id, tuple):
                # Library spectrum: (library_id, index)
                lib_id, index = spec_id
                spec = self._app_state.get_spectral_library(lib_id).get_spectrum(index)
            else:
                # Collected spectrum: plain integer ID
                spec = self._app_state.get_spectrum(spec_id)
            spectra.append(spec)
        return spectra

    def row_count(self) -> int:
        return self._table.rowCount()

    def clear(self) -> None:
        self._table.setRowCount(0)

    def on_add_collected_clicked(self) -> None:
        """Open the collected-spectra multi-select picker and add the chosen rows."""
        collected_spectra = self._app_state.get_collected_spectra()
        if not collected_spectra:
            QMessageBox.information(
                self._parent,
                self.tr("Add Collected Spectra"),
                self.tr("There are no collected spectra to add."),
            )
            return

        names = [spec.get_name() or self.tr("<unnamed>") for spec in collected_spectra]
        dlg = GenericMultiSelectDialog(
            names=names,
            data_items=list(collected_spectra),
            item_column_name=self.tr("Collected Spectrum"),
            title=self.tr("Add Collected Spectra"),
            parent=self._parent,
        )
        dlg.setWindowModality(Qt.WindowModal)
        if dlg.exec_() != QDialog.Accepted:
            return
        _, selected_specs = dlg.get_selected()
        self.add_spectra(list(selected_specs))

    def on_import_clicked(self) -> None:
        """Open a text-file importer, register the spectra, and add the rows."""
        start_dir = self._app_state.get_current_dir() or os.path.expanduser("~")
        filedlg = QFileDialog(
            self._parent,
            self.tr("Import Spectra from Text File"),
            start_dir,
            self.tr("Text files (*.txt);;All Files (*)"),
        )
        filedlg.setFileMode(QFileDialog.ExistingFile)
        filedlg.setAcceptMode(QFileDialog.AcceptOpen)
        # WindowModal blocks interaction with the owning dialog but not the whole
        # app, which is appropriate for a child file picker.
        filedlg.setWindowModality(Qt.WindowModal)
        if filedlg.exec_() != QDialog.Accepted:
            return
        path = filedlg.selectedFiles()[0]

        # Update the working directory so future file pickers open in the same place.
        self._app_state.update_cwd_from_path(path)
        dlg = ImportSpectraTextDialog(path, parent=self._parent)
        dlg.setWindowModality(Qt.WindowModal)
        if dlg.exec() != QDialog.Accepted:
            return
        specs = dlg.get_spectra()
        if not specs:
            return

        # Wrap the imported spectra in a ListSpectralLibrary so they are
        # registered with the app and appear in the spectrum chooser. This also
        # triggers set_id() on the library, which assigns compound tuple IDs
        # (library_id, index) to each spectrum — see _add_row.
        lib = ListSpectralLibrary(specs, path=path)
        self._app_state.add_spectral_library(lib)

        self.add_spectra(list(specs))

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def _add_row(self, spec: Spectrum) -> None:
        # Spectra collected in-app already have an integer ID assigned by
        # take_next_id(). Spectra from a spectral library get a compound
        # (library_id, index) tuple ID when the library is registered — see
        # ListSpectralLibrary.set_id(). The only case where get_id() is None
        # is a brand-new Spectrum that hasn't been added to the app yet.
        if spec.get_id() is None:
            spec.set_id(self._app_state.take_next_id())
        # Do NOT cast to int here, library spectra have tuple IDs, and int()
        # on a tuple raises TypeError. Keep the raw value; tuples are hashable
        # and compare correctly with ==, so everything downstream still works.
        spec_id = spec.get_id()

        # Deduplicate: if this spectrum is already in the table, silently ignore.
        if self._find_row(spec_id) is not None:
            return

        table = self._table
        row = table.rowCount()
        table.insertRow(row)

        name_item = QTableWidgetItem(spec.get_name() or self.tr("<unnamed>"))
        # Store the ID in UserRole so _find_row and _remove can look up the row by
        # identity rather than by display name (which could collide if two spectra
        # share a name).
        name_item.setData(Qt.UserRole, spec_id)
        table.setItem(row, self._NAME_COL, name_item)

        remove_button = build_trash_button(
            self._parent,
            # Default argument binding (sid=spec_id) captures the current value
            # of spec_id rather than a closure over the loop variable, which
            # would give every button the same ID by the time it's clicked.
            lambda checked=False, sid=spec_id: self._remove(sid),
            tooltip=self._remove_tooltip,
            fallback_text=self._remove_fallback,
        )
        table.setCellWidget(row, self._REMOVE_COL, remove_button)

    def _find_row(self, spec_id: Any) -> Optional[int]:
        """Return the row index for the given spectrum ID, or None if not present."""
        table = self._table
        for row in range(table.rowCount()):
            item = table.item(row, self._NAME_COL)
            if item is not None and item.data(Qt.UserRole) == spec_id:
                return row
        return None

    def _remove(self, spec_id: Any) -> None:
        row = self._find_row(spec_id)
        if row is not None:
            self._table.removeRow(row)

    def _on_collected_spectra_changed(self, state_change, _index: int, spec_id) -> None:
        # We only care about removals here. Additions don't implicitly add a
        # spectrum to the table — the user has to choose it.
        if state_change != StateChange.ITEM_REMOVED:
            return
        if spec_id == -1:
            # "Remove all collected spectra" sentinel (see
            # app_state.remove_all_collected_spectra). Drop every row whose ID is
            # a plain int — those came from the collected-spectra pool. Library
            # spectra have tuple IDs and are untouched by this signal.
            self._remove_rows_matching(lambda sid: not isinstance(sid, tuple))
        else:
            self._remove(spec_id)

    def _on_spectral_library_removed(self, lib_id: int) -> None:
        # Library spectra have compound IDs of the form (lib_id, index). Drop
        # every row whose tuple's first element matches the removed library.
        self._remove_rows_matching(lambda sid: isinstance(sid, tuple) and len(sid) >= 1 and sid[0] == lib_id)

    def _remove_rows_matching(self, predicate: Callable[[Any], bool]) -> None:
        # Iterate in reverse so row-index shifting from removeRow() doesn't
        # skip rows we still need to visit.
        table = self._table
        for row in range(table.rowCount() - 1, -1, -1):
            item = table.item(row, self._NAME_COL)
            if item is not None and predicate(item.data(Qt.UserRole)):
                table.removeRow(row)
