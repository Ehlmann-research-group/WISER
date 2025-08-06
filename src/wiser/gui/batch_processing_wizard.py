from __future__ import annotations
import os

from typing import TYPE_CHECKING, Dict, List, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser import plugins

from .generated.batch_processing_wizard_ui import Ui_BatchProcessingWizard

if TYPE_CHECKING:
    from .app_state import ApplicationState

def _folder_selector(parent: QWidget, line_edit: QLineEdit, title: str) -> None:
    """Open ``QFileDialog`` and put the chosen directory in *line_edit*."""
    directory = QFileDialog.getExistingDirectory(parent, title, os.getcwd())
    if directory:
        line_edit.setText(directory)

def _path_cell(parent: QWidget, title: str) -> Tuple[QWidget, QLineEdit]:
    """Return a small widget that holds a *read‑only* ``QLineEdit`` and a browse button."""
    container = QWidget(parent)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    edit = QLineEdit(container)
    edit.setReadOnly(True)
    btn = QPushButton("…", container)
    btn.setFixedWidth(28)
    btn.clicked.connect(lambda: _folder_selector(parent, edit, title))  # noqa: B023 – Qt slot

    layout.addWidget(edit, 1)
    layout.addWidget(btn, 0)
    return container, edit

class BatchProcessingWizard(QWizard):
    """Interactive wizard for batch‑processing plugins."""

    # Page *indexes* as returned by ``addPage``. Qt assigns these incrementally.
    PLUGIN_PAGE = 0
    INPUT_PAGE = 1
    OUTPUT_PAGE = 2
    REVIEW_PAGE = 3

    # Columns that are reused across the dynamic tables
    COL_TITLE = 0
    COL_PATH = 1

    def __init__(self, app_state: 'ApplicationState', parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state

        self._ui = Ui_BatchProcessingWizard()
        self._ui.setupUi(self)

        # All available batch‑processing plugins found in the app state
        #   List[Tuple[fully‑qualified‑name, plugin‑instance]]
        self._all_batch_plugins: List[Tuple[str, plugins.BatchProcessingPlugin]] = []
        # User‑chosen plugins (subset of ``_all_batch_plugins``)
        self._plugins_to_run: List[Tuple[str, plugins.BatchProcessingPlugin]] = []
        # For each plugin -> list[str] of input directories (same order as get_ordered_input_types)
        self._input_paths: Dict[str, List[str | None]] = {}
        # For each plugin -> list[str] of output directories (same order as get_ordered_output_types)
        self._output_paths: Dict[str, List[str | None]] = {}

        self._load_batch_processing_plugins()

        self._populate_plugin_selection_table()

        # Expand table headers nicely
        self._ui.tbl_wdgt_plugin_page.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # Call it right away so signals are live
        self._setup_designer_signals()  # type: ignore[misc]


    def _load_batch_processing_plugins(self):
        '''
        Loads all batch processing plugins and adds them to the UI.
        '''
        for plugin_name, plugin in self._app_state.get_plugins().items():
            if isinstance(plugin, plugins.BatchProcessingPlugin):
                print(f"plugin_name: {plugin_name}")
                self._all_batch_plugins.append((plugin_name, plugin))

    def _populate_plugin_selection_table(self) -> None:
        print(f"_populate_plugin_selection_table called, self._all_batch_plugins: {self._all_batch_plugins}")
        table = self._ui.tbl_wdgt_plugin_page
        table.clear()
        table.setRowCount(len(self._all_batch_plugins))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["", "Plugin"])
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)

        for row, (fq_name, _plugin) in enumerate(self._all_batch_plugins):
            short_name = fq_name.split(".")[-1]
            print(f"addiong plugin with name: {short_name}")

            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Unchecked)

            name_item = QTableWidgetItem(short_name)
            name_item.setFlags(Qt.ItemIsEnabled)

            table.setItem(row, 0, chk_item)
            table.setItem(row, 1, name_item)

        table.resizeColumnsToContents()

    def _build_input_selection_page(self) -> None:
        table: QTableWidget = self._ui.tbl_wdgt_input_page
        table.clear()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["", "Inputs", "Folder Paths"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setShowGrid(False)

        current_row = 0
        for fq_name, plugin in self._plugins_to_run:
            short_name = fq_name.split(".")[-1]
            input_types = plugin.get_ordered_input_types()
            self._input_paths[short_name] = [None] * len(input_types)

            for idx, in_type in enumerate(input_types):
                # make a new row for every input
                table.insertRow(current_row)

                if idx == 0:
                    # put plugin name in col 0 only on the first input row
                    name_item = QTableWidgetItem(short_name)
                    name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    table.setItem(current_row, 0, name_item)
                else:
                    # leave col 0 empty for subsequent inputs
                    empty = QTableWidgetItem()
                    empty.setFlags(Qt.ItemIsEnabled)
                    table.setItem(current_row, 0, empty)

                # description in col 1
                desc = QTableWidgetItem(
                    f"Input {idx+1} ({in_type.name.title().replace('_', ' ')})"
                )
                desc.setFlags(Qt.ItemIsEnabled)
                table.setItem(current_row, 1, desc)

                # folder-picker widget in col 2
                container, edit = _path_cell(self, "Select input folder")
                table.setCellWidget(current_row, 2, container)
                container.findChild(QPushButton).clicked.connect(  # type: ignore[attr-defined]
                    lambda _=None, sn=short_name, i=idx, le=edit: 
                        self._remember_path(sn, i, le, is_input=True)
                )

                current_row += 1


    def _build_output_selection_page(self) -> None:
        table: QTableWidget = self._ui.tbl_wdgt_output_page
        table.clear()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["", "Outputs", "Folder Paths"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setShowGrid(False)

        current_row = 0
        for fq_name, plugin in self._plugins_to_run:
            short_name = fq_name.split(".")[-1]
            output_types = plugin.get_ordered_output_types()
            self._output_paths[short_name] = [None] * len(output_types)

            # one row per output, merging plugin name into the first
            for idx, out_type in enumerate(output_types):
                table.insertRow(current_row)

                if idx == 0:
                    # plugin name only on first output row
                    name_item = QTableWidgetItem(short_name)
                    name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    table.setItem(current_row, 0, name_item)
                else:
                    # empty placeholder thereafter
                    empty = QTableWidgetItem()
                    empty.setFlags(Qt.ItemIsEnabled)
                    table.setItem(current_row, 0, empty)

                # description in col 1
                desc = QTableWidgetItem(
                    f"Output {idx+1} ({out_type.name.title().replace('_', ' ')})"
                )
                desc.setFlags(Qt.ItemIsEnabled)
                table.setItem(current_row, 1, desc)

                # folder-picker widget in col 2
                container, edit = _path_cell(self, "Select output folder")
                table.setCellWidget(current_row, 2, container)
                container.findChild(QPushButton).clicked.connect(  # type: ignore[attr-defined]
                    lambda _=None, sn=short_name, i=idx, le=edit:
                        self._remember_path(sn, i, le, is_input=False)
                )

                current_row += 1

    # ---------- PAGE 4 – Review ---------------------------------------------------------------------

    def _build_review_page(self) -> None:
        table: QTableWidget = self._ui.tbl_wdgt_review
        table.clear()
        # we’ll draw our own headers inside the rows
        table.setColumnCount(3)
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents) 

        current_row = -1
        for fq_name, plugin in self._plugins_to_run:
            short_name = fq_name.split(".")[-1]
            input_types = plugin.get_ordered_input_types()
            output_types = plugin.get_ordered_output_types()
            inp_paths = self._input_paths.get(short_name, [])
            out_paths = self._output_paths.get(short_name, [])

            # how many rows this plugin will occupy
            total_rows = 1 + len(input_types) + len(output_types)

            # 1) Insert the “header” row for this plugin
            table.insertRow(current_row + 1)
            current_row += 1

            # Plugin name cell, spanning all sub-rows
            plugin_item = QTableWidgetItem(short_name)
            plugin_item.setFlags(Qt.ItemIsEnabled)
            table.setItem(current_row, 0, plugin_item)
            table.setSpan(current_row, 0, total_rows, 1)

            # Our inline column headers
            io_hdr = QTableWidgetItem("Inputs / Outputs")
            io_hdr.setFlags(Qt.ItemIsEnabled)
            table.setItem(current_row, 1, io_hdr)

            path_hdr = QTableWidgetItem("Folder Paths")
            path_hdr.setFlags(Qt.ItemIsEnabled)
            table.setItem(current_row, 2, path_hdr)

            # 2) List each input
            for idx, in_type in enumerate(input_types):
                table.insertRow(current_row + 1)
                current_row += 1

                desc = QTableWidgetItem(
                    f"Input {idx+1} ({in_type.name.title().replace('_',' ')})"
                )
                desc.setFlags(Qt.ItemIsEnabled)
                table.setItem(current_row, 1, desc)

                lbl = QLabel(inp_paths[idx] or "")
                lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
                table.setCellWidget(current_row, 2, lbl)

            # 3) List each output
            for idx, out_type in enumerate(output_types):
                table.insertRow(current_row + 1)
                current_row += 1

                desc = QTableWidgetItem(
                    f"Output {idx+1} ({out_type.name.title().replace('_',' ')})"
                )
                desc.setFlags(Qt.ItemIsEnabled)
                table.setItem(current_row, 1, desc)

                lbl = QLabel(out_paths[idx] or "")
                lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
                table.setCellWidget(current_row, 2, lbl)


    # =================================================================================================
    # Data helpers -----------------------------------------------------------------------------------
    # =================================================================================================

    def _remember_path(self, plugin: str, idx: int, line_edit: QLineEdit, *, is_input: bool) -> None:  # noqa: D401 – simple helper
        """Store the folder path the *line_edit* currently shows into our dictionaries."""
        path = line_edit.text()
        if not path:
            return
        if is_input:
            self._input_paths[plugin][idx] = path
        else:
            self._output_paths[plugin][idx] = path

    # =================================================================================================
    # QWizard hooks ----------------------------------------------------------------------------------
    # =================================================================================================

    def validateCurrentPage(self) -> bool:
        page_id = self.currentId()

        # -------------------------------------------------------------------------------------------------
        # Leaving PLUGIN selection → collect chosen plugins and build INPUT page
        # -------------------------------------------------------------------------------------------------
        if page_id == self.PLUGIN_PAGE:
            self._plugins_to_run = []
            table = self._ui.tbl_wdgt_plugin_page
            for row, (fq_name, plugin) in enumerate(self._all_batch_plugins):
                if table.item(row, 0).checkState() == Qt.Checked:
                    self._plugins_to_run.append((fq_name, plugin))

            if not self._plugins_to_run:
                QMessageBox.warning(self, "No plugin selected", "Please select at least one plugin to continue.")
                return False

            self._build_input_selection_page()
            return True

        # -------------------------------------------------------------------------------------------------
        # Leaving INPUT selection → verify every input folder and build OUTPUT page
        # -------------------------------------------------------------------------------------------------
        if page_id == self.INPUT_PAGE:
            if not self._all_paths_chosen(self._input_paths, "input"):
                return False
            self._build_output_selection_page()
            return True

        # -------------------------------------------------------------------------------------------------
        # Leaving OUTPUT selection → verify and build REVIEW page
        # -------------------------------------------------------------------------------------------------
        if page_id == self.OUTPUT_PAGE:
            if not self._all_paths_chosen(self._output_paths, "output"):
                return False
            self._build_review_page()
            return True

        return super().validateCurrentPage()

    # -------------------------------------------------------------------------------------------------

    def _all_paths_chosen(self, mapping: Dict[str, List[str | None]], kind: str) -> bool:
        """Check that *every* required folder has been picked; complain otherwise."""
        missing: List[str] = []
        for plugin_name, paths in mapping.items():
            for idx, path in enumerate(paths):
                if not path:
                    missing.append(f"{plugin_name} – {kind.title()} {idx + 1}")
        if missing:
            QMessageBox.warning(
                self,
                "Missing folders",
                "The following folders have not been chosen:\n\n" + "\n".join(missing),
            )
            return False
        return True

    # =================================================================================================
    # Public API -------------------------------------------------------------------------------------
    # =================================================================================================

    def get_batch_plan(self) -> Tuple[
        List[Tuple[str, plugins.BatchProcessingPlugin]],
        Dict[str, List[str]],
        Dict[str, List[str]],
    ]:
        """Return everything the caller needs to run the batch externally."""
        # Paths are already validated → we can safely cast to ``str``
        cast_inputs = {k: [p or "" for p in v] for k, v in self._input_paths.items()}
        cast_outputs = {k: [p or "" for p in v] for k, v in self._output_paths.items()}
        return self._plugins_to_run, cast_inputs, cast_outputs


    # -------------------------------------------------------------------------------------------------
    # Connect Designer buttons -------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------

    def _on_run_clicked(self) -> None:  # noqa: D401 – slot
        # Placeholder – emit a signal or call the application’s batch runner
        QMessageBox.information(self, "Run", "Batch running not implemented yet.")

    def _on_cancel_clicked(self) -> None:  # noqa: D401 – slot
        self.reject()


    # Hook up buttons after the class body (they are created by Designer)
    def _setup_designer_signals(self) -> None:  # noqa: D401 – helper
        self._ui.btn_run.clicked.connect(self._on_run_clicked)
        self._ui.btn_cancel.clicked.connect(self._on_cancel_clicked)


    