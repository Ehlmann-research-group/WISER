from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from PySide2.QtCore import QTimer, Qt
from PySide2.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from wiser.gui.generated.activity_monitor_ui import Ui_ActivityMonitor


class ActivityMonitorWidget(QWidget):
    ACTIVE_COLUMNS = ("Task", "Activity")
    FINISHED_COLUMNS = ("Task", "Activity", "Remove")

    STATE_IDLE = "idle"
    STATE_RUNNING = "running"
    STATE_FINISHED = "finished"
    STATE_CANCELLED = "cancelled"
    STATE_FAILED = "failed"

    TERMINAL_STATES = {STATE_FINISHED, STATE_CANCELLED, STATE_FAILED}
    MOVE_DELAY_MS = 1000

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        self._ui = Ui_ActivityMonitor()
        self._ui.setupUi(self)

        self._next_task_id = 1
        self._task_locations: Dict[int, Tuple[QTableWidget, int]] = {}
        self._finished_expanded = True

        self._configure_table(self._ui.tbl_wdgt_active_tasks, self.ACTIVE_COLUMNS)
        self._configure_table(self._ui.tbl_wdgt_finished_tasks, self.FINISHED_COLUMNS)
        self._ui.btn_finished_tasks.clicked.connect(self._toggle_finished_tasks)
        self._sync_finished_section_button()

    def register_task(self, title: str, meta: Dict[str, str], cancel_callback: Callable) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1

        task_widget = self._build_task_widget(title=title, meta=meta, enabled=True)
        controls_widget, controls = self._build_controls_widget(
            progress_enabled=True,
            cancel_enabled=True,
            cancel_callback=lambda checked=False, task_id=task_id: self._cancel_task(task_id),
        )
        metadata = {
            "task_id": task_id,
            "title": title,
            "meta": dict(meta),
            "state": self.STATE_IDLE,
            "errors": [],
            "cancel_callback": cancel_callback,
            "controls": controls,
            "task_widget": task_widget,
            "move_scheduled": False,
        }

        self._insert_task_row(
            table=self._ui.tbl_wdgt_active_tasks,
            metadata=metadata,
            task_widget=task_widget,
            controls_widget=controls_widget,
        )
        return task_id

    def set_task_running(self, task_id: int) -> None:
        self._set_task_state(task_id, self.STATE_RUNNING)

    def set_task_progress(self, task_id: int, value: int) -> None:
        metadata = self._get_task_metadata(task_id)
        controls = metadata["controls"]
        progress_bar = controls["progress_bar"]
        progress_bar.setValue(max(0, min(100, int(value))))
        if metadata["state"] == self.STATE_IDLE:
            self._set_task_state(task_id, self.STATE_RUNNING)

    def append_task_error(self, task_id: int, error_message: str) -> None:
        metadata = self._get_task_metadata(task_id)
        errors: List[str] = metadata["errors"]
        errors.append(error_message)
        metadata["controls"]["view_errors_button"].setEnabled(True)
        self._store_task_metadata(task_id, metadata)

    def complete_task(self, task_id: int) -> None:
        self.set_task_finished(task_id)

    def set_task_finished(self, task_id: int) -> None:
        metadata = self._get_task_metadata(task_id)
        metadata["controls"]["progress_bar"].setValue(100)
        self._store_task_metadata(task_id, metadata)
        self._set_task_state(task_id, self.STATE_FINISHED)
        self._schedule_move_to_finished(task_id)

    def set_task_cancelled(self, task_id: int) -> None:
        self._set_task_state(task_id, self.STATE_CANCELLED)
        self._schedule_move_to_finished(task_id)

    def set_task_failed(self, task_id: int, error_message: Optional[str] = None) -> None:
        if error_message:
            self.append_task_error(task_id, error_message)
        self._set_task_state(task_id, self.STATE_FAILED)
        self._schedule_move_to_finished(task_id)

    def remove_task(self, task_id: int) -> None:
        location = self._task_locations.get(task_id)
        if location is None:
            raise KeyError(f"Unknown task id: {task_id}")

        table, row = location
        table.removeRow(row)
        self._rebuild_task_locations()

    def _configure_table(self, table: QTableWidget, columns: Tuple[str, ...]) -> None:
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(list(columns))
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(0, table.horizontalHeader().Stretch)
        if len(columns) > 1:
            table.horizontalHeader().setSectionResizeMode(1, table.horizontalHeader().ResizeToContents)
        if len(columns) > 2:
            table.horizontalHeader().setSectionResizeMode(2, table.horizontalHeader().ResizeToContents)
        table.setWordWrap(True)

    def _build_task_widget(self, title: str, meta: Dict[str, str], enabled: bool) -> QLabel:
        label = QLabel()
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setMargin(6)
        label.setText(self._format_task_html(title, meta))
        label.setEnabled(enabled)
        return label

    def _build_controls_widget(
        self,
        *,
        progress_enabled: bool,
        cancel_enabled: bool,
        cancel_callback: Callable,
    ) -> Tuple[QWidget, Dict[str, QWidget]]:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        view_errors_button = QPushButton(self.tr("View Errors"))
        view_errors_button.setEnabled(False)
        view_errors_button.clicked.connect(
            lambda checked=False, controls_container=container: self._show_errors_for_controls(
                controls_container
            )
        )

        cancel_button = QPushButton(self.tr("Cancel"))
        cancel_button.setEnabled(cancel_enabled)
        cancel_button.clicked.connect(cancel_callback)

        top_layout.addWidget(view_errors_button)
        top_layout.addWidget(cancel_button)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        progress_bar.setEnabled(progress_enabled)

        layout.addWidget(top_row)
        layout.addWidget(progress_bar)

        controls = {
            "container": container,
            "view_errors_button": view_errors_button,
            "cancel_button": cancel_button,
            "progress_bar": progress_bar,
        }
        return container, controls

    def _build_remove_button(self, callback: Callable) -> QPushButton:
        button = QPushButton()
        button.setToolTip(self.tr("Remove task from view"))
        trash_icon_enum = getattr(QStyle, "SP_TrashIcon", None)
        icon = self.style().standardIcon(trash_icon_enum) if trash_icon_enum is not None else None
        if icon is None or icon.isNull():
            button.setText(self.tr("Remove"))
        else:
            button.setIcon(icon)
        button.clicked.connect(callback)
        return button

    def _insert_task_row(
        self,
        *,
        table: QTableWidget,
        metadata: Dict[str, object],
        task_widget: QWidget,
        controls_widget: QWidget,
        remove_button: Optional[QWidget] = None,
    ) -> None:
        row = 0
        table.insertRow(row)

        item = QTableWidgetItem()
        item.setData(Qt.UserRole, metadata)
        table.setItem(row, 0, item)

        table.setCellWidget(row, 0, task_widget)
        table.setCellWidget(row, 1, controls_widget)
        if remove_button is not None and table.columnCount() > 2:
            table.setCellWidget(row, 2, remove_button)

        self._rebuild_task_locations()

    def _metadata_item(self, table: QTableWidget, row: int) -> QTableWidgetItem:
        item = table.item(row, 0)
        if item is None:
            raise RuntimeError(f"Missing metadata item at row {row}")
        return item

    def _get_task_metadata(self, task_id: int) -> Dict[str, object]:
        location = self._task_locations.get(task_id)
        if location is None:
            raise KeyError(f"Unknown task id: {task_id}")

        table, row = location
        return self._metadata_item(table, row).data(Qt.UserRole)

    def _store_task_metadata(self, task_id: int, metadata: Dict[str, object]) -> None:
        location = self._task_locations.get(task_id)
        if location is None:
            raise KeyError(f"Unknown task id: {task_id}")

        table, row = location
        self._metadata_item(table, row).setData(Qt.UserRole, metadata)

    def _set_task_state(self, task_id: int, state: str) -> None:
        metadata = self._get_task_metadata(task_id)
        metadata["state"] = state
        self._store_task_metadata(task_id, metadata)

    def _schedule_move_to_finished(self, task_id: int) -> None:
        metadata = self._get_task_metadata(task_id)
        if metadata["state"] not in self.TERMINAL_STATES:
            return
        if metadata.get("move_scheduled"):
            return

        metadata["move_scheduled"] = True
        self._store_task_metadata(task_id, metadata)
        QTimer.singleShot(self.MOVE_DELAY_MS, lambda task_id=task_id: self._move_task_to_finished(task_id))

    def _move_task_to_finished(self, task_id: int) -> None:
        location = self._task_locations.get(task_id)
        if location is None:
            return

        source_table, source_row = location
        if source_table is self._ui.tbl_wdgt_finished_tasks:
            return

        metadata = self._get_task_metadata(task_id)
        task_widget = self._build_task_widget(
            title=metadata["title"],
            meta=metadata["meta"],
            enabled=False,
        )
        controls_widget, controls = self._build_controls_widget(
            progress_enabled=False,
            cancel_enabled=False,
            cancel_callback=lambda checked=False: None,
        )
        controls["progress_bar"].setValue(metadata["controls"]["progress_bar"].value())
        if metadata["errors"]:
            controls["view_errors_button"].setEnabled(True)

        metadata["task_widget"] = task_widget
        metadata["controls"] = controls
        metadata["move_scheduled"] = False

        remove_button = self._build_remove_button(
            lambda checked=False, task_id=task_id: self.remove_task(task_id)
        )
        source_table.removeRow(source_row)
        self._rebuild_task_locations()

        self._insert_task_row(
            table=self._ui.tbl_wdgt_finished_tasks,
            metadata=metadata,
            task_widget=task_widget,
            controls_widget=controls_widget,
            remove_button=remove_button,
        )

        if not self._finished_expanded:
            self._toggle_finished_tasks()

    def _cancel_task(self, task_id: int) -> None:
        metadata = self._get_task_metadata(task_id)
        cancel_callback = metadata["cancel_callback"]
        if callable(cancel_callback):
            try:
                cancel_callback()
            except Exception as exc:
                self.set_task_failed(task_id, str(exc))
                return
        self.set_task_cancelled(task_id)

    def _show_errors_for_controls(self, controls_container: QWidget) -> None:
        task_id = self._find_task_id_by_controls(controls_container)
        if task_id is None:
            return

        metadata = self._get_task_metadata(task_id)
        errors: List[str] = metadata["errors"]
        message = "\n\n".join(errors) if errors else self.tr("No errors recorded for this task.")
        QMessageBox.information(self, self.tr("Task Errors"), message)

    def _find_task_id_by_controls(self, controls_container: QWidget) -> Optional[int]:
        for task_id in self._task_locations:
            metadata = self._get_task_metadata(task_id)
            if metadata["controls"]["container"] is controls_container:
                return task_id
        return None

    def _toggle_finished_tasks(self) -> None:
        self._finished_expanded = not self._finished_expanded
        self._ui.tbl_wdgt_finished_tasks.setVisible(self._finished_expanded)
        self._sync_finished_section_button()

    def _sync_finished_section_button(self) -> None:
        arrow = "\u25bc" if self._finished_expanded else "\u25b6"
        count = self._ui.tbl_wdgt_finished_tasks.rowCount()
        self._ui.btn_finished_tasks.setText(self.tr(f"Finished Tasks ({count}) {arrow}"))

    def _rebuild_task_locations(self) -> None:
        self._task_locations = {}
        for table in (self._ui.tbl_wdgt_active_tasks, self._ui.tbl_wdgt_finished_tasks):
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item is None:
                    continue
                metadata = item.data(Qt.UserRole)
                if not isinstance(metadata, dict):
                    continue
                task_id = metadata.get("task_id")
                if task_id is None:
                    continue
                self._task_locations[int(task_id)] = (table, row)
        self._sync_finished_section_button()

    def _format_task_html(self, title: str, meta: Dict[str, str]) -> str:
        lines = ["<b>Task Title:</b> " + self._escape_html(title)]
        lines.append("<b>Variables:</b>")
        if meta:
            for key, value in meta.items():
                lines.append(
                    "&nbsp;&nbsp;&nbsp;&nbsp;"
                    + self._escape_html(str(key))
                    + ": "
                    + self._escape_html(str(value))
                )
        else:
            lines.append("&nbsp;&nbsp;&nbsp;&nbsp;" + self._escape_html(self.tr("(none)")))
        return "<br/>".join(lines)

    def _escape_html(self, text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
