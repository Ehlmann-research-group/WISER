from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
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
    ACTIVE_COLUMNS = ("Task", "Activity", "Remove", "Task ID", "State")
    FINISHED_COLUMNS = ("Task", "Activity", "Remove", "Task ID", "State")

    TASK_ID_COLUMN = 3
    TASK_STATE_COLUMN = 4

    STATE_IDLE = "idle"
    STATE_RUNNING = "running"
    STATE_DONE = "done"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        self._ui = Ui_ActivityMonitor()
        self._ui.setupUi(self)

        self._next_task_id = 1
        self._task_rows: Dict[int, Dict[str, object]] = {}
        self._finished_expanded = True

        self._configure_table(self._ui.tbl_wdgt_active_tasks, self.ACTIVE_COLUMNS)
        self._configure_table(self._ui.tbl_wdgt_finished_tasks, self.FINISHED_COLUMNS)
        self._ui.btn_finished_tasks.clicked.connect(self._toggle_finished_tasks)
        self._sync_finished_section_button()

    def register_task(self, title: str, meta: Dict[str, str], cancel_callback: Callable) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1

        row = self._ui.tbl_wdgt_active_tasks.rowCount()
        self._ui.tbl_wdgt_active_tasks.insertRow(row)

        task_widget = self._build_task_widget(title=title, meta=meta, enabled=True)
        controls_widget, controls = self._build_controls_widget(
            progress_enabled=True,
            cancel_enabled=True,
            cancel_callback=lambda checked=False, task_id=task_id: self._cancel_task(task_id),
        )
        remove_button = self._build_remove_button(
            lambda checked=False, task_id=task_id: self.remove_task(task_id)
        )

        self._ui.tbl_wdgt_active_tasks.setCellWidget(row, 0, task_widget)
        self._ui.tbl_wdgt_active_tasks.setCellWidget(row, 1, controls_widget)
        self._ui.tbl_wdgt_active_tasks.setCellWidget(row, 2, remove_button)
        self._set_hidden_row_data(self._ui.tbl_wdgt_active_tasks, row, task_id, self.STATE_IDLE)

        self._task_rows[task_id] = {
            "title": title,
            "meta": dict(meta),
            "state": self.STATE_IDLE,
            "active_row": row,
            "finished_row": None,
            "cancel_callback": cancel_callback,
            "errors": [],
            "task_widget": task_widget,
            "controls": controls,
        }
        return task_id

    def set_task_running(self, task_id: int) -> None:
        self._set_task_state(task_id, self.STATE_RUNNING)

    def set_task_progress(self, task_id: int, value: int) -> None:
        task = self._get_task(task_id)
        progress_bar = task["controls"]["progress_bar"]
        progress_bar.setValue(max(0, min(100, int(value))))
        if task["state"] == self.STATE_IDLE:
            self._set_task_state(task_id, self.STATE_RUNNING)

    def append_task_error(self, task_id: int, error_message: str) -> None:
        task = self._get_task(task_id)
        errors: List[str] = task["errors"]
        errors.append(error_message)
        task["controls"]["view_errors_button"].setEnabled(True)

    def complete_task(self, task_id: int) -> None:
        task = self._get_task(task_id)
        task["controls"]["progress_bar"].setValue(100)
        self._move_task_to_finished(task_id)

    def remove_task(self, task_id: int) -> None:
        task = self._get_task(task_id)
        active_row = task["active_row"]
        finished_row = task["finished_row"]

        if active_row is not None:
            self._ui.tbl_wdgt_active_tasks.removeRow(active_row)
            self._reindex_rows(self._ui.tbl_wdgt_active_tasks, "active_row")

        if finished_row is not None:
            self._ui.tbl_wdgt_finished_tasks.removeRow(finished_row)
            self._reindex_rows(self._ui.tbl_wdgt_finished_tasks, "finished_row")

        self._task_rows.pop(task_id, None)

    def _configure_table(self, table: QTableWidget, columns: tuple[str, ...]) -> None:
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(list(columns))
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(0, table.horizontalHeader().Stretch)
        table.horizontalHeader().setSectionResizeMode(1, table.horizontalHeader().ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, table.horizontalHeader().ResizeToContents)
        table.setColumnHidden(self.TASK_ID_COLUMN, True)
        table.setColumnHidden(self.TASK_STATE_COLUMN, True)
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
    ) -> tuple[QWidget, Dict[str, QWidget]]:
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
        container.setProperty("controls", controls)
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

    def _set_hidden_row_data(self, table: QTableWidget, row: int, task_id: int, state: str) -> None:
        task_id_item = QTableWidgetItem(str(task_id))
        task_id_item.setData(Qt.UserRole, task_id)
        state_item = QTableWidgetItem(state)
        state_item.setData(Qt.UserRole, state)
        table.setItem(row, self.TASK_ID_COLUMN, task_id_item)
        table.setItem(row, self.TASK_STATE_COLUMN, state_item)

    def _set_task_state(self, task_id: int, state: str) -> None:
        task = self._get_task(task_id)
        task["state"] = state

        active_row = task["active_row"]
        if active_row is not None:
            active_item = self._ui.tbl_wdgt_active_tasks.item(active_row, self.TASK_STATE_COLUMN)
            if active_item is not None:
                active_item.setText(state)
                active_item.setData(Qt.UserRole, state)

        finished_row = task["finished_row"]
        if finished_row is not None:
            finished_item = self._ui.tbl_wdgt_finished_tasks.item(finished_row, self.TASK_STATE_COLUMN)
            if finished_item is not None:
                finished_item.setText(state)
                finished_item.setData(Qt.UserRole, state)

    def _move_task_to_finished(self, task_id: int) -> None:
        task = self._get_task(task_id)
        active_row = task["active_row"]
        if active_row is None:
            self._set_task_state(task_id, self.STATE_DONE)
            return

        controls = task["controls"]
        progress_value = controls["progress_bar"].value()

        finished_row = self._ui.tbl_wdgt_finished_tasks.rowCount()
        self._ui.tbl_wdgt_finished_tasks.insertRow(finished_row)

        finished_task_widget = self._build_task_widget(
            title=task["title"],
            meta=task["meta"],
            enabled=False,
        )
        finished_controls_widget, finished_controls = self._build_controls_widget(
            progress_enabled=False,
            cancel_enabled=False,
            cancel_callback=lambda checked=False: None,
        )
        finished_controls["progress_bar"].setValue(progress_value)
        remove_button = self._build_remove_button(
            lambda checked=False, task_id=task_id: self.remove_task(task_id)
        )

        self._ui.tbl_wdgt_finished_tasks.setCellWidget(finished_row, 0, finished_task_widget)
        self._ui.tbl_wdgt_finished_tasks.setCellWidget(finished_row, 1, finished_controls_widget)
        self._ui.tbl_wdgt_finished_tasks.setCellWidget(finished_row, 2, remove_button)
        self._set_hidden_row_data(self._ui.tbl_wdgt_finished_tasks, finished_row, task_id, self.STATE_DONE)

        self._ui.tbl_wdgt_active_tasks.removeRow(active_row)
        task["active_row"] = None
        task["finished_row"] = finished_row
        task["task_widget"] = finished_task_widget
        task["controls"] = finished_controls

        self._reindex_rows(self._ui.tbl_wdgt_active_tasks, "active_row")
        self._reindex_rows(self._ui.tbl_wdgt_finished_tasks, "finished_row")
        self._set_task_state(task_id, self.STATE_DONE)

        self._sync_finished_section_button()
        if not self._finished_expanded:
            self._toggle_finished_tasks()

    def _cancel_task(self, task_id: int) -> None:
        task = self._get_task(task_id)
        cancel_callback = task["cancel_callback"]
        if callable(cancel_callback):
            cancel_callback()

    def _show_errors_for_controls(self, controls_container: QWidget) -> None:
        task = self._find_task_by_controls(controls_container)
        if task is None:
            return

        errors: List[str] = task["errors"]
        message = "\n\n".join(errors) if errors else self.tr("No errors recorded for this task.")

        from PySide2.QtWidgets import QMessageBox

        QMessageBox.information(self, self.tr("Task Errors"), message)

    def _find_task_by_controls(self, controls_container: QWidget) -> Optional[Dict[str, object]]:
        for task in self._task_rows.values():
            controls = task["controls"]
            if controls["container"] is controls_container:
                return task
        return None

    def _toggle_finished_tasks(self) -> None:
        self._finished_expanded = not self._finished_expanded
        self._ui.tbl_wdgt_finished_tasks.setVisible(self._finished_expanded)
        self._sync_finished_section_button()

    def _sync_finished_section_button(self) -> None:
        arrow = "\u25bc" if self._finished_expanded else "\u25b6"
        count = self._ui.tbl_wdgt_finished_tasks.rowCount()
        self._ui.btn_finished_tasks.setText(self.tr(f"Finished Tasks ({count}) {arrow}"))

    def _reindex_rows(self, table: QTableWidget, row_key: str) -> None:
        row_lookup: Dict[int, int] = {}
        for row in range(table.rowCount()):
            item = table.item(row, self.TASK_ID_COLUMN)
            if item is None:
                continue
            task_id = int(item.data(Qt.UserRole))
            row_lookup[task_id] = row

        for task_id, row in row_lookup.items():
            if task_id in self._task_rows:
                self._task_rows[task_id][row_key] = row

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

    def _get_task(self, task_id: int) -> Dict[str, object]:
        if task_id not in self._task_rows:
            raise KeyError(f"Unknown task id: {task_id}")
        return self._task_rows[task_id]
