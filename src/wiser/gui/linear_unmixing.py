from typing import Callable, Optional

from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHeaderView,
    QPushButton,
    QStyle,
    QTableWidgetItem,
)

from wiser.gui.app_services import AppServices
from wiser.gui.app_state import ApplicationState
from wiser.gui.generated.linear_unmixing_dialog_ui import Ui_LinearUnmixingDialog


_ENDMEMBER_NAME_COL = 0
_ENDMEMBER_REMOVE_COL = 1


class LinearUnmixingDialog(QDialog):
    """Dialog for configuring and launching a linear unmixing task."""

    def __init__(
        self,
        app_state: ApplicationState,
        app_services: AppServices,
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.setModal(False)
        self._app_state = app_state
        self._app_services = app_services
        self._selected_dataset_id: Optional[int] = None

        self._ui = Ui_LinearUnmixingDialog()
        self._ui.setupUi(self)

        self._configure_endmember_table()
        self._ui.btn_add_collected_spec.clicked.connect(self._on_add_collected_spec)

        app_state.dataset_added.connect(self._on_datasets_changed)
        app_state.dataset_removed.connect(self._on_datasets_changed)

    def show_linear_unmixing(self, dataset_id: Optional[int] = None) -> None:
        cbox = self._ui.cbox_input_ds
        cbox.clear()
        cbox.addItem(self.tr("(no data)"), -1)
        for dataset in self._app_state.get_datasets():
            cbox.addItem(dataset.get_name(), dataset.get_id())

        if dataset_id is not None:
            index = cbox.findData(dataset_id)
            cbox.setCurrentIndex(index if index >= 0 else 0)
        else:
            cbox.setCurrentIndex(0)

    def showEvent(self, event):
        self.show_linear_unmixing(dataset_id=self._selected_dataset_id)
        super().showEvent(event)

    def select_dataset(self, dataset_id: Optional[int]) -> None:
        self._selected_dataset_id = dataset_id
        self.show_linear_unmixing(dataset_id=dataset_id)

    def _on_datasets_changed(self, *_args) -> None:
        current_id = self._ui.cbox_input_ds.currentData()
        preserve_id = current_id if (current_id is not None and int(current_id) >= 0) else None
        self.show_linear_unmixing(dataset_id=preserve_id)

    def get_selected_dataset(self):
        dataset_id = self._ui.cbox_input_ds.currentData()
        if dataset_id is None or int(dataset_id) < 0:
            return None
        return self._app_state.get_dataset(int(dataset_id))

    def _configure_endmember_table(self) -> None:
        table = self._ui.tbl_wdgt_endmembers
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels([self.tr("Spectrum"), self.tr("")])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        header = table.horizontalHeader()
        header.setSectionResizeMode(_ENDMEMBER_NAME_COL, QHeaderView.Stretch)
        header.setSectionResizeMode(_ENDMEMBER_REMOVE_COL, QHeaderView.ResizeToContents)

    def _on_add_collected_spec(self) -> None:
        spec = self._app_state.choose_spectrum_ui()
        if spec is None:
            return

        if spec.get_id() is None:
            spec.set_id(self._app_state.take_next_id())
        spec_id = int(spec.get_id())

        if self._find_endmember_row(spec_id) is not None:
            return

        table = self._ui.tbl_wdgt_endmembers
        row = table.rowCount()
        table.insertRow(row)

        name_item = QTableWidgetItem(spec.get_name() or self.tr("<unnamed>"))
        name_item.setData(Qt.UserRole, spec_id)
        table.setItem(row, _ENDMEMBER_NAME_COL, name_item)

        remove_button = self._build_remove_button(
            lambda checked=False, sid=spec_id: self._remove_endmember(sid)
        )
        table.setCellWidget(row, _ENDMEMBER_REMOVE_COL, remove_button)

    def _find_endmember_row(self, spec_id: int) -> Optional[int]:
        table = self._ui.tbl_wdgt_endmembers
        for row in range(table.rowCount()):
            item = table.item(row, _ENDMEMBER_NAME_COL)
            if item is not None and item.data(Qt.UserRole) == spec_id:
                return row
        return None

    def _remove_endmember(self, spec_id: int) -> None:
        row = self._find_endmember_row(spec_id)
        if row is not None:
            self._ui.tbl_wdgt_endmembers.removeRow(row)

    def _build_remove_button(self, callback: Callable) -> QPushButton:
        button = QPushButton()
        button.setToolTip(self.tr("Remove endmember"))
        trash_icon_enum = getattr(QStyle, "SP_TrashIcon", None)
        icon = self.style().standardIcon(trash_icon_enum) if trash_icon_enum is not None else None
        if icon is None or icon.isNull():
            button.setText(self.tr("Remove"))
        else:
            button.setIcon(icon)
        button.clicked.connect(callback)
        return button
