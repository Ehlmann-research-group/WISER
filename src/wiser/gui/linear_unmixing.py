from typing import Optional

from PySide2.QtWidgets import QDialog

from wiser.gui.app_services import AppServices
from wiser.gui.app_state import ApplicationState
from wiser.gui.generated.linear_unmixing_dialog_ui import Ui_LinearUnmixingDialog


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
