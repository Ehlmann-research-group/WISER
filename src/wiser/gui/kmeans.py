from typing import Optional, TYPE_CHECKING

from PySide2.QtWidgets import QDialog

from wiser.gui.app_services import AppServices
from wiser.gui.app_state import ApplicationState
from wiser.gui.generated.kmeans_dialog_ui import Ui_KMeansDialog

if TYPE_CHECKING:
    pass


class KMeansDialog(QDialog):
    def __init__(
        self,
        app_state: ApplicationState,
        app_services: AppServices,
        parent=None,
    ):
        super().__init__(parent=parent)
        self._app_state = app_state
        self._app_services = app_services
        self._selected_dataset_id: Optional[int] = None

        self._ui = Ui_KMeansDialog()
        self._ui.setupUi(self)

        self._ui.wdgt_advanced_options.setVisible(False)
        self._ui.btn_advanced_options.setText("Advanced Options \u25b6")
        self._ui.btn_advanced_options.clicked.connect(self._toggle_advanced_options)

    def _toggle_advanced_options(self) -> None:
        visible = not self._ui.wdgt_advanced_options.isVisible()
        self._ui.wdgt_advanced_options.setVisible(visible)
        arrow = "\u25bc" if visible else "\u25b6"
        self._ui.btn_advanced_options.setText(f"Advanced Options {arrow}")

    def show_kmeans(self, dataset_id: Optional[int] = None) -> None:
        pass

    def showEvent(self, event):
        self.show_kmeans(dataset_id=self._selected_dataset_id)
        super().showEvent(event)

    def select_dataset(self, dataset_id: Optional[int]) -> None:
        self._selected_dataset_id = dataset_id
        self.show_kmeans(dataset_id=dataset_id)

    def accept(self):
        super().accept()
