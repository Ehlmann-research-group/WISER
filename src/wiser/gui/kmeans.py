from enum import Enum
from typing import Optional, TYPE_CHECKING

from PySide2.QtGui import QIntValidator, QDoubleValidator
from PySide2.QtWidgets import QDialog

from wiser.gui.app_services import AppServices
from wiser.gui.app_state import ApplicationState
from wiser.gui.generated.kmeans_dialog_ui import Ui_KMeansDialog

if TYPE_CHECKING:
    pass


class KMeansInitMethod(Enum):
    KMEANS_PLUS_PLUS = "k-means++"
    RANDOM = "random"
    MANUAL = "manual"


class KMeansAlgorithm(Enum):
    LLOYD = "lloyd"
    ELKAN = "elkan"


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

        self._init_cbox_init_method()
        self._init_cbox_algo()
        self._init_validators()
        self._ui.cbox_init_method.currentIndexChanged.connect(self._on_init_method_changed)

    def _init_cbox_init_method(self) -> None:
        cbox = self._ui.cbox_init_method
        cbox.clear()
        for method in KMeansInitMethod:
            cbox.addItem(method.value, method)
        self._ui.tbl_wdgt_init_spectra.setVisible(False)

    def _init_cbox_algo(self) -> None:
        cbox = self._ui.cbox_algo
        cbox.clear()
        for algo in KMeansAlgorithm:
            cbox.addItem(algo.value, algo)

    def _init_validators(self) -> None:
        # Positive integers only (minimum 1)
        self._ui.ledit_k_clusters.setValidator(QIntValidator(1, 2_147_483_647, self))
        self._ui.ledit_num_inits.setValidator(QIntValidator(1, 2_147_483_647, self))
        self._ui.ledit_max_iter.setValidator(QIntValidator(1, 2_147_483_647, self))

        # Any integer (positive or negative)
        self._ui.ledit_seed.setValidator(QIntValidator(-2_147_483_648, 2_147_483_647, self))

        # Positive float
        pos_float_validator = QDoubleValidator(0.0, 1.0e308, 10, self)
        pos_float_validator.setNotation(QDoubleValidator.ScientificNotation)
        self._ui.ledit_tol.setValidator(pos_float_validator)

    def _on_init_method_changed(self, index: int) -> None:
        method = self._ui.cbox_init_method.itemData(index)
        is_manual = method is KMeansInitMethod.MANUAL

        self._ui.tbl_wdgt_init_spectra.setVisible(is_manual)

        # Disable num_inits and seed when manual (centroid positions are fixed)
        self._ui.ledit_num_inits.setEnabled(not is_manual)
        self._ui.lbl_num_inits.setEnabled(not is_manual)
        self._ui.ledit_seed.setEnabled(not is_manual)
        self._ui.lbl_seed.setEnabled(not is_manual)

    def _toggle_advanced_options(self) -> None:
        visible = not self._ui.wdgt_advanced_options.isVisible()
        self._ui.wdgt_advanced_options.setVisible(visible)
        arrow = "\u25bc" if visible else "\u25b6"
        self._ui.btn_advanced_options.setText(f"Advanced Options {arrow}")

    def get_k_clusters(self) -> Optional[int]:
        text = self._ui.ledit_k_clusters.text().strip()
        return int(text) if text else None

    def get_init_method(self) -> KMeansInitMethod:
        return self._ui.cbox_init_method.currentData()

    def get_num_inits(self) -> Optional[int]:
        text = self._ui.ledit_num_inits.text().strip()
        return int(text) if text else None

    def get_max_iter(self) -> Optional[int]:
        text = self._ui.ledit_max_iter.text().strip()
        return int(text) if text else None

    def get_tol(self) -> Optional[float]:
        text = self._ui.ledit_tol.text().strip()
        return float(text) if text else None

    def get_seed(self) -> Optional[int]:
        text = self._ui.ledit_seed.text().strip()
        return int(text) if text else None

    def get_algorithm(self) -> KMeansAlgorithm:
        return self._ui.cbox_algo.currentData()

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
