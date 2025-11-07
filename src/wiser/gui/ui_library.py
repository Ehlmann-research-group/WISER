from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from PySide2.QtWidgets import QDialog, QLabel, QComboBox, QGridLayout

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.dataset import RasterDataSet


class ChooserDialog:
    def get_chosen_object() -> Optional[Any]:
        """
        Retrieves the chosen object after a user has accepted the dialog.

        If the user didn't accept the dialog, this will return None.
        """
        raise NotImplementedError("This function must be implemented by the subclass.")


class ChooserDialogFactory(ABC):
    @abstractmethod
    def create_chooser_dialog(self) -> QDialog:
        """
        Creates a dialog to choose some object inside of WISER like a
        dataset or spectrum.
        """
        pass


class DatasetChooserDialog(ChooserDialog, QDialog):
    def __init__(self, app_state: "ApplicationState", parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state
        self._accepted = False

        # Create widgets
        self._lbl_ds = QLabel("Chooser Dataset", self)
        self._cbox_ds = QComboBox(self)  # leave empty for now

        # Create layout
        layout = QGridLayout(self)
        layout.addWidget(self._lbl_ds, 0, 0)
        layout.addWidget(self._cbox_ds, 0, 1)

        self.setLayout(layout)

        datasets = self._app_state.get_datasets()

        for dataset in datasets:
            self._cbox_ds.addItem(dataset.get_name(), dataset.get_id())

    def get_chosen_object(self) -> "RasterDataSet":
        if not self._accepted:
            return None
        ds_id = self._cbox_ds.currentData()
        dataset = self._app_state.get_dataset(ds_id)
        return dataset

    def accept(self):
        self._accepted = True
        super().accept()

    def reject(self):
        self._accepted = False
        return super().reject()


class DatasetChooserDialogFactory(ChooserDialogFactory):
    def __init__(self, app_state: "ApplicationState", widget_parent=None):
        self._app_state = app_state
        self._widget_parent = widget_parent

    def create_chooser_dialog(self) -> DatasetChooserDialog:
        return DatasetChooserDialog(
            app_state=self._app_state,
            parent=self._widget_parent,
        )
