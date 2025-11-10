from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from PySide2.QtWidgets import QDialog, QLabel, QComboBox, QGridLayout

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.spectrum import Spectrum


class ChooserDialog(QDialog):
    def __init__(self, dialog_name: str, app_state: "ApplicationState", parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state
        self._accepted = False

        # Create widgets
        self._lbl = QLabel(self.tr(dialog_name), self)
        self._cbox = QComboBox(self)  # leave empty for now

        # Create layout
        layout = QGridLayout(self)
        layout.addWidget(self._lbl, 0, 0)
        layout.addWidget(self._cbox, 0, 1)

        self.setLayout(layout)

    def get_chosen_object() -> Optional[Any]:
        """
        Retrieves the chosen object after a user has accepted the dialog.

        If the user didn't accept the dialog, this will return None.
        """
        raise NotImplementedError("This function must be implemented by the subclass.")

    def accept(self):
        self._accepted = True
        super().accept()

    def reject(self):
        self._accepted = False
        return super().reject()


class ChooserDialogFactory(ABC):
    @abstractmethod
    def create_chooser_dialog(self) -> QDialog:
        """
        Creates a dialog to choose some object inside of WISER like a
        dataset or spectrum.
        """
        pass


class SpectrumChooserDialog(ChooserDialog):
    def __init__(self, app_state: "ApplicationState", parent=None):
        super().__init__(
            dialog_name="Spectrum Chooser",
            app_state=app_state,
            parent=parent,
        )

        spectra = self._app_state.get_all_spectra()

        for spectrum in spectra.values():
            self._cbox.addItem(spectrum.get_name(), spectrum.get_id())

    def get_chosen_object(self) -> Optional["Spectrum"]:
        if not self._accepted:
            return None
        spectrum_id = self._cbox.currentData()
        spectrum = self._app_state.get_dataset(spectrum_id)
        return spectrum


class ROIChooserDialog(ChooserDialog):
    def __init__(self, app_state: "ApplicationState", parent=None):
        super().__init__(
            dialog_name="ROI (Region Of Interest) Chooser",
            app_state=app_state,
            parent=parent,
        )

        rois = self._app_state.get_rois()

        for roi in rois:
            self._cbox.addItem(roi.get_name(), roi.get_id())

    def get_chosen_object(self) -> Optional["RasterDataSet"]:
        if not self._accepted:
            return None
        roi_id = self._cbox.currentData()
        kwargs = {"id": roi_id}
        roi = self._app_state.get_roi(kwargs)
        return roi


class DatasetChooserDialog(ChooserDialog):
    def __init__(self, app_state: "ApplicationState", parent=None):
        super().__init__(
            dialog_name="Dataset Chooser",
            app_state=app_state,
            parent=parent,
        )

        datasets = self._app_state.get_datasets()

        for dataset in datasets:
            self._cbox.addItem(dataset.get_name(), dataset.get_id())

    def get_chosen_object(self) -> Optional["RasterDataSet"]:
        if not self._accepted:
            return None
        ds_id = self._cbox.currentData()
        dataset = self._app_state.get_dataset(ds_id)
        return dataset


class DatasetChooserDialogFactory(ChooserDialogFactory):
    def __init__(self, app_state: "ApplicationState", widget_parent=None):
        self._app_state = app_state
        self._widget_parent = widget_parent

    def create_chooser_dialog(self) -> DatasetChooserDialog:
        return DatasetChooserDialog(
            app_state=self._app_state,
            parent=self._widget_parent,
        )
