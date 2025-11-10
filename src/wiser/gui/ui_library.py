from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from PySide2.QtWidgets import (
    QDialog,
    QLabel,
    QComboBox,
    QGridLayout,
    QDialogButtonBox,
    QSpinBox,
)
from wiser.raster.dataset import RasterDataSet, RasterDataBand

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.spectrum import Spectrum


class SingleItemChooserDialog(QDialog):
    def __init__(self, dialog_name: str, app_state: "ApplicationState", parent=None):
        super().__init__(parent=parent)

        if app_state is None or dialog_name is None:
            return

        self._app_state = app_state
        self._accepted = False

        self._lbl = QLabel(self.tr(dialog_name), self)
        self._cbox = QComboBox(self)  # leave empty, subclass will populate

        layout = QGridLayout(self)
        layout.addWidget(self._lbl, 0, 0)
        layout.addWidget(self._cbox, 0, 1)

        self._create_button_box(layout=layout)

        self.setLayout(layout)

    def _create_button_box(
        self,
        layout: QGridLayout,
        bbox_row=1,
        bbox_col_span=2,
    ):
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )

        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        # Put buttons on next row spanning both columns
        layout.addWidget(self._button_box, bbox_row, 0, 1, bbox_col_span)

    def get_chosen_object(self) -> Optional[Any]:
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
    def __init__(self, app_state: "ApplicationState", widget_parent=None):
        self._app_state = app_state
        self._widget_parent = widget_parent

    @abstractmethod
    def create_chooser_dialog(self) -> QDialog:
        """
        Creates a dialog to choose some object inside of WISER like a
        dataset or spectrum.
        """
        pass


class SpectrumChooserDialog(SingleItemChooserDialog):
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
        spectrum = self._app_state.get_spectrum(spectrum_id)
        return spectrum


class ROIChooserDialog(SingleItemChooserDialog):
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
        roi = self._app_state.get_roi(**kwargs)
        return roi


class DatasetChooserDialog(SingleItemChooserDialog):
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


class BandChooserDialog(SingleItemChooserDialog):
    def __init__(self, app_state: "ApplicationState", parent=None):
        super().__init__(dialog_name=None, app_state=None, parent=parent)

        self._app_state = app_state

        # Dataset row
        self._lbl_dataset = QLabel(self.tr("Dataset"), self)
        self._cbox_dataset = QComboBox(self)

        # Band row
        self._lbl_band = QLabel(self.tr("Band #"), self)
        self._sbox_band = QSpinBox(self)
        self._sbox_band.setMinimum(0)  # set max later based on dataset
        self._cbox_band = QComboBox(self)

        layout = QGridLayout(self)
        layout.addWidget(self._lbl_dataset, 0, 0)
        layout.addWidget(self._cbox_dataset, 0, 1, 1, 2)

        layout.addWidget(self._lbl_band, 1, 0)
        layout.addWidget(self._sbox_band, 1, 1)
        layout.addWidget(self._cbox_band, 1, 2)

        self._create_button_box(layout=layout, bbox_row=2, bbox_col_span=3)

        self.setLayout(layout)

        # Populate combo box
        datasets = self._app_state.get_datasets()
        for dataset in datasets:
            self._cbox_dataset.addItem(dataset.get_name(), dataset.get_id())

        self._cbox_dataset.currentIndexChanged.connect(self._on_cbox_dataset_changed)

        # Initial band chooser sync
        self._sync_band_chooser()

        # Keeps the combo box and spin box in sync
        self._cbox_band.currentIndexChanged.connect(self._sync_cbox_to_sbox)
        self._sbox_band.valueChanged.connect(self._sync_sbox_to_cbox)

    def _on_cbox_dataset_changed(self, checked=False):
        self._sync_band_chooser()

    def _sync_band_chooser(self):
        ds_id = self._cbox_dataset.currentData()
        if self._app_state.has_dataset(ds_id):
            dataset = self._app_state.get_dataset(ds_id)
            bands = dataset.band_list()
            descriptions = list([band["description"] for band in bands])
            if descriptions[0]:
                band_descriptions = list(
                    (
                        f"Band {descriptions.index(descr)}: " + descr,
                        descriptions.index(descr),
                    )
                    for descr in descriptions
                )
            else:
                band_descriptions = list((f"Band {i}", i) for i in range(len(descriptions)))
            self._cbox_band.clear()
            for descr, index in band_descriptions:
                self._cbox_band.addItem(self.tr(f"{descr}"), index)
            self._sbox_band.setRange(0, len(band_descriptions) - 1)
        else:
            self._cbox_band.clear()

    def _sync_cbox_to_sbox(self, checked=False):
        idx = self._cbox_band.currentIndex()
        self._sbox_band.setValue(idx)

    def _sync_sbox_to_cbox(self, checked=False):
        idx = self._sbox_band.value()
        self._cbox_band.setCurrentIndex(idx)

    def get_chosen_object(self) -> RasterDataBand:
        if not self._accepted:
            return None
        dataset = self._app_state.get_dataset(self._cbox_dataset.currentData())
        band_id = self._cbox_band.currentData()
        band = RasterDataBand(dataset=dataset, band_index=band_id)
        return band


class DatasetChooserDialogFactory(ChooserDialogFactory):
    def create_chooser_dialog(self) -> DatasetChooserDialog:
        return DatasetChooserDialog(
            app_state=self._app_state,
            parent=self._widget_parent,
        )


class SpectrumChooserDialogFactory(ChooserDialogFactory):
    def create_chooser_dialog(self) -> SpectrumChooserDialog:
        return SpectrumChooserDialog(
            app_state=self._app_state,
            parent=self._widget_parent,
        )


class ROIChooserDialogFactory(ChooserDialogFactory):
    def create_chooser_dialog(self) -> ROIChooserDialog:
        return ROIChooserDialog(
            app_state=self._app_state,
            parent=self._widget_parent,
        )


class BandChooserDialogFactory(ChooserDialogFactory):
    def create_chooser_dialog(self) -> BandChooserDialog:
        return BandChooserDialog(
            app_state=self._app_state,
            parent=self._widget_parent,
        )
