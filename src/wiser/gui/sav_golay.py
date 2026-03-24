from typing import Dict, TYPE_CHECKING, Any

import numpy as np
from PySide2.QtCore import QObject, Signal, Slot
from PySide2.QtWidgets import QDialog, QMessageBox, QDialogButtonBox
from scipy.signal import savgol_filter

from wiser import plugins
from wiser.bandmath.types import VariableType
from wiser.gui.app_services import AppServices
from wiser.gui.generated.sav_golay_filter_dialog_ui import Ui_SavGolayFilter
from wiser.raster.spectrum import NumPyArraySpectrum, Spectrum
from wiser.utils.primitives import DataMeta, DataRef, DatasetRegionRef, PriorityClass
from wiser.utils.storage_layer import ExternalRasterHandle
from wiser.utils.task_stage_utils import (
    get_good_band_runs,
    get_savgol_filter_pipeline,
)
from wiser.utils.task_system import SemanticTask
from wiser.utils.worker_runtime import get_process_storage_client

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.dataset import RasterDataSet


def _good_band_cap_from_bad_bands(bad_bands: Any, band_count: int) -> int:
    if bad_bands is None:
        return band_count
    runs = get_good_band_runs(np.asarray(bad_bands))
    if len(runs) == 0:
        return 0
    return min(end - start for start, end in runs)


def savgol_filter_spectrum(
    spectrum: Spectrum,
    *,
    window_length: int,
    polyorder: int,
) -> NumPyArraySpectrum:
    spectrum_arr = np.asarray(spectrum.get_spectrum(), dtype=np.float32)
    filtered_arr = np.asarray(
        savgol_filter(
            spectrum_arr,
            window_length=window_length,
            polyorder=polyorder,
            deriv=0,
            axis=0,
            mode="interp",
        ),
        dtype=np.float32,
    )
    result = NumPyArraySpectrum(
        filtered_arr,
        name=None,
        source_name=spectrum.get_source_name(),
        wavelengths=spectrum.get_wavelengths(),
    )
    result.set_name(f"Savitzky-Golay on {spectrum.get_name()}")
    if spectrum.get_bad_bands() is not None:
        result.set_bad_bands(np.asarray(spectrum.get_bad_bands()))
    return result


class SavGolaySemanticTask(QObject, SemanticTask):
    result_ready = Signal(object, object)

    def __init__(
        self,
        app_state: "ApplicationState",
        source_dataset: "RasterDataSet",
        input_ref: DataRef,
        window_length: int,
        polyorder: int,
        output_ref_name: str = "savgol_filtered_dataset",
    ):
        QObject.__init__(self)
        SemanticTask.__init__(
            self,
            priority_class=PriorityClass.BACKGROUND,
            input_ref=input_ref,
            algorithm_pipeline=get_savgol_filter_pipeline(
                dataset_ref=input_ref,
                window_length=window_length,
                polyorder=polyorder,
                output_ref_name=output_ref_name,
            ),
            task_title="Savitzky-Golay Filter",
            task_variables={
                "Dataset": source_dataset.get_name(),
                "Window Length": window_length,
                "Polyorder": polyorder,
            },
        )
        self.id = app_state.take_next_id()
        self._app_state = app_state
        self._source_dataset = source_dataset
        self._window_length = window_length
        self._polyorder = polyorder
        self._output_ref_name = output_ref_name
        self.result_ready.connect(self._load_result_into_wiser)

    def completion_callback(self, bindings: Dict[str, DataRef]) -> None:
        output_ref = bindings.get(self._output_ref_name)
        if output_ref is None:
            raise KeyError(f"Missing Savitzky-Golay output binding: {self._output_ref_name}")

        storage_client = get_process_storage_client()
        output_meta = storage_client.get_meta(output_ref)
        height, width, bands = output_meta.shape
        output_region = DatasetRegionRef(y0=0, y1=height, x0=0, x1=width, b0=0, b1=bands)
        reduced_data, _ = storage_client.read_region(output_ref, output_region, filter_data=False)
        self.result_ready.emit(np.asarray(reduced_data), output_meta)

    @Slot(object, object)
    def _load_result_into_wiser(self, reduced_data: object, output_meta: object) -> None:
        reduced_array = np.asarray(reduced_data)
        reduced_array_by_band = reduced_array.transpose(2, 0, 1)

        loader = self._app_state.get_loader()
        cache = self._app_state.get_cache()
        reduced_dataset = loader.dataset_from_numpy_array(reduced_array_by_band, cache)

        source_name = self._source_dataset.get_name() or "Dataset"
        reduced_dataset.set_name(self._app_state.unique_dataset_name(f"Savitzky-Golay on {source_name}"))
        reduced_dataset.set_description(self._source_dataset.get_description())
        default_display_bands = self._source_dataset.default_display_bands()
        if default_display_bands is not None:
            reduced_dataset.set_default_display_bands(default_display_bands)

        if self._source_dataset.get_spatial_metadata().get_spatial_ref():
            reduced_dataset.copy_spatial_metadata(self._source_dataset.get_spatial_metadata())
        if self._source_dataset.has_wavelengths():
            reduced_dataset.copy_spectral_metadata(self._source_dataset.get_spectral_metadata())

        if isinstance(output_meta, DataMeta):
            reduced_dataset.set_data_ignore_value(output_meta.nodata)
            if output_meta.bad_bands is not None:
                reduced_dataset.set_bad_bands(np.asarray(output_meta.bad_bands).astype(int).tolist())
        else:
            reduced_dataset.set_data_ignore_value(self._source_dataset.get_data_ignore_value())
            if self._source_dataset.get_bad_bands() is not None:
                reduced_dataset.set_bad_bands(self._source_dataset.get_bad_bands())

        self._app_state.add_dataset(reduced_dataset, view_dataset=False)


class SavGolayDialog(QDialog):
    def __init__(
        self,
        app_state: "ApplicationState",
        app_services: AppServices,
        target_type: VariableType,
        target_id: int,
        parent=None,
    ):
        super().__init__(parent=parent)
        self._app_state = app_state
        self._app_services = app_services
        self._target_type = target_type
        self._target_id = target_id
        self._last_savgol_task = None

        self._ui = Ui_SavGolayFilter()
        self._ui.setupUi(self)

        self._ui.sbox_window_len.setMinimum(1)
        self._ui.sbox_window_len.setSingleStep(2)
        self._ui.sbox_poly_order.setMinimum(1)
        self._ui.cbox_choice.currentIndexChanged.connect(self._on_selection_changed)
        self._ui.sbox_window_len.valueChanged.connect(self._on_window_length_changed)
        self._populate_choices()

    def _is_spectrum_mode(self) -> bool:
        return self._target_type == VariableType.SPECTRUM

    def _is_dataset_mode(self) -> bool:
        return self._target_type in (VariableType.IMAGE_CUBE, VariableType.IMAGE_CUBE_DATASET)

    def _populate_choices(self) -> None:
        self._ui.cbox_choice.clear()

        if self._is_spectrum_mode():
            self._ui.lbl_choose_ds_spec.setText("Choose Spectrum")
            objects = list(self._app_state.get_all_spectra().values())
        elif self._is_dataset_mode():
            self._ui.lbl_choose_ds_spec.setText("Choose Dataset")
            objects = self._app_state.get_datasets()
        else:
            raise ValueError(f"Unsupported Savitzky-Golay target type: {self._target_type}")

        for obj in objects:
            name = obj.get_name() or "<unnamed>"
            obj_id = obj.get_id()
            if obj_id is None:
                obj_id = self._app_state.take_next_id()
                obj.set_id(obj_id)
                if self._is_spectrum_mode():
                    self._app_state.get_all_spectra()[obj_id] = obj
            self._ui.cbox_choice.addItem(name, obj_id)

        has_choices = self._ui.cbox_choice.count() > 0
        self._ui.cbox_choice.setEnabled(has_choices)
        self._ui.buttonBox.button(QDialogButtonBox.Ok).setEnabled(has_choices)
        if has_choices:
            target_index = self._ui.cbox_choice.findData(self._target_id)
            if target_index < 0:
                target_index = 0
            self._ui.cbox_choice.setCurrentIndex(target_index)
            self._on_selection_changed(target_index)
        else:
            self._ui.cbox_choice.addItem("<none available>")

    def _selected_object(self):
        index = self._ui.cbox_choice.currentIndex()
        if index < 0:
            return None
        obj_id = self._ui.cbox_choice.itemData(index)
        if obj_id is None:
            return None
        if self._is_spectrum_mode():
            return self._app_state.get_spectrum(int(obj_id))
        return self._app_state.get_dataset(int(obj_id))

    def _current_good_band_cap(self) -> int:
        obj = self._selected_object()
        if obj is None:
            return 1

        if self._is_spectrum_mode():
            band_count = obj.num_bands()
            bad_bands = obj.get_bad_bands()
        else:
            band_count = obj.num_bands()
            bad_bands = obj.get_bad_bands()

        cap = _good_band_cap_from_bad_bands(bad_bands, band_count)
        return max(1, int(cap))

    def _on_selection_changed(self, index: int) -> None:
        _ = index
        cap = self._current_good_band_cap()
        self._ui.sbox_window_len.setMaximum(cap)
        current_window = self._ui.sbox_window_len.value()
        if current_window <= 0:
            current_window = 1
        if current_window > cap:
            current_window = cap
        if current_window % 2 == 0:
            current_window = max(1, current_window - 1)
        self._ui.sbox_window_len.setValue(current_window)
        self._on_window_length_changed(current_window)

    def _on_window_length_changed(self, value: int) -> None:
        value = max(1, int(value))
        if value % 2 == 0:
            value = max(1, value - 1)
            self._ui.sbox_window_len.blockSignals(True)
            self._ui.sbox_window_len.setValue(value)
            self._ui.sbox_window_len.blockSignals(False)

        poly_max = max(1, value - 1)
        self._ui.sbox_poly_order.setMaximum(poly_max)
        poly_value = self._ui.sbox_poly_order.value()
        if poly_value <= 0:
            poly_value = 1
        if poly_value > poly_max:
            poly_value = poly_max
        self._ui.sbox_poly_order.setValue(poly_value)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Savitzky-Golay Filter", message, QMessageBox.Ok)

    def accept(self) -> None:
        selected = self._selected_object()
        if selected is None:
            self._show_error("Please select a dataset or spectrum.")
            return

        window_length = int(self._ui.sbox_window_len.value())
        polyorder = int(self._ui.sbox_poly_order.value())
        cap = self._current_good_band_cap()

        if window_length <= 0:
            self._show_error("Window length must be greater than 0.")
            return
        if polyorder <= 0:
            self._show_error("Polynomial order must be greater than 0.")
            return
        if window_length % 2 == 0:
            self._show_error("Window length must be odd.")
            return
        if polyorder >= window_length:
            self._show_error("Polynomial order must be less than window length.")
            return
        if window_length > cap:
            self._show_error(f"Window length must be <= the shortest good-band run ({cap}).")
            return

        try:
            if self._is_spectrum_mode():
                filtered_spec = savgol_filter_spectrum(
                    selected,
                    window_length=window_length,
                    polyorder=polyorder,
                )
                self._app_state.collect_spectrum(filtered_spec)
            else:
                dataset_ref = self._app_services.storage_service.register_external(
                    ExternalRasterHandle(dataset_obj=selected)
                )
                savgol_task = SavGolaySemanticTask(
                    app_state=self._app_state,
                    source_dataset=selected,
                    input_ref=dataset_ref,
                    window_length=window_length,
                    polyorder=polyorder,
                )
                self._last_savgol_task = savgol_task
                task_plan = self._app_services.task_planner.plan_semantic_task(savgol_task)
                self._app_services.task_manager.register_and_submit_task_plan(
                    self._app_services.scheduler, task_plan
                )
        except Exception as exc:
            self._show_error(str(exc))
            return

        super().accept()


class SavGolayPlugin(plugins.ContextMenuPlugin):
    def add_context_menu_items(self, context_type: plugins.ContextMenuType, context_menu, context):
        if context_type == plugins.ContextMenuType.DATASET_PICK:
            act = context_menu.addAction(context_menu.tr("Savitzky-Golay Filter"))
            act.triggered.connect(
                lambda checked=False: self._show_dialog(context, VariableType.IMAGE_CUBE_DATASET)
            )

        if context_type == plugins.ContextMenuType.SPECTRUM_PICK:
            act = context_menu.addAction(context_menu.tr("Savitzky-Golay Filter"))
            act.triggered.connect(lambda checked=False: self._show_dialog(context, VariableType.SPECTRUM))

    def _show_dialog(self, context: Dict[str, Any], target_type: VariableType) -> None:
        app_state = context["wiser"]
        app_services = context.get("app_services")
        if app_services is None:
            app = getattr(app_state, "_app", None)
            app_services = getattr(app, "_app_services", None)
        if app_services is None:
            raise RuntimeError("Savitzky-Golay dialog requires app services")

        if target_type == VariableType.SPECTRUM:
            spectrum = context.get("spectrum")
            if spectrum is None:
                raise ValueError("Missing spectrum in Savitzky-Golay spectrum context")
            target_id = spectrum.get_id()
            if target_id is None:
                target_id = app_state.take_next_id()
                spectrum.set_id(target_id)
                app_state.get_all_spectra()[target_id] = spectrum
        else:
            dataset = context.get("dataset")
            if dataset is None:
                raise ValueError("Missing dataset in Savitzky-Golay dataset context")
            target_id = dataset.get_id()
            if target_id is None:
                target_id = app_state.take_next_id()
                dataset.set_id(target_id)

        dialog = SavGolayDialog(
            app_state=app_state,
            app_services=app_services,
            target_type=target_type,
            target_id=int(target_id),
        )
        dialog.exec_()
