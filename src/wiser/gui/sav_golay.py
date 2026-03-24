from typing import Dict, TYPE_CHECKING

import numpy as np
from PySide2.QtCore import QObject, Signal, Slot

from wiser.utils.primitives import DataMeta, DataRef, DatasetRegionRef, PriorityClass
from wiser.utils.task_stage_utils import get_savgol_filter_pipeline
from wiser.utils.task_system import SemanticTask
from wiser.utils.worker_runtime import get_process_storage_client

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.dataset import RasterDataSet


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
        reduced_dataset.set_default_display_bands(self._source_dataset.default_display_bands())

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
