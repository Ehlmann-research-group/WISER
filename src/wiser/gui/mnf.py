import datetime
from dataclasses import dataclass
from dataclasses import replace
from functools import partial
from typing import Callable, Dict, Optional, TYPE_CHECKING

import numpy as np
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.app_services import AppServices
from wiser.gui.app_state import ApplicationState
from wiser.gui.generated.mnf_dialog_ui import Ui_MNFDialog
from wiser.utils.primitives import (
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataMeta,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    PriorityClass,
    SpectraListPlanMeta,
    SpatialTileScheme,
    SpectralBatchDatasetScheme,
)
from wiser.utils.task_stage_utils import (
    AdaptivePcaFitStage,
    CalcCovMatrixStage,
    EigendecompositionStage,
    MatrixMultiplicationStage,
    ProjectOntoEigenVectorsStage,
    SpectralMeanStage,
    WhiteningMatrixStage,
)
from wiser.utils.task_system import (
    AlgorithmPipeline,
    DatasetPlanMeta,
    MapStage,
    ResourceModel,
    SemanticTask,
    WriteSpec,
)
from wiser.utils.storage_layer import ExternalRasterHandle
from wiser.utils.worker_runtime import get_process_storage_client

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet

# region MNF


def _run_shift_y_diff(input_ref: DataRef, input_region: DataRegion, output_write: "WriteSpec") -> None:
    storage_client = get_process_storage_client()
    array, array_meta = storage_client.read_region(input_ref, input_region)
    noise = array[:-1, :, :] - array[1:, :, :]
    if np.ma.isMaskedArray(noise) and array_meta.nodata is not None:
        # If the nodata value is none but the array is still masked, we assume the mask
        # comes from the bad bands. Then we can still mask the bad bands when the next function
        # uses the noise array.
        noise = np.ma.filled(noise, fill_value=array_meta.nodata)
    assert output_write.region is not None, "output_write's region can not be none in _run_shift_y_diff"
    output_write.region.validate_array_shape(noise)
    storage_client.write_spec(output_write, noise)


@dataclass
class CalculateShiftYDiffNoise(MapStage):
    _output_ref_name: str = "shift_y_diff_noise"

    chunking_scheme_type: type[ChunkingScheme] = SpectralBatchDatasetScheme

    def __post_init__(self):
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        assert isinstance(
            input_region, DatasetRegionRef
        ), "Input region for calculate shift difference noise must be DatasetRegionRef"

        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1 - 1,
            x0=input_region.x0,
            x1=input_region.x1,
            b0=input_region.b0,
            b1=input_region.b1,
        )

    def generate_allocation_requests(
        self,
        *,
        input_meta: DatasetPlanMeta,
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "input_meta must be of type DatasetPlanMeta for CalculateShiftYDiffNoise"

        y = max(0, input_meta.height - 1)
        x = input_meta.width
        b = input_meta.bands
        size_est = y * x * b * input_meta.dtype.itemsize
        alloc_request = AllocationRequest(
            name=self._output_ref_name,
            kind="dataset",
            residency="ram_cacheable",
            size_est=size_est,
            shape=(y, x, b),
            dtype=input_meta.dtype,
        )
        return [alloc_request]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, DataRef] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(_run_shift_y_diff, input_ref, input_region, output_write)


def get_y_shift_noise(dataset_ref: DataRef, output_ref_name: str) -> CalculateShiftYDiffNoise:
    storage_client = get_process_storage_client()
    data_meta = storage_client.get_meta(dataset_ref)
    plan_meta = DatasetPlanMeta(shape=data_meta.shape, dtype=np.dtype(data_meta.elem_type))
    return CalculateShiftYDiffNoise(
        _output_ref_name=output_ref_name,
        default_executor="process",
        input_plan_meta=plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
    )


def get_mnf_pipeline(
    dataset_ref: DataRef,
    num_components: int,
    output_ref_name: str,
) -> AlgorithmPipeline:
    storage_client = get_process_storage_client()
    data_meta = storage_client.get_meta(dataset_ref)
    dataset_plan_meta = DatasetPlanMeta(shape=data_meta.shape, dtype=np.dtype(data_meta.elem_type))
    bands = dataset_plan_meta.bands
    data_pixels = dataset_plan_meta.height * dataset_plan_meta.width
    noise_pixels = max(0, dataset_plan_meta.height - 1) * dataset_plan_meta.width
    max_internal_components = min(bands, max(0, noise_pixels - 1))
    if max_internal_components <= 0:
        raise ValueError(
            f"MNF requires at least 2 samples in both data/noise domains; got "
            f"data_pixels={data_pixels}, noise_pixels={noise_pixels}"
        )
    if num_components <= 0 or num_components > max_internal_components:
        raise ValueError(f"num_components must be in [1, {max_internal_components}], got {num_components}")

    noise_ref_name = "mnf_shift_y_noise"
    noise_eigen_ref_name = "mnf_noise_eigen"
    noise_whitening_matrix_ref_name = "mnf_noise_whitening_matrix"
    input_mean_ref_name = "mnf_input_spectral_mean"
    input_covariance_ref_name = "mnf_input_covariance"
    whitened_covariance_ref_name = "mnf_whitened_covariance"
    whitened_eigen_ref_name = "mnf_whitened_eigen"

    noise_plan_meta = DatasetPlanMeta(
        shape=(max(0, dataset_plan_meta.height - 1), dataset_plan_meta.width, bands),
        dtype=dataset_plan_meta.dtype,
    )

    noise_stage = get_y_shift_noise(dataset_ref, noise_ref_name)

    noise_ipca_stage = AdaptivePcaFitStage(
        _num_components=max_internal_components,
        _data_variance_factor=2,
        _output_ref_name=noise_eigen_ref_name,
        _vectors_ref_name=f"{noise_eigen_ref_name}_vectors",
        _values_ref_name=f"{noise_eigen_ref_name}_values",
        _covariance_ref_name=f"{noise_eigen_ref_name}_covariance",
        _mean_ref_name=f"{noise_eigen_ref_name}_mean",
        _dataset_plan_meta=noise_plan_meta,
        default_executor="process",
        input_binding=DataBinding(noise_ref_name),
        input_plan_meta=noise_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
    )

    noise_whitening_stage = WhiteningMatrixStage(
        _output_ref_name=noise_whitening_matrix_ref_name,
        default_executor="process",
        input_binding=DataBinding(noise_eigen_ref_name),
        input_plan_meta=SpectraListPlanMeta(
            num_spectra=bands,
            spectrum_length=bands,
            dtype=np.dtype(np.float32),
        ),
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
    )

    input_mean_stage = SpectralMeanStage(
        _output_ref_name=input_mean_ref_name,
        default_executor="process",
        input_plan_meta=dataset_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        broadcast_input={"total": data_pixels},
    )

    input_covariance_stage = CalcCovMatrixStage(
        _total_spectra=data_pixels,
        _output_ref_name=input_covariance_ref_name,
        default_executor="process",
        input_plan_meta=dataset_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        broadcast_input={"mean": DataBinding(input_mean_ref_name)},
    )

    whitened_covariance_stage = MatrixMultiplicationStage(
        _output_ref_name=whitened_covariance_ref_name,
        _matrix_input_names=("matrix_ref_0", "matrix_ref_1", "matrix_ref_2"),
        _output_shape=(bands, bands),
        _output_dtype=np.dtype(np.float32),
        default_executor="process",
        input_binding=DataBinding(input_covariance_ref_name),
        input_plan_meta=SpectraListPlanMeta(
            num_spectra=bands,
            spectrum_length=bands,
            dtype=np.dtype(np.float32),
        ),
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        broadcast_input={
            "matrix_ref_0": DataBinding(noise_whitening_matrix_ref_name),
            "matrix_ref_1": DataBinding(input_covariance_ref_name),
            "matrix_ref_2": DataBinding(noise_whitening_matrix_ref_name),
        },
    )

    whitened_eigendecomposition_stage = EigendecompositionStage(
        _output_ref_name=whitened_eigen_ref_name,
        _vectors_ref_name=f"{whitened_eigen_ref_name}_vectors",
        _values_ref_name=f"{whitened_eigen_ref_name}_values",
        default_executor="process",
        input_binding=DataBinding(whitened_covariance_ref_name),
        input_plan_meta=SpectraListPlanMeta(
            num_spectra=bands,
            spectrum_length=bands,
            dtype=np.dtype(np.float32),
        ),
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
    )

    project_stage = ProjectOntoEigenVectorsStage(
        _num_components=num_components,
        _output_ref_name=output_ref_name,
        _eigen_descriptor_ref=None,
        default_executor="process",
        input_plan_meta=dataset_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=SpatialTileScheme,
        broadcast_input={
            "eigen_descriptor_ref": DataBinding(whitened_eigen_ref_name),
            "spectral_mean_ref": DataBinding(input_mean_ref_name),
            "eigenvector_matrix_ref_0": DataBinding(noise_whitening_matrix_ref_name),
        },
        _eigenvector_multiply_matrix_names=("eigenvector_matrix_ref_0",),
    )

    return AlgorithmPipeline(
        [
            noise_stage,
            noise_ipca_stage,
            noise_whitening_stage,
            input_mean_stage,
            input_covariance_stage,
            whitened_covariance_stage,
            whitened_eigendecomposition_stage,
            project_stage,
        ]
    )


class MNFSemanticTask(QObject, SemanticTask):
    result_ready = Signal(object)

    def __init__(
        self,
        app_state: ApplicationState,
        source_dataset: "RasterDataSet",
        input_ref: DataRef,
        num_components: int,
        output_ref_name: str = "mnf_data",
    ):
        QObject.__init__(self)
        SemanticTask.__init__(
            self,
            priority_class=PriorityClass.BACKGROUND,
            input_ref=input_ref,
            algorithm_pipeline=get_mnf_pipeline(input_ref, num_components, output_ref_name),
            task_title="Minimum Noise Fraction",
            task_variables={
                "Num Components": num_components,
                "Dataset": source_dataset.get_name(),
            },
        )
        self.id = app_state.take_next_id()
        self._app_state = app_state
        self._source_dataset = source_dataset
        self._output_ref_name = output_ref_name
        self.result_ready.connect(self._load_result_into_wiser)

    def completion_callback(self, bindings: Dict[str, DataRef]) -> None:
        output_ref = bindings.get(self._output_ref_name)
        if output_ref is None:
            raise KeyError(f"Missing MNF output binding: {self._output_ref_name}")

        storage_client = get_process_storage_client()
        data_meta = storage_client.get_meta(output_ref)
        height, width, bands = data_meta.shape
        output_region = DatasetRegionRef(y0=0, y1=height, x0=0, x1=width, b0=0, b1=bands)
        reduced_data, _ = storage_client.read_region(output_ref, output_region)
        self.result_ready.emit(np.asarray(reduced_data))

    @Slot(object)
    def _load_result_into_wiser(self, reduced_data: object) -> None:
        reduced_array = np.asarray(reduced_data)
        reduced_array_by_band = reduced_array.transpose(2, 0, 1)

        loader = self._app_state.get_loader()
        cache = self._app_state.get_cache()
        reduced_dataset = loader.dataset_from_numpy_array(reduced_array_by_band, cache)

        source_name = self._source_dataset.get_name() or "Dataset"
        timestamp = datetime.datetime.now().isoformat()
        reduced_dataset.set_name(self._app_state.unique_dataset_name(f"MNF, Img: {source_name}"))
        reduced_dataset.set_description(f"MNF reduced image-cube: {source_name} ({timestamp})")
        reduced_dataset.copy_spatial_metadata(self._source_dataset.get_spatial_metadata())
        self._app_state.add_dataset(reduced_dataset, view_dataset=False)


class MinimumNoiseFractionDialog(QDialog):
    """
    Use the shift difference method. Let the user have a dark image option. Let the user
    save their statistics

    Calculate noise -> Aggregate into covariance matrrix

    Get noise, get mean, mean zero noise, incremental covariance build, get whitened matrix
    using covariance, apply to ever spectra, run pca
    """

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

        self._ui = Ui_MNFDialog()
        self._ui.setupUi(self)

    def show_mnf(self, dataset_id: Optional[int] = None) -> None:
        cbox_dataset = self._ui.comboBox
        datasets = self._app_state.get_datasets()
        cbox_dataset.clear()
        cbox_dataset.addItem(self.tr("(no data)"), -1)
        for dataset in datasets:
            cbox_dataset.addItem(dataset.get_name(), dataset.get_id())

        if dataset_id is not None:
            index = cbox_dataset.findData(dataset_id)
            if index >= 0:
                cbox_dataset.setCurrentIndex(index)
            else:
                cbox_dataset.setCurrentIndex(0)
        else:
            cbox_dataset.setCurrentIndex(0)

        self._ui.sbox_component.setMinimum(1)
        self._ui.sbox_component.setMaximum(10_000)
        if self._ui.sbox_component.value() < 1:
            self._ui.sbox_component.setValue(1)

    def showEvent(self, event):
        self.show_mnf(dataset_id=self._selected_dataset_id)
        super().showEvent(event)

    def select_dataset(self, dataset_id: Optional[int]) -> None:
        self._selected_dataset_id = dataset_id
        self.show_mnf(dataset_id=dataset_id)

    def get_selected_dataset_id(self) -> Optional[int]:
        dataset_id = self._ui.comboBox.currentData()
        if dataset_id is None or int(dataset_id) < 0:
            return None
        return int(dataset_id)

    def get_selected_dataset(self):
        dataset_id = self.get_selected_dataset_id()
        if dataset_id is None:
            return None
        return self._app_state.get_dataset(dataset_id)

    def get_num_components(self) -> int:
        return int(self._ui.sbox_component.value())

    def perform_mnf(self):
        selected_dataset = self.get_selected_dataset()
        if selected_dataset is None:
            raise ValueError("No dataset selected for MNF")

        self._selected_dataset_id = selected_dataset.get_id()

        dataset_ref = self._app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=selected_dataset)
        )
        storage_client = get_process_storage_client()
        data_meta = storage_client.get_meta(dataset_ref)
        height, width, bands = data_meta.shape
        data_pixels = height * width
        noise_pixels = max(0, height - 1) * width
        max_components = min(bands, max(0, data_pixels - 1), max(0, noise_pixels - 1))
        num_components = min(self.get_num_components(), max_components)
        if num_components <= 0:
            raise ValueError("No valid MNF component count for selected dataset")

        mnf_task = MNFSemanticTask(
            app_state=self._app_state,
            source_dataset=selected_dataset,
            input_ref=dataset_ref,
            num_components=num_components,
        )

        task_plan = self._app_services.task_planner.plan_semantic_task(mnf_task)
        future = self._app_services.task_manager.register_and_submit_task_plan(
            self._app_services.scheduler, task_plan
        )
        return future

    def accept(self):
        self.perform_mnf()
        super().accept()
