import datetime
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import partial
from typing import Callable, Dict, Optional, TYPE_CHECKING

import numpy as np
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.app_services import AppServices
from wiser.gui.app_state import ApplicationState
from wiser.gui.run_history import RunHistoryManagerBase
from wiser.gui.generated.mnf_dialog_ui import Ui_MNFDialog
from wiser.utils.primitives import (
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    PriorityClass,
    SpectraListPlanMeta,
    SpatialTileScheme,
    SpectralBatchDatasetScheme,
)
from wiser.utils.task_stage_utils import (
    CalcCovMatrixStage,
    EigenDecompositionStage,
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
from wiser.utils.primitives import ExternalRasterHandle
from wiser.utils.worker_runtime import get_process_storage_client

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet


@dataclass(frozen=True)
class MNFRunRecord:
    """Immutable record of one completed MNF run.

    ``eigenvalues`` is the full eigenvalue spectrum produced by the whitened
    eigendecomposition (length equal to the number of good bands), regardless
    of how many components the user asked the pipeline to project onto.
    """

    run_id: int
    timestamp: datetime.datetime
    input_dataset_id: int
    input_dataset_name_snapshot: str
    num_components_chosen: int
    max_components_available: int
    eigenvalues: np.ndarray


class MNFHistoryManager(RunHistoryManagerBase[MNFRunRecord]):
    """Owns the in-memory list of completed MNF runs."""


# region MNF


class ShiftDiffNoiseDirection(IntEnum):
    """Spatial shift direction for first-order difference noise (not along band axis)."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


def shift_diff_noise_output_shape(
    height: int, width: int, bands: int, direction: ShiftDiffNoiseDirection
) -> tuple[int, int, int]:
    """Return ``(y, x, b)`` shape of shift-difference noise for a full ``(height, width, bands)`` input."""
    if direction in (ShiftDiffNoiseDirection.DOWN, ShiftDiffNoiseDirection.UP):
        return max(0, height - 1), width, bands
    return height, max(0, width - 1), bands


def _run_shift_y_diff(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    direction: ShiftDiffNoiseDirection = ShiftDiffNoiseDirection.DOWN,
) -> None:
    storage_client = get_process_storage_client()
    array, array_meta = storage_client.read_region(input_ref, input_region)
    if direction == ShiftDiffNoiseDirection.DOWN:
        noise = array[:-1, :, :] - array[1:, :, :]
    elif direction == ShiftDiffNoiseDirection.UP:
        noise = array[1:, :, :] - array[:-1, :, :]
    elif direction == ShiftDiffNoiseDirection.LEFT:
        noise = array[:, :-1, :] - array[:, 1:, :]
    elif direction == ShiftDiffNoiseDirection.RIGHT:
        noise = array[:, 1:, :] - array[:, :-1, :]
    else:
        raise ValueError(f"Unsupported shift difference direction: {direction!r}")
    if np.ma.isMaskedArray(noise) and array_meta.nodata is not None:
        # If the nodata value is none but the array is still masked, we assume the mask
        # comes from the bad bands. Then we can still mask the bad bands when the next function
        # uses the noise array.
        noise = np.ma.filled(noise, fill_value=array_meta.nodata)
    assert output_write.region is not None, "output_write's region can not be none in _run_shift_y_diff"
    output_write.region.validate_array_shape(noise)
    storage_client.write_spec(output_write, noise)


def _write_shift_y_diff_noise_meta(
    input_ref: DataRef,
    full_input_region: DataRegion,
    output_write: "WriteSpec",
) -> None:
    _ = full_input_region
    storage_client = get_process_storage_client()
    array_meta = storage_client.get_meta(input_ref)
    output_meta = storage_client.get_meta(output_write.ref)
    noise_meta = replace(
        output_meta,
        elem_type=array_meta.elem_type,
        nodata=array_meta.nodata,
        bad_bands=array_meta.bad_bands,
    )
    storage_client.write_meta(output_write.ref, noise_meta)


@dataclass
class CalculateShiftYDiffNoise(MapStage):
    _output_ref_name: str = "shift_y_diff_noise"
    chunking_scheme_type: type[ChunkingScheme] = SpectralBatchDatasetScheme
    shift_diff_direction: ShiftDiffNoiseDirection = ShiftDiffNoiseDirection.DOWN

    def __post_init__(self):
        self.output_bindings = self.output_bindings + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        assert isinstance(
            input_region, DatasetRegionRef
        ), "Input region for calculate shift difference noise must be DatasetRegionRef"

        if self.shift_diff_direction in (ShiftDiffNoiseDirection.DOWN, ShiftDiffNoiseDirection.UP):
            return DatasetRegionRef(
                y0=input_region.y0,
                y1=input_region.y1 - 1,
                x0=input_region.x0,
                x1=input_region.x1,
                b0=input_region.b0,
                b1=input_region.b1,
            )
        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1,
            x0=input_region.x0,
            x1=input_region.x1 - 1,
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

        y, x, b = shift_diff_noise_output_shape(
            input_meta.height, input_meta.width, input_meta.bands, self.shift_diff_direction
        )
        size_est = y * x * b * input_meta.dtype.itemsize
        alloc_request = AllocationRequest(
            name=self._output_ref_name,
            kind="dataset",
            residency="ram_cacheable",
            size_est=size_est,
            shape=(y, x, b),
            dtype=input_meta.dtype,
            delete_policy=self.get_output_delete_policy(self._output_ref_name),
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
        return partial(
            _run_shift_y_diff,
            input_ref,
            input_region,
            output_write,
            self.shift_diff_direction,
        )

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, DataRef] = {},
    ) -> Callable:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(_write_shift_y_diff_noise_meta, input_ref, full_input_region, output_write)


def get_y_shift_noise(
    dataset_ref: DataRef,
    output_ref_name: str,
    shift_diff_direction: ShiftDiffNoiseDirection = ShiftDiffNoiseDirection.DOWN,
) -> CalculateShiftYDiffNoise:
    storage_client = get_process_storage_client()
    data_meta = storage_client.get_meta(dataset_ref)
    plan_meta = DatasetPlanMeta(shape=data_meta.shape, dtype=np.dtype(data_meta.elem_type))
    return CalculateShiftYDiffNoise(
        _output_ref_name=output_ref_name,
        shift_diff_direction=shift_diff_direction,
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
    num_components: Optional[int] = None,
    output_ref_name: str = "mnf_output",
    noise_ref_name: str = "mnf_shift_y_noise",
    noise_eigen_ref_name: str = "mnf_noise_eigen",
    noise_whitening_matrix_ref_name: str = "mnf_noise_whitening_matrix",
    input_mean_ref_name: str = "mnf_input_spectral_mean",
    input_total_ref_name: str = "mnf_input_valid_pixel_total",
    input_covariance_ref_name: str = "mnf_input_covariance",
    whitened_covariance_ref_name: str = "mnf_whitened_covariance",
    whitened_eigen_ref_name: str = "mnf_whitened_eigen",
    data_variance_factor: float = 2,
    shift_diff_noise_direction: ShiftDiffNoiseDirection = ShiftDiffNoiseDirection.DOWN,
) -> AlgorithmPipeline:
    storage_client = get_process_storage_client()
    data_meta = storage_client.get_meta(dataset_ref)
    dataset_plan_meta = DatasetPlanMeta(shape=data_meta.shape, dtype=np.dtype(np.float64))
    if data_meta.bad_bands is not None:
        num_features = np.sum(data_meta.bad_bands)
    else:
        num_features = dataset_plan_meta.bands
    bands = dataset_plan_meta.bands
    if num_components is None:
        num_components = num_features
    if num_components <= 0 or num_components > num_features:
        raise ValueError(f"num_components must be in [1, {num_features}], got {num_components}")

    noise_y, noise_x, noise_b = shift_diff_noise_output_shape(
        dataset_plan_meta.height, dataset_plan_meta.width, bands, shift_diff_noise_direction
    )
    noise_plan_meta = DatasetPlanMeta(
        shape=(noise_y, noise_x, noise_b),
        dtype=dataset_plan_meta.dtype,
    )

    noise_stage = get_y_shift_noise(dataset_ref, noise_ref_name, shift_diff_noise_direction)

    # Manually compute the noise eigen decomposition:
    #   1) SpectralMeanStage   → noise spectral mean
    #   2) CalcCovMatrixStage  → noise covariance (uses the mean above)
    #   3) EigenDecompositionStage → eigen vectors / values + JSON descriptor
    # The JSON descriptor produced by step 3 is what `noise_whitening_stage`
    # consumes (it expects `{"eigen": EigenVectorsAndValues}`), matching the
    # behavior previously provided by AdaptivePcaFitStage.
    noise_mean_ref_name = f"{noise_eigen_ref_name}_mean"
    noise_total_ref_name = f"{noise_eigen_ref_name}_total"
    noise_covariance_ref_name = f"{noise_eigen_ref_name}_covariance"

    noise_mean_stage = SpectralMeanStage(
        _output_ref_name=noise_mean_ref_name,
        _internal_total_ref_name=noise_total_ref_name,
        # The noise dataset inherits bad_bands from the source dataset, so use the
        # source dataset_ref to size the mean spectrum to num_features (good bands).
        _dataset_ref=dataset_ref,
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

    noise_covariance_stage = CalcCovMatrixStage(
        _total_spectra=0,
        _num_features=num_features,
        _output_ref_name=noise_covariance_ref_name,
        _internal_total_ref_name=noise_total_ref_name,
        _data_variance_factor=data_variance_factor,
        default_executor="process",
        input_binding=DataBinding(noise_ref_name),
        input_plan_meta=noise_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        broadcast_input={
            "mean": DataBinding(noise_mean_ref_name),
            "total": DataBinding(noise_total_ref_name),
        },
    )

    noise_eigendecomposition_stage = EigenDecompositionStage(
        _output_ref_name=noise_eigen_ref_name,
        _vectors_ref_name=f"{noise_eigen_ref_name}_vectors",
        _values_ref_name=f"{noise_eigen_ref_name}_values",
        default_executor="process",
        input_binding=DataBinding(noise_covariance_ref_name),
        input_plan_meta=SpectraListPlanMeta(
            num_spectra=num_features,
            spectrum_length=num_features,
            dtype=np.dtype(np.float64),
        ),
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
            num_spectra=num_features,
            spectrum_length=num_features,
            dtype=np.dtype(np.float64),
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
        _internal_total_ref_name=input_total_ref_name,
        _dataset_ref=dataset_ref,
        default_executor="process",
        input_plan_meta=dataset_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
    )

    input_covariance_stage = CalcCovMatrixStage(
        _total_spectra=0,
        _num_features=num_features,
        _output_ref_name=input_covariance_ref_name,
        _internal_total_ref_name=f"{input_covariance_ref_name}_total",
        _data_variance_factor=data_variance_factor,
        default_executor="process",
        input_plan_meta=dataset_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        broadcast_input={
            "mean": DataBinding(input_mean_ref_name),
            "total": DataBinding(input_total_ref_name),
        },
    )

    whitened_covariance_stage = MatrixMultiplicationStage(
        _output_ref_name=whitened_covariance_ref_name,
        _matrix_input_names=("matrix_ref_0", "matrix_ref_1", "matrix_ref_2"),
        _output_shape=(num_features, num_features),
        _output_dtype=np.dtype(np.float64),
        default_executor="process",
        input_binding=DataBinding(input_covariance_ref_name),
        input_plan_meta=SpectraListPlanMeta(
            num_spectra=num_features,
            spectrum_length=num_features,
            dtype=np.dtype(np.float64),
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

    whitened_eigendecomposition_stage = EigenDecompositionStage(
        _output_ref_name=whitened_eigen_ref_name,
        _vectors_ref_name=f"{whitened_eigen_ref_name}_vectors",
        _values_ref_name=f"{whitened_eigen_ref_name}_values",
        default_executor="process",
        input_binding=DataBinding(whitened_covariance_ref_name),
        input_plan_meta=SpectraListPlanMeta(
            num_spectra=num_features,
            spectrum_length=num_features,
            dtype=np.dtype(np.float64),
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
            noise_mean_stage,
            noise_covariance_stage,
            noise_eigendecomposition_stage,
            noise_whitening_stage,
            input_mean_stage,
            input_covariance_stage,
            whitened_covariance_stage,
            whitened_eigendecomposition_stage,
            project_stage,
        ]
    )


class MNFSemanticTask(QObject, SemanticTask):
    # (reduced_data, eigenvalues) — second arg is the full whitened-noise
    # eigenvalue spectrum, snapshotted in completion_callback before the
    # ref it lived in goes out of scope.
    result_ready = Signal(object, object)
    # Emitted from _load_result_into_wiser; payload is an MNFRunRecord.
    run_recorded = Signal(object)

    # Allocation name that EigenDecompositionStage registers for the whitened
    # eigenvalues, derived from `whitened_eigen_ref_name="mnf_whitened_eigen"`
    # in get_mnf_pipeline.  Kept as a class constant so the binding lookup
    # below isn't a magic string.
    _WHITENED_EIGEN_VALUES_REF_NAME = "mnf_whitened_eigen_values"

    def __init__(
        self,
        app_state: ApplicationState,
        source_dataset: "RasterDataSet",
        input_ref: DataRef,
        num_components: int,
        max_components_available: int,
        output_ref_name: str = "mnf_data",
    ):
        QObject.__init__(self)
        SemanticTask.__init__(
            self,
            priority_class=PriorityClass.BACKGROUND,
            input_ref=input_ref,
            algorithm_pipeline=get_mnf_pipeline(
                input_ref, num_components=num_components, output_ref_name=output_ref_name
            ),
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
        self._num_components_chosen = num_components
        self._max_components_available = max_components_available
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

        # Snapshot the full whitened-noise eigenvalue spectrum so the scree
        # plot in the past-runs viewer survives after the underlying ref is
        # released.  EigenDecompositionStage stores values descending, which
        # is what the scree plot expects.
        eigen_values_ref = bindings.get(self._WHITENED_EIGEN_VALUES_REF_NAME)
        if eigen_values_ref is None:
            raise KeyError(f"Missing MNF eigenvalues binding: {self._WHITENED_EIGEN_VALUES_REF_NAME}")
        eigen_values_raw, _ = storage_client.read_data(eigen_values_ref)
        eigenvalues = np.asarray(np.ma.getdata(eigen_values_raw), dtype=np.float64).ravel().copy()

        self.result_ready.emit(np.asarray(reduced_data), eigenvalues)

    @Slot(object, object)
    def _load_result_into_wiser(self, reduced_data: object, eigenvalues: object) -> None:
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
        reduced_dataset.set_data_ignore_value(self._source_dataset.get_data_ignore_value())
        self._app_state.add_dataset(reduced_dataset, view_dataset=False)

        self.run_recorded.emit(
            MNFRunRecord(
                run_id=self.id,
                timestamp=datetime.datetime.now(),
                input_dataset_id=self._source_dataset.get_id(),
                input_dataset_name_snapshot=source_name,
                num_components_chosen=self._num_components_chosen,
                max_components_available=self._max_components_available,
                eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
            )
        )


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

        # Keep the component-count spin box in sync with whichever dataset is
        # picked: max changes per dataset (depends on its dimensions), and we
        # default to the max so the user can see the ceiling at a glance.
        self._ui.comboBox.currentIndexChanged.connect(self._refresh_component_limits)

    @staticmethod
    def _compute_max_components(dataset) -> int:
        """Maximum components the MNF pipeline will accept for this dataset.

        Matches the ceiling enforced inside ``get_mnf_pipeline``: the pipeline
        only accepts ``num_components in [1, num_features]`` where
        ``num_features`` is the count of *good* bands (see mnf.py:242-250).
        Also bounded by the shift-difference noise pixel count so a small
        image can't request more components than there are noise samples.
        """
        bad_bands = dataset.get_bad_bands()
        if bad_bands is not None:
            num_features = int(np.sum(bad_bands))
        else:
            num_features = dataset.num_bands()
        height = dataset.get_height()
        width = dataset.get_width()
        data_pixels = height * width
        noise_pixels = max(0, height - 1) * width
        return min(num_features, max(0, data_pixels - 1), max(0, noise_pixels - 1))

    def _refresh_component_limits(self) -> None:
        """Set spin-box min/max/value based on the currently selected dataset."""
        sbox = self._ui.sbox_component
        sbox.setMinimum(1)
        dataset = self.get_selected_dataset()
        if dataset is None:
            # No dataset chosen yet; leave a generous ceiling and a sane floor
            # so the spin box is interactable but won't accept an invalid value
            # (perform_mnf will recompute the true cap on submit).
            sbox.setMaximum(10_000)
            sbox.setValue(1)
            return
        max_components = max(1, self._compute_max_components(dataset))
        sbox.setMaximum(max_components)
        sbox.setValue(max_components)

    def show_mnf(self, dataset_id: Optional[int] = None) -> None:
        cbox_dataset = self._ui.comboBox
        datasets = self._app_state.get_datasets()
        # Block signals while repopulating so currentIndexChanged doesn't fire
        # for every intermediate addItem — we call _refresh_component_limits
        # once at the end.
        cbox_dataset.blockSignals(True)
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
        cbox_dataset.blockSignals(False)

        self._refresh_component_limits()

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
        # Cap to the pipeline's accepted range (good-band count + noise-pixel
        # ceiling) using the same helper that drives the spin box, so the
        # dialog's display and the submission agree.
        max_components = self._compute_max_components(selected_dataset)
        num_components = min(self.get_num_components(), max_components)
        if num_components <= 0:
            raise ValueError("No valid MNF component count for selected dataset")

        mnf_task = MNFSemanticTask(
            app_state=self._app_state,
            source_dataset=selected_dataset,
            input_ref=dataset_ref,
            num_components=num_components,
            max_components_available=max_components,
        )
        # Record the run in the application-level history once the task
        # finishes loading its result into the app (mirrors linear_unmixing.py:393).
        mnf_task.run_recorded.connect(self._app_state.get_mnf_history().add_record)

        task_plan = self._app_services.task_planner.plan_semantic_task(mnf_task)
        future = self._app_services.task_manager.register_and_submit_task_plan(
            self._app_services.scheduler, task_plan
        )
        return future

    def accept(self):
        self.perform_mnf()
        super().accept()
