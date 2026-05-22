from typing import Callable, Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np
from PySide2.QtCore import QObject, Signal, Slot

from wiser.utils.primitives import (
    DataBinding,
    DataRef,
    DatasetPlanMeta,
    DatasetRegionRef,
    PriorityClass,
    SpectraListPlanMeta,
)
from wiser.utils.task_stage_utils import (
    BandSubsetStage,
    CalcCovMatrixStage,
    DecorrelationStretchStage,
    EigenDecompositionStage,
    SpectralMeanStage,
)
from wiser.utils.task_system import AlgorithmPipeline, ResourceModel, SemanticTask
from wiser.utils.worker_runtime import get_process_storage_client

if TYPE_CHECKING:
    from wiser.gui.app_services import AppServices
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.dataset import RasterDataSet

NUM_STRETCH_BANDS = 3

DEFAULT_DECORRELATION_TIMEOUT_SECONDS = 300.0


def _default_resource_model() -> ResourceModel:
    return ResourceModel(
        fixed_overhead_bytes=0,
        bytes_per_scalar_in=1,
        bytes_per_scalar_out=1,
        scratch_bytes_per_scalar_in=0,
    )


def get_decorrelation_stretch_pipeline(
    dataset_ref: DataRef,
    bands: Sequence[int],
    output_ref_name: str = "decorrelation_stretch_output",
    subdataset_ref_name: str = "decorrelation_stretch_subdataset",
    mean_ref_name: str = "decorrelation_stretch_mean",
    correlation_ref_name: str = "decorrelation_stretch_correlation",
    eigen_ref_name: str = "decorrelation_stretch_eigen",
) -> AlgorithmPipeline:
    """
    Build the decorrelation-stretch pipeline for ``bands`` of ``dataset_ref``.

    The chain is: copy the chosen bands into a sub-dataset, compute that
    sub-dataset's correlation matrix (covariance normalized to unit variance),
    eigendecompose it, and apply the ``R^T s R`` decorrelation-stretch transform
    back onto the sub-dataset. The output has the same size as the sub-dataset.
    """
    storage_client = get_process_storage_client()
    dataset_meta = storage_client.get_meta(dataset_ref)
    if len(dataset_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {dataset_meta.shape}")

    bands_tuple = tuple(int(band) for band in bands)
    if len(bands_tuple) != NUM_STRETCH_BANDS:
        raise ValueError(
            f"Decorrelation stretch requires exactly {NUM_STRETCH_BANDS} bands, got {bands_tuple}"
        )

    total_rows, total_cols, total_bands = dataset_meta.shape
    for band in bands_tuple:
        if not (0 <= band < total_bands):
            raise ValueError(f"Band index {band} out of range for total_bands={total_bands}")

    # The sub-dataset is a clean float64 copy of the selected bands over the full
    # spatial extent. Downstream stages plan against this shape directly because
    # the sub-dataset ref does not exist until the pipeline runs.
    subset_shape = (total_rows, total_cols, NUM_STRETCH_BANDS)
    subset_plan_meta = DatasetPlanMeta(shape=subset_shape, dtype=np.dtype(np.float64))
    matrix_plan_meta = SpectraListPlanMeta(
        num_spectra=NUM_STRETCH_BANDS,
        spectrum_length=NUM_STRETCH_BANDS,
        dtype=np.dtype(np.float64),
    )

    band_subset_stage = BandSubsetStage(
        _output_ref_name=subdataset_ref_name,
        _bands=bands_tuple,
        _y0=0,
        _y1=total_rows,
        _x0=0,
        _x1=total_cols,
        default_executor="process",
        input_plan_meta=subset_plan_meta,
        resource_model=_default_resource_model(),
    )

    mean_stage = SpectralMeanStage(
        _output_ref_name=mean_ref_name,
        default_executor="process",
        input_binding=DataBinding(subdataset_ref_name),
        input_plan_meta=subset_plan_meta,
        resource_model=_default_resource_model(),
    )

    correlation_stage = CalcCovMatrixStage(
        _num_features=NUM_STRETCH_BANDS,
        _calc_as_correlation=True,
        _output_ref_name=correlation_ref_name,
        default_executor="process",
        input_binding=DataBinding(subdataset_ref_name),
        input_plan_meta=subset_plan_meta,
        resource_model=_default_resource_model(),
        broadcast_input={"mean": DataBinding(mean_ref_name)},
    )

    eigendecomposition_stage = EigenDecompositionStage(
        _output_ref_name=eigen_ref_name,
        _vectors_ref_name=f"{eigen_ref_name}_vectors",
        _values_ref_name=f"{eigen_ref_name}_values",
        default_executor="process",
        input_binding=DataBinding(correlation_ref_name),
        input_plan_meta=matrix_plan_meta,
        resource_model=_default_resource_model(),
    )

    decorrelation_stage = DecorrelationStretchStage(
        _output_ref_name=output_ref_name,
        default_executor="process",
        input_binding=DataBinding(subdataset_ref_name),
        input_plan_meta=subset_plan_meta,
        resource_model=_default_resource_model(),
        broadcast_input={
            "eigen_descriptor_ref": DataBinding(eigen_ref_name),
            "spectral_mean_ref": DataBinding(mean_ref_name),
        },
    )

    return AlgorithmPipeline(
        [
            band_subset_stage,
            mean_stage,
            correlation_stage,
            eigendecomposition_stage,
            decorrelation_stage,
        ]
    )


class DecorrelationStretchSemanticTask(QObject, SemanticTask):
    result_ready = Signal(object)

    def __init__(
        self,
        app_state: "ApplicationState",
        source_dataset: "RasterDataSet",
        input_ref: DataRef,
        bands: Sequence[int],
        decorr_callback: Callable[[np.ndarray], None],
        output_ref_name: str = "decorrelation_stretch_output",
    ):
        QObject.__init__(self)
        bands_tuple = tuple(int(band) for band in bands)
        SemanticTask.__init__(
            self,
            priority_class=PriorityClass.BACKGROUND,
            input_ref=input_ref,
            algorithm_pipeline=get_decorrelation_stretch_pipeline(
                dataset_ref=input_ref,
                bands=bands_tuple,
                output_ref_name=output_ref_name,
            ),
            task_title="Decorrelation Stretch",
            task_variables={
                "Dataset": source_dataset.get_name(),
                "Bands": ", ".join(str(band) for band in bands_tuple),
            },
        )
        self._init_parameters(
            app_state=app_state,
            source_dataset=source_dataset,
            bands=bands_tuple,
            decorr_callback=decorr_callback,
            output_ref_name=output_ref_name,
        )
        self.result_ready.connect(self._load_result_into_wiser)

    def _init_parameters(
        self,
        app_state: "ApplicationState",
        source_dataset: "RasterDataSet",
        bands: Sequence[int],
        decorr_callback: Callable[[np.ndarray], None],
        output_ref_name: str,
    ) -> None:
        self.id = app_state.take_next_id()
        self._app_state = app_state
        self._source_dataset = source_dataset
        self._bands = tuple(int(band) for band in bands)
        self._decorr_callback = decorr_callback
        self._output_ref_name = output_ref_name

        # Holds the stretched (height, width, 3) result once the pipeline
        # completes. This is populated inside completion_callback (which the
        # scheduler runs *before* resolving the plan's completion future), so a
        # caller that blocks on that future can read this attribute directly
        self._result: Optional[np.ndarray] = None

    def completion_callback(self, bindings: Dict[str, DataRef]) -> None:
        output_ref = bindings.get(self._output_ref_name)
        if output_ref is None:
            raise KeyError(f"Missing decorrelation stretch output binding: {self._output_ref_name}")

        storage_client = get_process_storage_client()
        output_meta = storage_client.get_meta(output_ref)
        height, width, bands = output_meta.shape
        output_region = DatasetRegionRef(y0=0, y1=height, x0=0, x1=width, b0=0, b1=bands)
        # Read with masking enabled so nodata and bad-band pixels come back masked.
        stretched_data, _ = storage_client.read_region(output_ref, output_region, filter_data=True)
        result = np.ma.asarray(stretched_data)

        # Store the result by value before emitting. A blocking caller reads
        # self._result (a masked array) after the completion future resolves; the
        # signal remains for non-blocking callers.
        self._result = result
        self.result_ready.emit(result)

    def get_result(self) -> Optional[np.ndarray]:
        """Return the stretched (height, width, 3) array, or None if the task
        has not completed yet."""
        return self._result

    @Slot(object)
    def _load_result_into_wiser(self, stretched_data: object) -> None:
        self._decorr_callback(np.asarray(stretched_data))


def compute_decorrelation_stretch(
    app_state: "ApplicationState",
    app_services: "AppServices",
    source_dataset: "RasterDataSet",
    input_ref: DataRef,
    bands: Sequence[int],
    timeout: Optional[float] = DEFAULT_DECORRELATION_TIMEOUT_SECONDS,
    output_ref_name: str = "decorrelation_stretch_output",
) -> np.ndarray:
    """
    Run the decorrelation-stretch pipeline synchronously and return the
    stretched ``(height, width, 3)`` array.

    The caller is responsible for registering ``source_dataset`` and passing the
    resulting ``input_ref``.

    This blocks the calling thread until the pipeline completes. The work
    scheduler runs on its own threads / process pool, so blocking here does not
    stall the computation -- only the calling (GUI) thread waits. Because that
    thread is blocked, the task's ``result_ready`` Qt signal is not delivered;
    we read the result by value via
    :meth:`DecorrelationStretchSemanticTask.get_result` instead.

    Raises:
        concurrent.futures.TimeoutError: if the pipeline does not finish within
            ``timeout`` seconds.
        RuntimeError: if the pipeline completes but produced no result.
    """
    task = DecorrelationStretchSemanticTask(
        app_state=app_state,
        source_dataset=source_dataset,
        input_ref=input_ref,
        bands=bands,
        decorr_callback=lambda _data: None,
        output_ref_name=output_ref_name,
    )

    task_plan = app_services.task_planner.plan_semantic_task(task)
    future = app_services.task_manager.register_and_submit_task_plan(app_services.scheduler, task_plan)

    # Blocks until the plan's completion future resolves. future.result()
    # re-raises any exception captured during pipeline execution.
    future.result(timeout=timeout)

    result = task.get_result()
    if result is None:
        raise RuntimeError("Decorrelation stretch completed without producing a result")
    return result
