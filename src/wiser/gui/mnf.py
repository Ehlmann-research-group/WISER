from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, cast
from numba.core.target_extension import NonexistentTargetError
import numpy as np
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.app_services import AppServices
from wiser.utils.primitives import (
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    PriorityClass,
    SpatialTileScheme,
    SpectralBatchDatasetScheme,
)
from wiser.utils.task_system import (
    AlgorithmPipeline,
    BasePlanMeta,
    DatasetPlanMeta,
    MapStage,
    ResourceModel,
    SemanticTask,
    WriteSpec,
)
from wiser.utils.worker_runtime import get_process_storage_client


# region MNF


def _run_shift_y_diff(input_ref: DataRef, input_region: DataRegion, output_write: "WriteSpec") -> None:
    storage_client = get_process_storage_client()
    array, meta = storage_client.read_region(input_ref, input_region)
    noise = array[:-1, :, :] - array[1:, :, :]
    assert output_write.region is not None, "output_write's region can not be none in _run_shift_y_diff"
    print(f"output_write.region type: {type(output_write.region)}")
    output_write.region.validate_array_shape(noise)
    storage_client.write_spec(output_write, noise)


class CalculateShiftYDiffNoise(MapStage):
    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        """
        Because this algorithm uses shift difference with just +1
        """
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
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        """
        This stage will just allocate data for the covariance matrix. We
        will be writing to this array.
        """
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "input_meta must be of type DatasetPlanMeta for CalculateShiftYDiffNoise"

        size_est = input_meta.bands * input_meta.bands
        alloc_request = AllocationRequest(
            name="shift_y_diff_noise",
            kind="array",
            residency="ram_cacheable",
            size_est=size_est,
            shape=(input_meta.bands, input_meta.bands),
            dtype=np.dtype(np.float32),
        )
        return [alloc_request]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, DataRef] = {},
    ) -> Callable:
        output_write = output_writes["shift_y_diff_noise"]
        return partial(_run_shift_y_diff, input_ref, input_region, output_write)


class MinimumNoiseFractionDialog:
    """
    Use the shift difference method. Let the user have a dark image option. Let the user
    save their statistics

    Calculate noise -> Aggregate into covariance matrrix

    Get noise, get mean, mean zero noise, incremental covariance build, get whitened matrix
    using covariance, apply to ever spectra, run pca
    """

    def __init__(self, app_services: AppServices):
        self._app_services = app_services

    def perform_mnf(self, dataset_ref: DataRef):
        # Calculate noise

        # Get mean then covariance of noise

        # Get eigen vectors of noise covariance

        # Multiple data with eigen vectors of noise

        # Get mean and subtract out from new data and get covariance

        # Get eigen vectors

        # Project
        storage_client = get_process_storage_client()

        data_meta = storage_client.get_meta(dataset_ref)
        plan_meta = DatasetPlanMeta(shape=data_meta.shape)
        algo_pipeline = AlgorithmPipeline(
            [
                CalculateShiftYDiffNoise(
                    default_executor="process",
                    input_plan_meta=plan_meta,
                    resource_model=ResourceModel(
                        fixed_overhead_bytes=0,
                        bytes_per_scalar_in=1,
                        bytes_per_scalar_out=1,
                        scratch_bytes_per_scalar_in=0,
                    ),
                    chunking_scheme_type=SpectralBatchDatasetScheme,
                    output_bindings=[DataBinding("shift_y_diff_noise")],
                )
            ]
        )

        mnf_task = SemanticTask(  # noqa: F841
            priority_class=PriorityClass.BACKGROUND,
            input_ref=dataset_ref,
            algorithm_pipeline=algo_pipeline,
        )

        task_plan = self._app_services.task_planner.plan_semantic_task(mnf_task)
        future = self._app_services.scheduler.run_task_plan(task_plan)

        return future
