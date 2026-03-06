from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional

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
    SpectraListPlanMeta,
    SpatialTileScheme,
    SpectralBatchDatasetScheme,
)
from wiser.utils.task_stage_utils import (
    ApplyMatrixToDatasetStage,
    IncrementalPcaPartialFitStage,
    ProjectOntoEigenVectorsStage,
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
from wiser.utils.worker_runtime import get_process_storage_client


# region MNF


def _run_shift_y_diff(input_ref: DataRef, input_region: DataRegion, output_write: "WriteSpec") -> None:
    storage_client = get_process_storage_client()
    array, _ = storage_client.read_region(input_ref, input_region)
    noise = array[:-1, :, :] - array[1:, :, :]
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


def _build_approximately_1mb_tile_scheme(meta: DatasetPlanMeta) -> SpatialTileScheme:
    target_bytes = 1024 * 1024
    bytes_per_pixel = max(1, meta.bands * meta.dtype.itemsize)
    target_pixels = max(1, target_bytes // bytes_per_pixel)

    tile_h = max(1, min(meta.height, int(np.sqrt(target_pixels))))
    tile_w = max(1, min(meta.width, max(1, target_pixels // tile_h)))
    return SpatialTileScheme(tile_h=tile_h, tile_w=tile_w)


def _tile_ratio(meta: DatasetPlanMeta, scheme: SpatialTileScheme) -> float:
    tile_bytes = scheme.tile_h * scheme.tile_w * meta.bands * meta.dtype.itemsize
    total_bytes = meta.height * meta.width * meta.bands * meta.dtype.itemsize
    return tile_bytes / total_bytes if total_bytes > 0 else 1.0


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

    if num_components <= 0 or num_components > bands:
        raise ValueError(f"num_components must be in [1, {bands}], got {num_components}")

    noise_ref_name = "mnf_shift_y_noise"
    noise_eigen_ref_name = "mnf_noise_eigen"
    noise_whitening_matrix_ref_name = "mnf_noise_whitening_matrix"
    whitened_dataset_ref_name = "mnf_noise_whitened_dataset"
    whitened_eigen_ref_name = "mnf_whitened_eigen"

    noise_plan_meta = DatasetPlanMeta(
        shape=(max(0, dataset_plan_meta.height - 1), dataset_plan_meta.width, bands),
        dtype=dataset_plan_meta.dtype,
    )
    noise_tile_scheme = _build_approximately_1mb_tile_scheme(noise_plan_meta)
    whitened_tile_scheme = _build_approximately_1mb_tile_scheme(dataset_plan_meta)

    noise_stage = get_y_shift_noise(dataset_ref, noise_ref_name)

    noise_ipca_stage = IncrementalPcaPartialFitStage(
        _num_components=bands,
        _output_ref_name=noise_eigen_ref_name,
        _vectors_ref_name=f"{noise_eigen_ref_name}_vectors",
        _values_ref_name=f"{noise_eigen_ref_name}_values",
        _tile_scheme=noise_tile_scheme,
        _dataset_plan_meta=noise_plan_meta,
        default_executor="process",
        input_binding=DataBinding(noise_ref_name),
        input_plan_meta=noise_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=_tile_ratio(noise_plan_meta, noise_tile_scheme),
            bytes_per_scalar_out=_tile_ratio(noise_plan_meta, noise_tile_scheme),
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

    apply_whitening_stage = ApplyMatrixToDatasetStage(
        _output_ref_name=whitened_dataset_ref_name,
        _matrix_ref=None,
        default_executor="process",
        input_plan_meta=dataset_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=SpatialTileScheme,
        broadcast_input={"matrix_ref": DataBinding(noise_whitening_matrix_ref_name)},
    )

    whitened_ipca_stage = IncrementalPcaPartialFitStage(
        _num_components=bands,
        _output_ref_name=whitened_eigen_ref_name,
        _vectors_ref_name=f"{whitened_eigen_ref_name}_vectors",
        _values_ref_name=f"{whitened_eigen_ref_name}_values",
        _tile_scheme=whitened_tile_scheme,
        _dataset_plan_meta=dataset_plan_meta,
        default_executor="process",
        input_binding=DataBinding(whitened_dataset_ref_name),
        input_plan_meta=dataset_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=_tile_ratio(dataset_plan_meta, whitened_tile_scheme),
            bytes_per_scalar_out=_tile_ratio(dataset_plan_meta, whitened_tile_scheme),
            scratch_bytes_per_scalar_in=0,
        ),
    )

    project_stage = ProjectOntoEigenVectorsStage(
        _num_components=num_components,
        _output_ref_name=output_ref_name,
        _eigen_descriptor_ref=None,
        default_executor="process",
        input_binding=DataBinding(whitened_dataset_ref_name),
        input_plan_meta=dataset_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=SpatialTileScheme,
        broadcast_input={"eigen_descriptor_ref": DataBinding(whitened_eigen_ref_name)},
    )

    return AlgorithmPipeline(
        [
            noise_stage,
            noise_ipca_stage,
            noise_whitening_stage,
            apply_whitening_stage,
            whitened_ipca_stage,
            project_stage,
        ]
    )


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
        storage_client = get_process_storage_client()
        data_meta = storage_client.get_meta(dataset_ref)
        num_components = min(10, data_meta.shape[2])

        mnf_task = SemanticTask(
            priority_class=PriorityClass.BACKGROUND,
            input_ref=dataset_ref,
            algorithm_pipeline=get_mnf_pipeline(dataset_ref, num_components, "mnf_data"),
        )

        task_plan = self._app_services.task_planner.plan_semantic_task(mnf_task)
        future = self._app_services.scheduler.run_task_plan(task_plan)
        return future
