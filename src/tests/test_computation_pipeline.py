import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import tests.context

from test_utils.memory_cleanup import release_kept_refs
from wiser.raster.loader import RasterDataLoader
from wiser.raster.spectrum import NumPyArraySpectrum
from wiser.utils.primitives import (
    AllocationRequest,
    DataBinding,
    DataRef,
    DatasetRegionRef,
    DeletePolicy,
    PriorityClass,
)
from wiser.utils.storage_client import StorageClient
from wiser.utils.primitives import ExternalRasterHandle, ExternalSpectrumHandle
from wiser.utils.storage_service import StorageService
from wiser.utils.task_system import (
    AlgorithmPipeline,
    DatasetPlanMeta,
    MapStage,
    PlanningContext,
    ResourceModel,
    SemanticTask,
    SimpleChunkingPolicy,
    TaskPlanner,
    WriteSpec,
)
from wiser.utils.worker_runtime import get_process_storage_client
from wiser.utils.work_scheduler import SchedulerConfig, WorkScheduler


class _NoopSchedulerConfig:
    pass


def _run_map(
    input_ref: DataRef,
    input_region: DatasetRegionRef,
    output_write: WriteSpec,
    spectrum_ref: DataRef,
) -> None:
    """Top-level process-safe worker for one map unit."""
    client = get_process_storage_client()
    input_arr, _input_meta = client.read_region(input_ref, input_region)
    spectrum_arr, _spectrum_meta = client.read_data(spectrum_ref)
    client.write_spec(output_write, input_arr / spectrum_arr)


def _run_multiply_map(
    input_ref: DataRef,
    input_region: DatasetRegionRef,
    output_write: WriteSpec,
    spectrum_ref: DataRef,
) -> None:
    """Top-level process-safe worker for one multiply map unit."""
    client = get_process_storage_client()
    input_arr, _input_meta = client.read_region(input_ref, input_region)
    spectrum_arr, _spectrum_meta = client.read_data(spectrum_ref)
    client.write_spec(output_write, input_arr * spectrum_arr)


class _DivideBySpectrumStage(MapStage):
    def output_region_for(self, input_region: DatasetRegionRef) -> DatasetRegionRef:
        return input_region

    def generate_allocation_requests(self, *, input_meta, chosen_scheme):
        _ = chosen_scheme
        total_bytes = input_meta.height * input_meta.width * input_meta.bands * input_meta.dtype.itemsize
        return [
            AllocationRequest(
                name="stage_out",
                kind="dataset",
                residency="ram_cacheable",
                size_est=total_bytes,
                shape=input_meta.shape,
                dtype=input_meta.dtype,
                chunks=None,
                delete_policy=self.get_output_delete_policy("stage_out"),
            )
        ]

    def task_fn(self, input_ref, input_region, output_writes, broadcast_inputs={}):
        output_write = output_writes["stage_out"]
        spectrum_ref: DataRef = broadcast_inputs["spectrum_ref"]
        return partial(_run_map, input_ref, input_region, output_write, spectrum_ref)


class _MultiplyBySpectrumStage(MapStage):
    def output_region_for(self, input_region: DatasetRegionRef) -> DatasetRegionRef:
        return input_region

    def generate_allocation_requests(self, *, input_meta, chosen_scheme):
        _ = chosen_scheme
        total_bytes = input_meta.height * input_meta.width * input_meta.bands * input_meta.dtype.itemsize
        return [
            AllocationRequest(
                name="stage_out_2",
                kind="dataset",
                residency="ram_cacheable",
                size_est=total_bytes,
                shape=input_meta.shape,
                dtype=input_meta.dtype,
                chunks=None,
                delete_policy=self.get_output_delete_policy("stage_out_2"),
            )
        ]

    def task_fn(self, input_ref, input_region, output_writes, broadcast_inputs={}):
        output_write = output_writes["stage_out_2"]
        spectrum_ref: DataRef = broadcast_inputs["spectrum_ref"]
        return partial(_run_multiply_map, input_ref, input_region, output_write, spectrum_ref)


class TestComputationPipeline(unittest.TestCase):
    def test_semantic_task_computation_pipeline_divides_by_broadcast_spectrum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            address, authkey = service.get_connection_bootstrap()
            client = StorageClient(service=service, service_address=address, service_authkey=authkey)
            scheduler = WorkScheduler(
                SchedulerConfig(_process_budget=3, _thread_budget=3),
                service,
            )
            try:
                fixture_path = (
                    Path(__file__).resolve().parent
                    / ".."
                    / "test_utils"
                    / "test_datasets"
                    / "caltech_425_7_7_nm"
                )
                dataset = RasterDataLoader().load_from_file(str(fixture_path), interactive=False)[0]
                dataset_ref = service.register_external(ExternalRasterHandle(dataset_obj=dataset))

                spectrum_arr = np.asarray(dataset.get_all_bands_at(0, 0))
                spectrum_obj = NumPyArraySpectrum(
                    arr=spectrum_arr,
                    name="pixel_0_0",
                    source_name="caltech_425_7_7_nm",
                )
                spectrum_ref = service.register_external(ExternalSpectrumHandle(spectrum_obj=spectrum_obj))

                input_meta = DatasetPlanMeta(
                    kind="dataset",
                    dtype=np.dtype(np.float32),
                    shape=dataset_ref.shape,
                )
                stage = _DivideBySpectrumStage(
                    default_executor="process",
                    input_plan_meta=input_meta,
                    resource_model=ResourceModel(
                        fixed_overhead_bytes=0,
                        bytes_per_scalar_in=1,
                        bytes_per_scalar_out=1,
                        scratch_bytes_per_scalar_in=0,
                    ),
                    output_bindings=(DataBinding("stage_out"),),
                    broadcast_input={"spectrum_ref": spectrum_ref},
                )
                stage.set_output_delete_policy("stage_out", DeletePolicy.KEEP)
                semantic_task = SemanticTask(
                    priority_class=PriorityClass.INTERACTIVE,
                    input_ref=dataset_ref,
                    algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
                )
                semantic_task.id = 1

                planner = TaskPlanner(
                    PlanningContext(
                        sched_cfg=_NoopSchedulerConfig(),
                        storage=service,
                        chunking_policy=SimpleChunkingPolicy(),
                    )
                )
                plan = planner.plan_semantic_task(semantic_task)
                completion = scheduler.run_task_plan(plan)
                completion.result(timeout=120)

                output_ref = plan.bindings["stage_out"]
                output_arr, _ = client.read_data(output_ref)
                input_arr, _ = client.read_data(dataset_ref)
                spectrum_full, _ = client.read_data(spectrum_ref)
                expected = input_arr / spectrum_full

                np.testing.assert_allclose(output_arr, expected, equal_nan=True)
            finally:
                release_kept_refs(service)
                scheduler.shutdown(wait=True)
                client.close()
                service.close()

    def test_semantic_task_computation_pipeline_divide_then_multiply_returns_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            address, authkey = service.get_connection_bootstrap()
            client = StorageClient(service=service, service_address=address, service_authkey=authkey)
            scheduler = WorkScheduler(
                SchedulerConfig(_process_budget=3, _thread_budget=3),
                service,
            )
            try:
                fixture_path = (
                    Path(__file__).resolve().parent
                    / ".."
                    / "test_utils"
                    / "test_datasets"
                    / "caltech_425_7_7_nm"
                )
                dataset = RasterDataLoader().load_from_file(str(fixture_path), interactive=False)[0]
                dataset_ref = service.register_external(ExternalRasterHandle(dataset_obj=dataset))

                spectrum_arr = np.asarray(dataset.get_all_bands_at(1, 1, filter_bad_values=True))
                spectrum_obj = NumPyArraySpectrum(
                    arr=spectrum_arr,
                    name="pixel_0_0",
                    source_name="caltech_425_7_7_nm",
                )
                spectrum_ref = service.register_external(ExternalSpectrumHandle(spectrum_obj=spectrum_obj))

                input_meta = DatasetPlanMeta(
                    kind="dataset",
                    dtype=np.dtype(np.float32),
                    shape=dataset_ref.shape,
                )
                divide_stage = _DivideBySpectrumStage(
                    default_executor="process",
                    input_plan_meta=input_meta,
                    resource_model=ResourceModel(
                        fixed_overhead_bytes=0,
                        bytes_per_scalar_in=1,
                        bytes_per_scalar_out=1,
                        scratch_bytes_per_scalar_in=0,
                    ),
                    output_bindings=(DataBinding("stage_out"),),
                    broadcast_input={"spectrum_ref": spectrum_ref},
                )
                multiply_stage = _MultiplyBySpectrumStage(
                    default_executor="process",
                    input_plan_meta=input_meta,
                    resource_model=ResourceModel(
                        fixed_overhead_bytes=0,
                        bytes_per_scalar_in=1,
                        bytes_per_scalar_out=1,
                        scratch_bytes_per_scalar_in=0,
                    ),
                    input_binding=DataBinding("stage_out"),
                    output_bindings=(DataBinding("stage_out_2"),),
                    broadcast_input={"spectrum_ref": spectrum_ref},
                )
                multiply_stage.set_output_delete_policy("stage_out_2", DeletePolicy.KEEP)
                semantic_task = SemanticTask(
                    priority_class=PriorityClass.INTERACTIVE,
                    input_ref=dataset_ref,
                    algorithm_pipeline=AlgorithmPipeline(stages=[divide_stage, multiply_stage]),
                )
                semantic_task.id = 2

                planner = TaskPlanner(
                    PlanningContext(
                        sched_cfg=_NoopSchedulerConfig(),
                        storage=service,
                        chunking_policy=SimpleChunkingPolicy(),
                    )
                )
                plan = planner.plan_semantic_task(semantic_task)
                completion = scheduler.run_task_plan(plan)
                completion.result(timeout=120)

                output_ref = plan.bindings["stage_out_2"]
                output_arr, _ = client.read_data(output_ref)
                input_arr, _ = client.read_data(dataset_ref)
                spectrum_arr, _ = client.read_data(spectrum_ref)

                # Divide-then-multiply preserves input only where spectrum is finite;
                # NaNs in spectrum propagate through both stages and should remain NaN.
                valid_spectrum_mask = np.where(np.isnan(spectrum_arr), np.nan, 1.0)
                expected_arr = input_arr * valid_spectrum_mask
                np.testing.assert_allclose(output_arr, expected_arr, rtol=1e-5, atol=1e-8, equal_nan=True)
            finally:
                release_kept_refs(service)
                scheduler.shutdown(wait=True)
                client.close()
                service.close()
