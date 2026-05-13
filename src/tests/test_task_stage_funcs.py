import unittest
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.decomposition import PCA

import tests.context
# import context

from test_utils.memory_cleanup import release_kept_refs
from wiser.gui.app_services import AppServices
from wiser.utils.task_stage_utils import (
    CalcCovMatrixStage,
    EigenDecompositionStage,
    EigenVectorsAndValues,
    ProjectOntoEigenVectorsStage,
    SavGolayFilterStage,
    SpectralMeanStage,
    count_valid_dataset_pixels,
    get_apply_matrices_to_dataset_stage,
    get_apply_matrix_to_dataset_stage,
    get_adaptive_pca_partial_fit_stage,
    get_good_band_runs,
    get_matrix_multiplication_stage,
    get_noise_covariance_pipeline,
    get_pos_semi_def_matrix_inverse_stage,
    get_project_onto_eigenvectors_stage,
    get_savgol_filter_pipeline,
    recombine_dataset_tile_from_good_band_runs,
    get_spectral_mean_stage,
    split_dataset_tile_by_good_band_runs,
    validate_no_unmasked_nonfinite_values,
    get_whitening_matrix_stage,
    get_eigendecomposition_pipeline,
)
from wiser.raster.loader import RasterDataLoader
from wiser.utils.primitives import (
    AllocationRequest,
    DataBinding,
    DataMeta,
    DeletionState,
    DeletePolicy,
    NoChunkingScheme,
    PriorityClass,
    SpectraListPlanMeta,
)
from wiser.utils.storage_client import StorageClient
from wiser.utils.storage_service import shared_mem_exists
from wiser.utils.primitives import ExternalRasterHandle
from wiser.utils.worker_runtime import get_process_storage_client
from wiser.utils.task_system import (
    AlgorithmPipeline,
    DatasetPlanMeta,
    ResourceModel,
    SemanticTask,
)
from test_utils.test_model import WiserTestModel

pytestmark = [
    pytest.mark.integration,
]


class TestTaskStageFuncs(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _keep_outputs(self, stage, *output_names: str) -> None:
        for output_name in output_names:
            stage.set_output_delete_policy(output_name, DeletePolicy.KEEP)

    def _keep_adaptive_pca_outputs(self, stage, *, keep_resolved: bool = False) -> None:
        self._keep_outputs(
            stage,
            stage._output_ref_name,
            stage._vectors_ref_name,
            stage._values_ref_name,
            stage._mean_ref_name,
            stage._covariance_ref_name,
            stage._good_band_mask_ref_name,
        )
        if keep_resolved:
            self._keep_outputs(stage, stage._resolved_num_components_ref_name)

    def test_get_good_band_runs_handles_edge_cases(self) -> None:
        self.assertEqual(
            get_good_band_runs(np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1])),
            [(0, 5), (8, 11), (12, 14)],
        )
        self.assertEqual(get_good_band_runs(np.array([0, 0, 0])), [])
        self.assertEqual(get_good_band_runs(np.array([1, 0, 1, 0, 1, 0])), [(0, 1), (2, 3), (4, 5)])
        self.assertEqual(get_good_band_runs(np.array([0, 0, 1, 1, 0, 1])), [(2, 4), (5, 6)])
        self.assertEqual(get_good_band_runs(np.array([1, 1, 0, 1, 0, 0])), [(0, 2), (3, 4)])
        self.assertEqual(get_good_band_runs(np.array([0, 1, 1, 0])), [(1, 3)])

    def test_split_and_recombine_dataset_tile_by_good_band_runs_round_trips(self) -> None:
        tile = np.arange(2 * 3 * 6, dtype=np.float32).reshape(2, 3, 6)
        runs = [(0, 2), (3, 5), (5, 6)]

        chunks = split_dataset_tile_by_good_band_runs(tile, runs)

        self.assertEqual([chunk.shape for chunk in chunks], [(2, 3, 2), (2, 3, 2), (2, 3, 1)])

        base = np.full(tile.shape, -1.0, dtype=np.float32)
        recombined = recombine_dataset_tile_from_good_band_runs(tile.shape, runs, chunks, base_array=base)

        self.assertTrue(np.allclose(recombined[:, :, 0:2], tile[:, :, 0:2]))
        self.assertTrue(np.allclose(recombined[:, :, 3:5], tile[:, :, 3:5]))
        self.assertTrue(np.allclose(recombined[:, :, 5:6], tile[:, :, 5:6]))
        self.assertTrue(np.all(recombined[:, :, 2] == -1.0))

    def test_validate_no_unmasked_nonfinite_values_allows_masked_nonfinite(self) -> None:
        masked = np.ma.array(
            [[[1.0, np.nan], [2.0, np.inf]]],
            mask=[[[False, True], [False, True]]],
            dtype=np.float32,
        )
        validate_no_unmasked_nonfinite_values(masked)

    def test_validate_no_unmasked_nonfinite_values_rejects_unmasked_nonfinite(self) -> None:
        with self.assertRaisesRegex(ValueError, "unmasked NaN or Inf"):
            validate_no_unmasked_nonfinite_values(np.array([[[1.0, np.nan], [2.0, 3.0]]], dtype=np.float32))

    def test_get_savgol_filter_pipeline_rejects_window_longer_than_shortest_good_run(self) -> None:
        app_services = AppServices()
        try:
            dataset = np.zeros((2, 2, 6), dtype=np.float32)
            dataset_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="savgol_validation_dataset",
                    kind="dataset",
                    residency="ram_cacheable",
                    size_est=dataset.size * dataset.dtype.itemsize,
                    shape=dataset.shape,
                    dtype=dataset.dtype,
                )
            )
            process_storage_client = get_process_storage_client()
            process_storage_client.write_data(dataset_ref, dataset)
            app_services.storage_service.update_meta(
                dataset_ref,
                bad_bands=np.asarray([1, 1, 0, 1, 1, 1], dtype=np.int32),
            )

            with self.assertRaisesRegex(ValueError, "shortest_good_run=2"):
                get_savgol_filter_pipeline(
                    dataset_ref=dataset_ref,
                    window_length=3,
                    polyorder=1,
                    output_ref_name="savgol_invalid",
                )
        finally:
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_spectral_mean_stage_pipeline_execution(self) -> None:
        # RasterDataLoader expects [band][y][x]. Each pixel has a constant spectrum value.
        array_2x2x4 = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            dtype=np.float32,
        )
        dataset = RasterDataLoader().dataset_from_numpy_array(array_2x2x4)

        app_services = AppServices()
        storage_client = None
        try:
            input_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )

            output_ref_name = "spectral_mean"
            stage = get_spectral_mean_stage(input_ref, output_ref_name)
            self._keep_outputs(stage, output_ref_name)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=input_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 1001

            task_plan = app_services.task_planner.plan_semantic_task(task)

            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=20)

            output_ref = task_plan.bindings[output_ref_name]

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_spectrum, _ = storage_client.read_data(output_ref)
            self.assertEqual(output_spectrum.shape, (4,))
            self.assertTrue(np.allclose(output_spectrum, 2.5))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_spectral_mean_stage_pre_task_computes_internal_total(self) -> None:
        array_2x2x4 = np.array(
            [
                [[1.0, 2.0], [-9999.0, 4.0]],
                [[100.0, 100.0], [100.0, 100.0]],
                [[10.0, 20.0], [30.0, np.nan]],
                [[1000.0, 2000.0], [3000.0, 4000.0]],
            ],
            dtype=np.float32,
        )
        dataset = RasterDataLoader().dataset_from_numpy_array(array_2x2x4)
        dataset.set_bad_bands([1, 0, 1, 1])
        dataset.set_data_ignore_value(-9999.0)

        app_services = AppServices()
        storage_client = None
        try:
            input_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )

            stage = get_spectral_mean_stage(input_ref, "spectral_mean")
            self._keep_outputs(stage, "_internal_total")
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=input_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 10011

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=20)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            internal_total = storage_client.read_json_value(task_plan.bindings["_internal_total"])
            self.assertEqual(internal_total, {"total": 2})
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_spectral_mean_stage_filters_nodata_and_bad_bands(self) -> None:
        array_2x2x4 = np.array(
            [
                [[1.0, 2.0], [-9999.0, 4.0]],
                [[100.0, 100.0], [100.0, 100.0]],
                [[10.0, 20.0], [30.0, np.nan]],
                [[1000.0, 2000.0], [3000.0, 4000.0]],
            ],
            dtype=np.float32,
        )
        dataset = RasterDataLoader().dataset_from_numpy_array(array_2x2x4)
        dataset.set_bad_bands([1, 0, 1, 1])
        dataset.set_data_ignore_value(-9999.0)

        app_services = AppServices()
        storage_client = None
        try:
            input_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )

            output_ref_name = "spectral_mean_filtered"
            stage = get_spectral_mean_stage(input_ref, output_ref_name)
            self._keep_outputs(stage, output_ref_name)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=input_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 10012

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=15)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_ref = task_plan.bindings[output_ref_name]
            output_spectrum, output_meta = storage_client.read_data(output_ref)

            expected = np.array([1.5, 15.0, 1500.0], dtype=np.float32)
            self.assertEqual(output_spectrum.shape, (3,))
            self.assertTrue(np.allclose(output_spectrum, expected, atol=1e-6))
            self.assertIsNone(output_meta.bad_bands)
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_spectral_mean_stage_copies_provided_total_into_internal_total_ref(self) -> None:
        array_2x2x2 = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
            ],
            dtype=np.float32,
        )
        dataset = RasterDataLoader().dataset_from_numpy_array(array_2x2x2)

        app_services = AppServices()
        storage_client = None
        try:
            input_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )

            stage = SpectralMeanStage(
                _output_ref_name="spectral_mean_with_total",
                _internal_total_ref_name="spectral_mean_with_total_ref",
                _meta_ref=input_ref,
                default_executor="process",
                input_plan_meta=DatasetPlanMeta(shape=(2, 2, 2), dtype=np.dtype(np.float32)),
                resource_model=ResourceModel(
                    fixed_overhead_bytes=0,
                    bytes_per_scalar_in=1,
                    bytes_per_scalar_out=1,
                    scratch_bytes_per_scalar_in=0,
                ),
                broadcast_input={"total": 4},
            )
            self._keep_outputs(stage, "spectral_mean_with_total_ref")
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=input_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 10013

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=20)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            internal_total = storage_client.read_json_value(
                task_plan.bindings["spectral_mean_with_total_ref"]
            )
            self.assertEqual(internal_total, {"total": 4})
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_count_valid_dataset_pixels_filters_nodata_bad_bands_and_nonfinite(self) -> None:
        array_2x2x4 = np.array(
            [
                [[1.0, 2.0], [-9999.0, 4.0]],
                [[100.0, 100.0], [100.0, 100.0]],
                [[10.0, 20.0], [30.0, np.nan]],
                [[1000.0, 2000.0], [3000.0, 4000.0]],
            ],
            dtype=np.float32,
        )
        dataset = RasterDataLoader().dataset_from_numpy_array(array_2x2x4)
        dataset.set_bad_bands([1, 0, 1, 1])
        dataset.set_data_ignore_value(-9999.0)

        app_services = AppServices()
        try:
            input_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            self.assertEqual(count_valid_dataset_pixels(input_ref), 2)
        finally:
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_noise_covariance_pipeline_execution(self) -> None:
        # RasterDataLoader expects [band][y][x]. In [y][x][b], top row pixels are [1,2,1,2]
        # and bottom row pixels are [4,1,4,1].
        array_2x2x4 = np.array(
            [
                [[1.0, 1.0], [4.0, 4.0]],
                [[2.0, 2.0], [1.0, 1.0]],
                [[1.0, 1.0], [4.0, 4.0]],
                [[2.0, 2.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        dataset = RasterDataLoader().dataset_from_numpy_array(array_2x2x4)

        app_services = AppServices()
        storage_client = None
        try:
            input_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )

            output_ref_name = "noise_covariance"
            noise_cov_pipeline = get_noise_covariance_pipeline(input_ref, output_ref_name)
            noise_cov_pipeline.stages[-1].set_output_delete_policy(output_ref_name, DeletePolicy.KEEP)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=input_ref,
                algorithm_pipeline=noise_cov_pipeline,
            )
            task.id = 1002

            task_plan = app_services.task_planner.plan_semantic_task(task)

            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=10)

            output_ref = task_plan.bindings[output_ref_name]

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_cov, _ = storage_client.read_data(output_ref)

            noise_yxb = array_2x2x4.transpose(1, 2, 0)
            flattened_noise = noise_yxb.reshape(-1, noise_yxb.shape[2])
            # rowvar=False because we are getting the noise in a channel
            expected_cov = np.cov(flattened_noise, rowvar=False).astype(np.float32)[..., None]
            self.assertTrue(np.allclose(output_cov, expected_cov, atol=1e-5))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_covariance_stage_filters_bad_bands_and_uses_num_features(self) -> None:
        array_2x2x3 = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[100.0, 100.0], [100.0, 100.0]],
                [[10.0, 20.0], [30.0, 40.0]],
            ],
            dtype=np.float32,
        )
        dataset = RasterDataLoader().dataset_from_numpy_array(array_2x2x3)
        dataset.set_bad_bands([1, 0, 1])

        app_services = AppServices()
        storage_client = None
        try:
            input_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )

            mean_output_ref_name = "cov_filtered_mean"
            cov_output_ref_name = "cov_filtered_covariance"
            num_pixels = 4

            mean_stage = get_spectral_mean_stage(input_ref, mean_output_ref_name)
            cov_stage = CalcCovMatrixStage(
                _total_spectra=num_pixels,
                _output_ref_name=cov_output_ref_name,
                _num_features=2,
                default_executor="process",
                input_plan_meta=mean_stage.input_plan_meta,
                broadcast_input={"mean": DataBinding(mean_output_ref_name)},
            )
            self._keep_outputs(cov_stage, cov_output_ref_name)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=input_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[mean_stage, cov_stage]),
            )
            task.id = 10021

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=10)

            output_ref = task_plan.bindings[cov_output_ref_name]
            self.assertEqual(output_ref.shape, (2, 2, 1))

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_cov, _ = storage_client.read_data(output_ref)

            dataset_yxb = array_2x2x3.transpose(1, 2, 0)
            flattened = dataset_yxb.reshape(-1, dataset_yxb.shape[2])[:, [0, 2]]
            expected_cov = np.cov(flattened, rowvar=False).astype(np.float32)[..., None]

            self.assertEqual(output_cov.shape, (2, 2, 1))
            self.assertTrue(np.allclose(output_cov, expected_cov, atol=1e-5))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_covariance_stage_computes_internal_total_when_not_provided(self) -> None:
        array_2x2x3 = np.array(
            [
                [[1.0, -9999.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
                [[100.0, 200.0], [300.0, np.nan]],
            ],
            dtype=np.float32,
        )
        dataset = RasterDataLoader().dataset_from_numpy_array(array_2x2x3)
        dataset.set_data_ignore_value(-9999.0)

        app_services = AppServices()
        storage_client = None
        try:
            input_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )

            mean_stage = get_spectral_mean_stage(input_ref, "cov_total_mean")
            cov_stage = CalcCovMatrixStage(
                _total_spectra=0,
                _output_ref_name="cov_total_covariance",
                _internal_total_ref_name="cov_total_ref",
                default_executor="process",
                input_plan_meta=mean_stage.input_plan_meta,
                broadcast_input={"mean": DataBinding("cov_total_mean")},
            )
            self._keep_outputs(cov_stage, "cov_total_ref")
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=input_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[mean_stage, cov_stage]),
            )
            task.id = 10022

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=20)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            internal_total = storage_client.read_json_value(task_plan.bindings["cov_total_ref"])
            self.assertEqual(internal_total, {"total": 2})
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_covariance_stage_reuses_provided_total_ref(self) -> None:
        array_2x2x3 = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
                [[100.0, 200.0], [300.0, 400.0]],
            ],
            dtype=np.float32,
        )
        dataset = RasterDataLoader().dataset_from_numpy_array(array_2x2x3)

        app_services = AppServices()
        storage_client = None
        try:
            input_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )

            mean_stage = SpectralMeanStage(
                _output_ref_name="cov_reuse_mean",
                _internal_total_ref_name="shared_total_ref",
                _meta_ref=input_ref,
                default_executor="process",
                input_plan_meta=DatasetPlanMeta(shape=(2, 2, 3), dtype=np.dtype(np.float32)),
                resource_model=ResourceModel(
                    fixed_overhead_bytes=0,
                    bytes_per_scalar_in=1,
                    bytes_per_scalar_out=1,
                    scratch_bytes_per_scalar_in=0,
                ),
            )
            cov_stage = CalcCovMatrixStage(
                _total_spectra=0,
                _output_ref_name="cov_reuse_covariance",
                _internal_total_ref_name="cov_reuse_total_ref",
                default_executor="process",
                input_plan_meta=mean_stage.input_plan_meta,
                broadcast_input={
                    "mean": DataBinding("cov_reuse_mean"),
                    "total": DataBinding("shared_total_ref"),
                },
            )
            self._keep_outputs(cov_stage, "cov_reuse_total_ref")
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=input_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[mean_stage, cov_stage]),
            )
            task.id = 10023

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=20)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            internal_total = storage_client.read_json_value(task_plan.bindings["cov_reuse_total_ref"])
            self.assertEqual(internal_total, {"total": 4})
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_eigendecomposition_pipeline_stores_lightweight_json_descriptor(self) -> None:
        app_services = AppServices()
        storage_client = None
        try:
            matrix = np.array(
                [
                    [2.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
            input_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="eigen_input_matrix",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=matrix.size * matrix.dtype.itemsize,
                    shape=matrix.shape,
                    dtype=matrix.dtype,
                )
            )
            process_storage_client = get_process_storage_client()
            process_storage_client.write_data(input_ref, matrix)

            output_ref_name = "eigen_descriptor"
            eigen_pipeline = get_eigendecomposition_pipeline(input_ref, output_ref_name)
            self._keep_outputs(
                eigen_pipeline.stages[-1],
                output_ref_name,
                f"{output_ref_name}_vectors",
                f"{output_ref_name}_values",
            )
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=input_ref,
                algorithm_pipeline=eigen_pipeline,
            )
            task.id = 1003

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=10)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )

            descriptor_ref = task_plan.bindings[output_ref_name]
            envelope_payload = storage_client.read_json_value(descriptor_ref)
            self.assertIn("eigen", envelope_payload)
            descriptor: EigenVectorsAndValues = envelope_payload["eigen"]
            self.assertIsInstance(descriptor, EigenVectorsAndValues)

            self.assertEqual(descriptor.count(), 2)
            eigen_values = np.array([descriptor.get_eigen_value(0), descriptor.get_eigen_value(1)])
            self.assertTrue(np.allclose(eigen_values, np.array([2.0, 1.0]), atol=1e-5))

            for i in range(descriptor.count()):
                vector_i = descriptor.get_eigen_vector(i)
                value_i = descriptor.get_eigen_value(i)
                self.assertTrue(np.allclose(matrix @ vector_i, value_i * vector_i, atol=1e-5))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_whitening_matrix_stage_computes_lambda_inverse_sqrt_times_e_transpose(self) -> None:
        app_services = AppServices()
        storage_client = None
        try:
            process_storage_client = get_process_storage_client()
            vectors = np.array(
                [
                    [0.6, 0.8],
                    [-0.8, 0.6],
                ],
                dtype=np.float32,
            )
            values = np.array([9.0, 4.0], dtype=np.float32)

            vectors_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="test_whiten_vectors",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=vectors.size * vectors.dtype.itemsize,
                    shape=vectors.shape,
                    dtype=vectors.dtype,
                )
            )
            values_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="test_whiten_values",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=values.size * values.dtype.itemsize,
                    shape=values.shape,
                    dtype=values.dtype,
                )
            )
            descriptor_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="test_whiten_descriptor",
                    kind="json",
                    residency="ram_cacheable",
                    size_est=1024,
                )
            )

            process_storage_client.write_data(vectors_ref, vectors)
            process_storage_client.write_data(values_ref, values)
            process_storage_client.write_json_value(
                descriptor_ref,
                {
                    "eigen": EigenVectorsAndValues(
                        eigen_vectors_ref=vectors_ref,
                        eigen_values_ref=values_ref,
                        num_vectors=2,
                        vector_dimension=2,
                    )
                },
            )

            output_ref_name = "whitening_matrix"
            stage = get_whitening_matrix_stage(descriptor_ref, output_ref_name)
            self._keep_outputs(stage, output_ref_name)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=descriptor_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 1004

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=10)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_ref = task_plan.bindings[output_ref_name]
            whitening_matrix, _ = storage_client.read_data(output_ref)

            expected = np.array(
                [
                    [0.44, -0.08],
                    [-0.08, 0.3933333],
                ],
                dtype=np.float32,
            )
            self.assertEqual(whitening_matrix.shape, vectors.shape)
            self.assertTrue(np.allclose(whitening_matrix, expected, atol=1e-6))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_apply_whitening_matrix_stage_applies_matrix_to_each_dataset_spectrum(self) -> None:
        app_services = AppServices()
        storage_client = None
        try:
            process_storage_client = get_process_storage_client()
            dataset = np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=np.float32,
            )
            whitening_matrix = np.array(
                [
                    [2.0, 0.0],
                    [0.0, 0.5],
                ],
                dtype=np.float32,
            )

            dataset_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="apply_whitening_input_dataset",
                    kind="dataset",
                    residency="ram_cacheable",
                    size_est=dataset.size * dataset.dtype.itemsize,
                    shape=dataset.shape,
                    dtype=dataset.dtype,
                )
            )
            whitening_matrix_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="apply_whitening_matrix",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=whitening_matrix.size * whitening_matrix.dtype.itemsize,
                    shape=whitening_matrix.shape,
                    dtype=whitening_matrix.dtype,
                )
            )
            process_storage_client.write_data(dataset_ref, dataset)
            process_storage_client.write_data(whitening_matrix_ref, whitening_matrix)

            output_ref_name = "noise_whitened_dataset"
            stage = get_apply_matrix_to_dataset_stage(
                dataset_ref,
                whitening_matrix_ref,
                output_ref_name,
            )
            self._keep_outputs(stage, output_ref_name)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 1005

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=10)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_ref = task_plan.bindings[output_ref_name]
            whitened_dataset, _ = storage_client.read_data(output_ref)

            expected = np.array(
                [
                    [[2.0, 1.0], [6.0, 2.0]],
                    [[10.0, 3.0], [14.0, 4.0]],
                ],
                dtype=np.float32,
            )
            self.assertEqual(whitened_dataset.shape, dataset.shape)
            self.assertTrue(np.allclose(whitened_dataset, expected, atol=1e-6))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_apply_matrix_to_dataset_stage_supports_left_and_right_matrix_chains(self) -> None:
        app_services = AppServices()
        storage_client = None
        try:
            process_storage_client = get_process_storage_client()
            dataset = np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=np.float32,
            )
            left_matrix = np.array(
                [
                    [2.0, 0.0],
                    [0.0, 0.5],
                    [1.0, -1.0],
                ],
                dtype=np.float32,
            )
            right_matrix = np.array(
                [
                    [1.0, 2.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
                dtype=np.float32,
            )

            dataset_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="apply_chain_input_dataset",
                    kind="dataset",
                    residency="ram_cacheable",
                    size_est=dataset.size * dataset.dtype.itemsize,
                    shape=dataset.shape,
                    dtype=dataset.dtype,
                )
            )
            left_matrix_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="apply_chain_left_matrix",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=left_matrix.size * left_matrix.dtype.itemsize,
                    shape=left_matrix.shape,
                    dtype=left_matrix.dtype,
                )
            )
            right_matrix_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="apply_chain_right_matrix",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=right_matrix.size * right_matrix.dtype.itemsize,
                    shape=right_matrix.shape,
                    dtype=right_matrix.dtype,
                )
            )
            process_storage_client.write_data(dataset_ref, dataset)
            process_storage_client.write_data(left_matrix_ref, left_matrix)
            process_storage_client.write_data(right_matrix_ref, right_matrix)

            output_ref_name = "matrix_chain_dataset"
            stage = get_apply_matrices_to_dataset_stage(
                dataset_ref=dataset_ref,
                left_multiply_matrices=(left_matrix_ref,),
                right_multiply_matrices=(right_matrix_ref,),
                output_ref_name=output_ref_name,
            )
            self._keep_outputs(stage, output_ref_name)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 10051

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=10)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_ref = task_plan.bindings[output_ref_name]
            transformed_dataset, _ = storage_client.read_data(output_ref)

            flattened = dataset.reshape(-1, dataset.shape[2])
            expected = (flattened @ left_matrix.T) @ right_matrix
            expected = expected.reshape(dataset.shape[0], dataset.shape[1], right_matrix.shape[1])

            self.assertEqual(transformed_dataset.shape, expected.shape)
            self.assertTrue(np.allclose(transformed_dataset, expected, atol=1e-6))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_matrix_multiplication_stage_multiplies_matrix_chain_in_order(self) -> None:
        app_services = AppServices()
        storage_client = None
        try:
            process_storage_client = get_process_storage_client()
            matrix_a = np.array(
                [
                    [1.0, 2.0, 0.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            matrix_b = np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            )
            matrix_c = np.array(
                [
                    [2.0, 1.0],
                    [0.0, 3.0],
                ],
                dtype=np.float32,
            )

            matrix_refs = []
            for name, matrix in (
                ("matrix_mult_a", matrix_a),
                ("matrix_mult_b", matrix_b),
                ("matrix_mult_c", matrix_c),
            ):
                matrix_ref = app_services.storage_service.allocate_data(
                    AllocationRequest(
                        name=name,
                        kind="array",
                        residency="ram_cacheable",
                        size_est=matrix.size * matrix.dtype.itemsize,
                        shape=matrix.shape,
                        dtype=matrix.dtype,
                    )
                )
                process_storage_client.write_data(matrix_ref, matrix)
                matrix_refs.append(matrix_ref)

            output_ref_name = "matrix_chain_product"
            stage = get_matrix_multiplication_stage(matrix_refs, output_ref_name)
            self._keep_outputs(stage, output_ref_name)
            self.assertIs(stage.chunking_scheme_type, NoChunkingScheme)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=matrix_refs[0],
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 1012

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=10)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_ref = task_plan.bindings[output_ref_name]
            product, _ = storage_client.read_data(output_ref)

            expected = matrix_a @ matrix_b @ matrix_c
            self.assertEqual(product.shape, expected.shape)
            self.assertTrue(np.allclose(product, expected, atol=1e-6))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_matrix_multiplication_stage_rejects_incompatible_chain(self) -> None:
        app_services = AppServices()
        try:
            process_storage_client = get_process_storage_client()
            matrix_a = np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                dtype=np.float32,
            )
            matrix_b = np.array(
                [
                    [1.0, 2.0, 3.0],
                ],
                dtype=np.float32,
            )

            matrix_refs = []
            for name, matrix in (("bad_matrix_a", matrix_a), ("bad_matrix_b", matrix_b)):
                matrix_ref = app_services.storage_service.allocate_data(
                    AllocationRequest(
                        name=name,
                        kind="array",
                        residency="ram_cacheable",
                        size_est=matrix.size * matrix.dtype.itemsize,
                        shape=matrix.shape,
                        dtype=matrix.dtype,
                    )
                )
                process_storage_client.write_data(matrix_ref, matrix)
                matrix_refs.append(matrix_ref)

            with self.assertRaisesRegex(ValueError, "Matrix chain shape mismatch"):
                get_matrix_multiplication_stage(matrix_refs, "bad_matrix_chain")
        finally:
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_project_onto_eigenvectors_stage_projects_to_requested_component_count(self) -> None:
        app_services = AppServices()
        storage_client = None
        try:
            process_storage_client = get_process_storage_client()
            dataset = np.array(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
                dtype=np.float32,
            )
            eigen_vectors = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            eigen_values = np.array([3.0, 2.0, 1.0], dtype=np.float32)

            dataset_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="project_input_dataset",
                    kind="dataset",
                    residency="ram_cacheable",
                    size_est=dataset.size * dataset.dtype.itemsize,
                    shape=dataset.shape,
                    dtype=dataset.dtype,
                )
            )
            eigen_vectors_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="project_eigen_vectors",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=eigen_vectors.size * eigen_vectors.dtype.itemsize,
                    shape=eigen_vectors.shape,
                    dtype=eigen_vectors.dtype,
                )
            )
            eigen_values_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="project_eigen_values",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=eigen_values.size * eigen_values.dtype.itemsize,
                    shape=eigen_values.shape,
                    dtype=eigen_values.dtype,
                )
            )
            descriptor_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="project_eigen_descriptor",
                    kind="json",
                    residency="ram_cacheable",
                    size_est=1024,
                )
            )

            process_storage_client.write_data(dataset_ref, dataset)
            process_storage_client.write_data(eigen_vectors_ref, eigen_vectors)
            process_storage_client.write_data(eigen_values_ref, eigen_values)
            process_storage_client.write_json_value(
                descriptor_ref,
                {
                    "eigen": EigenVectorsAndValues(
                        eigen_vectors_ref=eigen_vectors_ref,
                        eigen_values_ref=eigen_values_ref,
                        num_vectors=3,
                        vector_dimension=3,
                    )
                },
            )

            output_ref_name = "projected_dataset"
            stage = get_project_onto_eigenvectors_stage(
                dataset_ref=dataset_ref,
                eigen_descriptor_ref=descriptor_ref,
                num_components=2,
                output_ref_name=output_ref_name,
            )
            self._keep_outputs(stage, output_ref_name)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 1006

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=10)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_ref = task_plan.bindings[output_ref_name]
            projected_dataset, _ = storage_client.read_data(output_ref)

            expected = dataset[:, :, :2]
            self.assertEqual(projected_dataset.shape, (2, 2, 2))
            self.assertTrue(np.allclose(projected_dataset, expected, atol=1e-6))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_project_onto_eigenvectors_stage_shrinks_bad_bands_and_applies_nodata_mask(self) -> None:
        app_services = self.test_model.app_services
        storage_client = None
        try:
            process_storage_client = get_process_storage_client()
            nodata = np.float32(-9999.0)
            dataset = np.array(
                [
                    [[2.0, 100.0, 20.0], [3.0, 100.0, 30.0]],
                    [[nodata, 100.0, 40.0], [5.0, 100.0, 50.0]],
                ],
                dtype=np.float32,
            )
            bad_bands = np.array([1, 0, 1], dtype=np.int32)
            spectral_mean = np.array([1.0, 1000.0, 10.0], dtype=np.float32)
            eigen_vectors = np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
            eigen_values = np.array([3.0, 1.0], dtype=np.float32)

            dataset_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="project_bad_bands_dataset",
                    kind="dataset",
                    residency="ram_cacheable",
                    size_est=dataset.size * dataset.dtype.itemsize,
                    shape=dataset.shape,
                    dtype=dataset.dtype,
                )
            )
            mean_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="project_bad_bands_mean",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=spectral_mean.size * spectral_mean.dtype.itemsize,
                    shape=spectral_mean.shape,
                    dtype=spectral_mean.dtype,
                )
            )
            vectors_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="project_bad_bands_vectors",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=eigen_vectors.size * eigen_vectors.dtype.itemsize,
                    shape=eigen_vectors.shape,
                    dtype=eigen_vectors.dtype,
                )
            )
            values_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="project_bad_bands_values",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=eigen_values.size * eigen_values.dtype.itemsize,
                    shape=eigen_values.shape,
                    dtype=eigen_values.dtype,
                )
            )
            descriptor_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="project_bad_bands_descriptor",
                    kind="json",
                    residency="ram_cacheable",
                    size_est=1024,
                )
            )

            process_storage_client.write_data(dataset_ref, dataset)
            process_storage_client.write_meta(
                dataset_ref,
                DataMeta(
                    kind="dataset",
                    shape=dataset.shape,
                    elem_type=np.dtype(np.float32),
                    nodata=nodata,
                    bad_bands=bad_bands,
                ),
            )
            process_storage_client.write_data(mean_ref, spectral_mean)
            process_storage_client.write_data(vectors_ref, eigen_vectors)
            process_storage_client.write_data(values_ref, eigen_values)
            process_storage_client.write_json_value(
                descriptor_ref,
                {
                    "eigen": EigenVectorsAndValues(
                        eigen_vectors_ref=vectors_ref,
                        eigen_values_ref=values_ref,
                        num_vectors=2,
                        vector_dimension=2,
                    )
                },
            )

            stage = ProjectOntoEigenVectorsStage(
                _num_components=2,
                _output_ref_name="project_bad_bands_output",
                _eigen_descriptor_ref=descriptor_ref,
                _spectral_mean_ref=mean_ref,
                default_executor="process",
                input_plan_meta=get_project_onto_eigenvectors_stage(
                    dataset_ref=dataset_ref,
                    eigen_descriptor_ref=descriptor_ref,
                    num_components=2,
                    output_ref_name="project_bad_bands_output_template",
                ).input_plan_meta,
                resource_model=ResourceModel(
                    fixed_overhead_bytes=0,
                    bytes_per_scalar_in=1,
                    bytes_per_scalar_out=1,
                    scratch_bytes_per_scalar_in=0,
                ),
            )
            self._keep_outputs(stage, "project_bad_bands_output")

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 10061

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=10)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_ref = task_plan.bindings["project_bad_bands_output"]
            projected_dataset, projected_meta = storage_client.read_data(output_ref, filter_data=False)

            expected = np.array(
                [
                    [[1.0, 10.0], [2.0, 20.0]],
                    [[nodata, nodata], [4.0, 40.0]],
                ],
                dtype=np.float32,
            )
            self.assertEqual(projected_dataset.shape, (2, 2, 2))
            self.assertTrue(np.allclose(projected_dataset, expected, atol=1e-6))
            self.assertEqual(projected_meta.nodata, nodata)
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_incremental_pca_partial_fit_stage_known_answer(self) -> None:
        app_services = self.test_model.app_services
        storage_client = None
        try:
            process_storage_client = get_process_storage_client()
            dataset = np.array(
                [
                    [[1.0, 0.0], [2.0, 0.0]],
                    [[-1.0, 0.0], [-2.0, 0.0]],
                ],
                dtype=np.float32,
            )
            dataset_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="ipca_known_dataset",
                    kind="dataset",
                    residency="ram_cacheable",
                    size_est=dataset.size * dataset.dtype.itemsize,
                    shape=dataset.shape,
                    dtype=dataset.dtype,
                )
            )
            process_storage_client.write_data(dataset_ref, dataset)

            output_ref_name = "ipca_known_descriptor"
            stage = get_adaptive_pca_partial_fit_stage(
                dataset_ref=dataset_ref,
                num_components=2,
                output_ref_name=output_ref_name,
            )
            self._keep_adaptive_pca_outputs(stage)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 1007

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=10)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            descriptor_ref = task_plan.bindings[output_ref_name]
            envelope_payload = storage_client.read_json_value(descriptor_ref)
            descriptor: EigenVectorsAndValues = envelope_payload["eigen"]
            self.assertEqual(descriptor.count(), 2)

            eigen_values = np.array(
                [descriptor.get_eigen_value(0), descriptor.get_eigen_value(1)],
                dtype=np.float32,
            )
            self.assertTrue(
                np.allclose(eigen_values, np.array([10.0 / 3.0, 0.0], dtype=np.float32), atol=1e-4)
            )

            first_vec = descriptor.get_eigen_vector(0)
            second_vec = descriptor.get_eigen_vector(1)
            self.assertTrue(np.allclose(np.abs(first_vec), np.array([1.0, 0.0], dtype=np.float32), atol=1e-4))
            self.assertTrue(
                np.allclose(np.abs(second_vec), np.array([0.0, 0.0], dtype=np.float32), atol=1e-4)
            )
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_adaptive_pca_stage_default_outputs_are_reclaimed(self) -> None:
        app_services = AppServices()
        try:
            process_storage_client = get_process_storage_client()
            rng = np.random.default_rng(7)
            dataset = rng.standard_normal((4, 4, 3), dtype=np.float32)
            dataset_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="ipca_reclaim_dataset",
                    kind="dataset",
                    residency="ram_cacheable",
                    size_est=dataset.size * dataset.dtype.itemsize,
                    shape=dataset.shape,
                    dtype=dataset.dtype,
                )
            )
            process_storage_client.write_data(dataset_ref, dataset)

            output_ref_name = "ipca_reclaim_descriptor"
            stage = get_adaptive_pca_partial_fit_stage(
                dataset_ref=dataset_ref,
                num_components=3,
                output_ref_name=output_ref_name,
            )
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 1014

            task_plan = app_services.task_planner.plan_semantic_task(task)

            output_names = [
                stage._output_ref_name,
                stage._vectors_ref_name,
                stage._values_ref_name,
                stage._mean_ref_name,
                stage._covariance_ref_name,
                stage._good_band_mask_ref_name,
            ]
            planned_outputs = {name: task_plan.bindings[name] for name in output_names}
            shared_mem_names = {
                ref.ref_id: app_services.storage_service._shared_mem_handles_names.get(ref.uri)
                for ref in planned_outputs.values()
                if ref.materialization_loc == "ram" and ref.kind != "json"
            }

            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=20)

            for name, ref in planned_outputs.items():
                record = app_services.storage_service.get_lease_record(ref.ref_id)
                self.assertEqual(record.deletion_state, DeletionState.DELETED)
                self.assertNotIn(ref.ref_id, app_services.storage_service.data_refs)
                self.assertNotIn(ref.ref_id, app_services.storage_service.meta_by_ref)
                self.assertNotIn(ref.uri, app_services.storage_service.ram_objects)
                self.assertNotIn(ref.uri, app_services.storage_service.ram_est_bytes)

                shared_mem_name = shared_mem_names.get(ref.ref_id)
                # Output ref name is the only name that's not saved as a SharedMemoryAray
                if name is not output_ref_name:
                    self.assertFalse(shared_mem_exists(shared_mem_name))
        finally:
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_adaptive_pca_stage_resolves_num_components_in_pre_task_when_unset(self) -> None:
        app_services = AppServices()
        storage_client = None
        try:
            process_storage_client = get_process_storage_client()
            dataset = np.array(
                [
                    [[1.0, -9999.0], [2.0, 3.0]],
                    [[10.0, 10.0], [10.0, 10.0]],
                    [[100.0, 200.0], [np.nan, 400.0]],
                    [[1000.0, 2000.0], [3000.0, 4000.0]],
                ],
                dtype=np.float32,
            )
            dataset_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="ipca_resolve_dataset",
                    kind="dataset",
                    residency="ram_cacheable",
                    size_est=dataset.size * dataset.dtype.itemsize,
                    shape=dataset.shape,
                    dtype=dataset.dtype,
                )
            )
            process_storage_client.write_data(dataset_ref, dataset)
            app_services.storage_service.update_meta(
                dataset_ref,
                bad_bands=np.asarray([1, 1], dtype=np.int32),
                nodata=-9999.0,
            )

            stage = get_adaptive_pca_partial_fit_stage(
                dataset_ref=dataset_ref,
                num_components=None,
                output_ref_name="ipca_resolve_descriptor",
            )
            self._keep_adaptive_pca_outputs(stage, keep_resolved=True)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 10071

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=20)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            resolved_payload = storage_client.read_json_value(
                task_plan.bindings["ipca_resolve_descriptor_resolved_num_components"]
            )
            self.assertEqual(resolved_payload, {"num_components": 2})
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_incremental_pca_partial_fit_matches_covariance_eigendecomposition(self) -> None:
        app_services = AppServices()
        storage_client = None
        try:
            process_storage_client = get_process_storage_client()
            rng = np.random.default_rng(1)
            dataset = rng.standard_normal((5, 7, 3), dtype=np.float32)
            dataset_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="ipca_vs_eig_dataset",
                    kind="dataset",
                    residency="ram_cacheable",
                    size_est=dataset.size * dataset.dtype.itemsize,
                    shape=dataset.shape,
                    dtype=dataset.dtype,
                )
            )
            process_storage_client.write_data(dataset_ref, dataset)

            ipca_output_name = "ipca_vs_eig_descriptor"
            ipca_stage = get_adaptive_pca_partial_fit_stage(
                dataset_ref=dataset_ref,
                num_components=3,
                output_ref_name=ipca_output_name,
            )
            self._keep_adaptive_pca_outputs(ipca_stage)
            ipca_task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[ipca_stage]),
            )
            ipca_task.id = 1008
            ipca_plan = app_services.task_planner.plan_semantic_task(ipca_task)
            ipca_future = app_services.scheduler.run_task_plan(ipca_plan)
            ipca_future.result(timeout=10)

            mean_output_ref_name = "ipca_vs_eig_mean"
            cov_output_ref_name = "ipca_vs_eig_covariance"
            eig_output_name = "eig_descriptor_from_cov"
            num_pixels = dataset.shape[0] * dataset.shape[1]

            mean_stage = get_spectral_mean_stage(dataset_ref, mean_output_ref_name)
            cov_stage = CalcCovMatrixStage(
                _total_spectra=num_pixels,
                _output_ref_name=cov_output_ref_name,
                default_executor="process",
                input_plan_meta=mean_stage.input_plan_meta,
                broadcast_input={"mean": DataBinding(mean_output_ref_name)},
            )
            eig_stage = EigenDecompositionStage(
                _output_ref_name=eig_output_name,
                _vectors_ref_name=f"{eig_output_name}_vectors",
                _values_ref_name=f"{eig_output_name}_values",
                default_executor="process",
                input_binding=DataBinding(cov_output_ref_name),
                input_plan_meta=SpectraListPlanMeta(
                    num_spectra=dataset.shape[2],
                    spectrum_length=dataset.shape[2],
                    dtype=np.dtype(dataset.dtype),
                ),
                resource_model=ResourceModel(
                    fixed_overhead_bytes=0,
                    bytes_per_scalar_in=1,
                    bytes_per_scalar_out=1,
                    scratch_bytes_per_scalar_in=0,
                ),
            )
            self._keep_outputs(
                eig_stage,
                eig_output_name,
                f"{eig_output_name}_vectors",
                f"{eig_output_name}_values",
            )
            eig_task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[mean_stage, cov_stage, eig_stage]),
            )
            eig_task.id = 1009
            eig_plan = app_services.task_planner.plan_semantic_task(eig_task)
            eig_future = app_services.scheduler.run_task_plan(eig_plan)
            eig_future.result(timeout=10)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )

            ipca_descriptor_ref = ipca_plan.bindings[ipca_output_name]
            ipca_descriptor: EigenVectorsAndValues = storage_client.read_json_value(ipca_descriptor_ref)[
                "eigen"
            ]
            eig_descriptor_ref = eig_plan.bindings[eig_output_name]
            eig_descriptor: EigenVectorsAndValues = storage_client.read_json_value(eig_descriptor_ref)[
                "eigen"
            ]

            ipca_eigen_values = np.array(
                [ipca_descriptor.get_eigen_value(0), ipca_descriptor.get_eigen_value(1)],
                dtype=np.float32,
            )
            eig_eigen_values = np.array(
                [eig_descriptor.get_eigen_value(0), eig_descriptor.get_eigen_value(1)],
                dtype=np.float32,
            )

            self.assertTrue(np.allclose(ipca_eigen_values, eig_eigen_values, atol=1e-4))

            for i in range(dataset.shape[2]):
                ipca_vec = np.asarray(ipca_descriptor.get_eigen_vector(i), dtype=np.float32)
                eig_vec = np.asarray(eig_descriptor.get_eigen_vector(i), dtype=np.float32)
                ipca_norm = np.linalg.norm(ipca_vec)
                eig_norm = np.linalg.norm(eig_vec)
                self.assertGreater(ipca_norm, 0.0)
                self.assertGreater(eig_norm, 0.0)
                alignment = abs(float(np.dot(ipca_vec / ipca_norm, eig_vec / eig_norm)))
                self.assertTrue(np.isclose(alignment, 1.0, atol=1e-4))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_incremental_pca_partial_fit_full_pca_matches_incremental_path(self) -> None:
        app_services = AppServices()
        storage_client = None
        try:
            dataset_path = (
                Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_425_7_7.hdr"
            )
            dataset = RasterDataLoader().load_from_file(str(dataset_path))[0]
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )

            full_output_name = "ipca_full_pca_descriptor"
            full_stage = get_adaptive_pca_partial_fit_stage(
                dataset_ref=dataset_ref,
                num_components=4,
                output_ref_name=full_output_name,
            )
            self._keep_adaptive_pca_outputs(full_stage)
            full_stage.test_full_pca = True
            full_task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[full_stage]),
            )
            full_task.id = 1010
            full_plan = app_services.task_planner.plan_semantic_task(full_task)
            full_future = app_services.scheduler.run_task_plan(full_plan)
            full_future.result(timeout=20)

            incremental_output_name = "ipca_incremental_descriptor"
            incremental_stage = get_adaptive_pca_partial_fit_stage(
                dataset_ref=dataset_ref,
                num_components=4,
                output_ref_name=incremental_output_name,
            )
            self._keep_adaptive_pca_outputs(incremental_stage)
            incremental_stage.test_full_pca = False
            incremental_task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[incremental_stage]),
            )
            incremental_task.id = 1011
            incremental_plan = app_services.task_planner.plan_semantic_task(incremental_task)
            incremental_future = app_services.scheduler.run_task_plan(incremental_plan)
            incremental_future.result(timeout=20)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )

            full_descriptor_ref = full_plan.bindings[full_output_name]
            full_descriptor: EigenVectorsAndValues = storage_client.read_json_value(full_descriptor_ref)[
                "eigen"
            ]
            incremental_descriptor_ref = incremental_plan.bindings[incremental_output_name]
            incremental_descriptor: EigenVectorsAndValues = storage_client.read_json_value(
                incremental_descriptor_ref
            )["eigen"]

            self.assertEqual(full_descriptor.num_vectors, incremental_descriptor.num_vectors)
            self.assertEqual(full_descriptor.vector_dimension, incremental_descriptor.vector_dimension)

            full_values, _ = storage_client.read_data(full_descriptor.eigen_values_ref)
            incremental_values, _ = storage_client.read_data(incremental_descriptor.eigen_values_ref)
            self.assertTrue(
                np.allclose(
                    np.asarray(full_values, dtype=np.float32),
                    np.asarray(incremental_values, dtype=np.float32),
                    atol=1e-3,
                )
            )

            full_mean, _ = storage_client.read_data(full_descriptor.mean_ref)
            incremental_mean, _ = storage_client.read_data(incremental_descriptor.mean_ref)
            self.assertTrue(
                np.allclose(
                    np.asarray(full_mean, dtype=np.float32),
                    np.asarray(incremental_mean, dtype=np.float32),
                    atol=1e-4,
                )
            )

            full_vectors, _ = storage_client.read_data(full_descriptor.eigen_vectors_ref)
            incremental_vectors, _ = storage_client.read_data(incremental_descriptor.eigen_vectors_ref)
            full_vectors_array = np.asarray(full_vectors, dtype=np.float32)
            incremental_vectors_array = np.asarray(incremental_vectors, dtype=np.float32)
            self.assertEqual(full_vectors_array.shape, incremental_vectors_array.shape)

            for i in range(full_vectors_array.shape[0]):
                full_vec = full_vectors_array[i]
                incremental_vec = incremental_vectors_array[i]
                alignment = abs(float(np.dot(full_vec, incremental_vec)))
                self.assertTrue(np.isclose(alignment, 1.0, atol=1e-4))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_adaptive_pca_stage_matches_sklearn_pca_on_data_ignore_fixture(self) -> None:
        app_services = AppServices()
        storage_client = None
        try:
            dataset_path = (
                Path(__file__).resolve().parent
                / ".."
                / "test_utils"
                / "test_datasets"
                / "caltech_425_6_6_data_ignore.hdr"
            )
            dataset = RasterDataLoader().load_from_file(str(dataset_path))[0]
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )

            output_ref_name = "pca_data_ignore_descriptor"
            stage = get_adaptive_pca_partial_fit_stage(
                dataset_ref=dataset_ref,
                num_components=4,
                output_ref_name=output_ref_name,
            )
            self._keep_adaptive_pca_outputs(stage)
            stage.test_full_pca = True

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 1013

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=20)

            image_arr = dataset.get_image_data(filter_data_ignore_value=True)
            nbands = image_arr.shape[0]
            sklearn_rows = image_arr.transpose(1, 2, 0).reshape(-1, nbands)
            bad_bands = dataset.get_bad_bands()
            if bad_bands is not None:
                good_band_mask = np.asarray(bad_bands, dtype=bool)
                sklearn_rows = sklearn_rows[:, good_band_mask]
            else:
                good_band_mask = np.ones((nbands,), dtype=bool)

            sklearn_mask = np.ma.getmaskarray(sklearn_rows)
            if sklearn_mask is not np.ma.nomask:
                valid_rows = np.all(~sklearn_mask, axis=1)
                sklearn_rows = sklearn_rows.data[valid_rows, :]
            sklearn_rows = np.asarray(sklearn_rows, dtype=np.float32)
            if not np.isfinite(sklearn_rows).all():
                raise ValueError("Cleaned sklearn PCA rows still contain non-finite values")

            sklearn_pca = PCA(n_components=4)
            sklearn_pca.fit(sklearn_rows)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )

            descriptor_ref = task_plan.bindings[output_ref_name]
            descriptor: EigenVectorsAndValues = storage_client.read_json_value(descriptor_ref)["eigen"]

            stage_values, _ = storage_client.read_data(descriptor.eigen_values_ref)
            stage_vectors, _ = storage_client.read_data(descriptor.eigen_vectors_ref)
            stage_mean, _ = storage_client.read_data(descriptor.mean_ref)

            stage_values = np.asarray(stage_values, dtype=np.float32)
            stage_vectors = np.asarray(stage_vectors, dtype=np.float32)
            stage_mean = np.asarray(stage_mean, dtype=np.float32)

            expected_mean = np.zeros((nbands,), dtype=np.float32)
            expected_mean[good_band_mask] = np.asarray(sklearn_pca.mean_, dtype=np.float32)
            self.assertTrue(np.allclose(stage_values, sklearn_pca.explained_variance_, atol=1e-4))
            self.assertTrue(np.allclose(stage_mean, expected_mean, atol=1e-4))

            expected_vectors = np.zeros((4, nbands), dtype=np.float32)
            expected_vectors[:, good_band_mask] = np.asarray(sklearn_pca.components_, dtype=np.float32)
            self.assertEqual(stage_vectors.shape, expected_vectors.shape)
            for i in range(expected_vectors.shape[0]):
                stage_vec = stage_vectors[i]
                expected_vec = expected_vectors[i]
                stage_norm = np.linalg.norm(stage_vec)
                expected_norm = np.linalg.norm(expected_vec)
                self.assertGreater(stage_norm, 0.0)
                self.assertGreater(expected_norm, 0.0)
                alignment = abs(float(np.dot(stage_vec / stage_norm, expected_vec / expected_norm)))
                self.assertTrue(np.isclose(alignment, 1.0, atol=1e-4))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def _run_psd_inverse_pipeline(
        self,
        matrix: np.ndarray,
        output_ref_name: str,
        task_id: int,
    ) -> np.ndarray:
        """Allocate *matrix*, run :class:`PosSemiDefMatrixInverse`, and return the result.

        Args:
            matrix: The input square float32 PSD matrix to invert.
            output_ref_name: Unique allocation name for the pseudoinverse output.
            task_id: Unique integer ID for the :class:`SemanticTask`.

        Returns:
            The pseudoinverse as a float32 numpy array of the same shape as *matrix*.
        """
        app_services = self.test_model.app_services
        storage_client = None
        try:
            process_storage_client = get_process_storage_client()

            input_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name=f"{output_ref_name}_input",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=matrix.size * matrix.dtype.itemsize,
                    shape=matrix.shape,
                    dtype=matrix.dtype,
                )
            )
            process_storage_client.write_data(input_ref, matrix)

            stage = get_pos_semi_def_matrix_inverse_stage(input_ref, output_ref_name)
            self._keep_outputs(stage, output_ref_name)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=input_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = task_id

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=15)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_ref = task_plan.bindings[output_ref_name]
            result, _ = storage_client.read_data(output_ref)
            return np.asarray(result, dtype=np.float32)
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_pos_semi_def_matrix_inverse_identity_matrix_inverts_to_itself(self) -> None:
        """The pseudoinverse of the 3x3 identity matrix is itself."""
        identity = np.eye(3, dtype=np.float32)

        result = self._run_psd_inverse_pipeline(
            matrix=identity,
            output_ref_name="psd_inv_identity",
            task_id=2001,
        )

        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(
            np.allclose(result, identity, atol=1e-5),
            msg=f"Expected identity, got:\n{result}",
        )

    def test_pos_semi_def_matrix_inverse_known_2x2_psd_matrix(self) -> None:
        """The pseudoinverse of a known full-rank 2x2 PSD matrix matches the analytic inverse.

        For M = [[5, 2], [2, 3]], det(M) = 11, so
        M⁻¹ = (1/11) * [[3, -2], [-2, 5]].
        """
        matrix = np.array([[5.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        expected_inverse = np.array(
            [[3.0 / 11.0, -2.0 / 11.0], [-2.0 / 11.0, 5.0 / 11.0]],
            dtype=np.float32,
        )

        result = self._run_psd_inverse_pipeline(
            matrix=matrix,
            output_ref_name="psd_inv_2x2",
            task_id=2002,
        )

        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(
            np.allclose(result, expected_inverse, atol=1e-5),
            msg=f"Expected:\n{expected_inverse}\nGot:\n{result}",
        )

    def test_pos_semi_def_matrix_inverse_gram_matrix_times_inverse_is_identity(self) -> None:
        """G⁺ @ G ≈ I for a full-rank 5x5 Gram matrix G = AᵀA.

        A random (7, 5) matrix A has rank 5 with probability 1, so G = AᵀA is
        symmetric positive definite and its pseudoinverse is a true inverse.
        Multiplying G by G⁺ should recover the 5x5 identity.
        """
        rng = np.random.default_rng(42)
        A = rng.standard_normal((7, 5)).astype(np.float32)
        gram = (A.T @ A).astype(np.float32)  # (5, 5), symmetric PD

        result = self._run_psd_inverse_pipeline(
            matrix=gram,
            output_ref_name="psd_inv_gram_5x5",
            task_id=2003,
        )

        self.assertEqual(result.shape, (5, 5))

        # G⁺ @ G should equal the identity within the column space
        product = result.astype(np.float64) @ gram.astype(np.float64)
        self.assertTrue(
            np.allclose(product, np.eye(5), atol=1e-4),
            msg=f"G⁺ @ G deviates from identity:\n{product}",
        )

        # G @ G⁺ should also equal the identity
        product_right = gram.astype(np.float64) @ result.astype(np.float64)
        self.assertTrue(
            np.allclose(product_right, np.eye(5), atol=1e-4),
            msg=f"G @ G⁺ deviates from identity:\n{product_right}",
        )


if __name__ == "__main__":
    test_stage_funcs = TestTaskStageFuncs()
    test_stage_funcs.setUp()
    test_stage_funcs.test_whitening_matrix_stage_computes_lambda_inverse_sqrt_times_e_transpose()
    test_stage_funcs.tearDown()
