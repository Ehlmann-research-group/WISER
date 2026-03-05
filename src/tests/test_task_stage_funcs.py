import unittest
from unittest.mock import patch

import numpy as np
import pytest
import tests.context

from wiser.gui.app_services import AppServices
from wiser.utils.task_stages import (
    EigenVectorsAndValues,
    get_apply_matrix_to_dataset_stage,
    get_eigendecomposition_pipeline,
    get_noise_covariance_pipeline,
    get_spectral_mean_stage,
    get_whitening_matrix_stage,
)
from wiser.raster.loader import RasterDataLoader
from wiser.utils.primitives import AllocationRequest, PriorityClass
from wiser.utils.storage_client import StorageClient
from wiser.utils.storage_layer import ExternalRasterHandle
from wiser.utils.worker_runtime import get_process_storage_client
from wiser.utils.task_system import (
    AlgorithmPipeline,
    SemanticTask,
)

pytestmark = [
    pytest.mark.integration,
]


class TestTaskStageFuncs(unittest.TestCase):
    # def test_spectral_mean_stage_pipeline_execution(self) -> None:
    #     # RasterDataLoader expects [band][y][x]. Each pixel has a constant spectrum value.
    #     array_2x2x4 = np.array(
    #         [
    #             [[1.0, 2.0], [3.0, 4.0]],
    #             [[1.0, 2.0], [3.0, 4.0]],
    #             [[1.0, 2.0], [3.0, 4.0]],
    #             [[1.0, 2.0], [3.0, 4.0]],
    #         ],
    #         dtype=np.float32,
    #     )
    #     dataset = RasterDataLoader().dataset_from_numpy_array(array_2x2x4)

    #     app_services = AppServices()
    #     storage_client = None
    #     try:
    #         input_ref = app_services.storage_service.register_external(
    #             ExternalRasterHandle(dataset_obj=dataset)
    #         )

    #         output_ref_name = "spectral_mean"
    #         stage = get_spectral_mean_stage(input_ref, output_ref_name)

    #         task = SemanticTask(
    #             priority_class=PriorityClass.BACKGROUND,
    #             input_ref=input_ref,
    #             algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
    #         )
    #         task.id = 1001

    #         task_plan = app_services.task_planner.plan_semantic_task(task)

    #         future = app_services.scheduler.run_task_plan(task_plan)
    #         future.result(timeout=5)

    #         output_ref = task_plan.bindings[output_ref_name]

    #         listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
    #         storage_client = StorageClient(
    #             service=None,  # type: ignore[arg-type]
    #             service_address=listener_address,
    #             service_authkey=listener_authkey,
    #         )
    #         output_spectrum, _ = storage_client.read_data(output_ref)
    #         self.assertEqual(output_spectrum.shape, (4,))
    #         self.assertTrue(np.allclose(output_spectrum, 2.5))
    #     finally:
    #         if storage_client is not None:
    #             storage_client.close()
    #         app_services.scheduler.shutdown(wait=True)
    #         app_services.storage_service.close()

    # def test_noise_covariance_pipeline_execution(self) -> None:
    #     # RasterDataLoader expects [band][y][x]. In [y][x][b], top row pixels are [1,2,1,2]
    #     # and bottom row pixels are [4,1,4,1].
    #     array_2x2x4 = np.array(
    #         [
    #             [[1.0, 1.0], [4.0, 4.0]],
    #             [[2.0, 2.0], [1.0, 1.0]],
    #             [[1.0, 1.0], [4.0, 4.0]],
    #             [[2.0, 2.0], [1.0, 1.0]],
    #         ],
    #         dtype=np.float32,
    #     )
    #     dataset = RasterDataLoader().dataset_from_numpy_array(array_2x2x4)

    #     app_services = AppServices()
    #     storage_client = None
    #     try:
    #         input_ref = app_services.storage_service.register_external(
    #             ExternalRasterHandle(dataset_obj=dataset)
    #         )

    #         output_ref_name = "noise_covariance"
    #         noise_cov_pipeline = get_noise_covariance_pipeline(input_ref, output_ref_name)

    #         task = SemanticTask(
    #             priority_class=PriorityClass.BACKGROUND,
    #             input_ref=input_ref,
    #             algorithm_pipeline=noise_cov_pipeline,
    #         )
    #         task.id = 1002

    #         task_plan = app_services.task_planner.plan_semantic_task(task)

    #         future = app_services.scheduler.run_task_plan(task_plan)
    #         future.result(timeout=10)

    #         output_ref = task_plan.bindings[output_ref_name]

    #         listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
    #         storage_client = StorageClient(
    #             service=None,  # type: ignore[arg-type]
    #             service_address=listener_address,
    #             service_authkey=listener_authkey,
    #         )
    #         output_cov, _ = storage_client.read_data(output_ref)

    #         expected_cov = np.array(
    #             [
    #                 [3.0, -1.0, 3.0, -1.0],
    #                 [-1.0, 0.33333333, -1.0, 0.33333333],
    #                 [3.0, -1.0, 3.0, -1.0],
    #                 [-1.0, 0.33333333, -1.0, 0.33333333],
    #             ],
    #             dtype=np.float32,
    #         )[..., None]
    #         self.assertTrue(np.allclose(output_cov, expected_cov))
    #     finally:
    #         if storage_client is not None:
    #             storage_client.close()
    #         app_services.scheduler.shutdown(wait=True)
    #         app_services.storage_service.close()

    # def test_eigendecomposition_pipeline_stores_lightweight_json_descriptor(self) -> None:
    #     app_services = AppServices()
    #     storage_client = None
    #     try:
    #         matrix = np.array(
    #             [
    #                 [2.0, 0.0],
    #                 [0.0, 1.0],
    #             ],
    #             dtype=np.float32,
    #         )
    #         input_ref = app_services.storage_service.allocate_data(
    #             AllocationRequest(
    #                 name="eigen_input_matrix",
    #                 kind="array",
    #                 residency="ram_cacheable",
    #                 size_est=matrix.size * matrix.dtype.itemsize,
    #                 shape=matrix.shape,
    #                 dtype=matrix.dtype,
    #             )
    #         )
    #         process_storage_client = get_process_storage_client()
    #         process_storage_client.write_data(input_ref, matrix)

    #         output_ref_name = "eigen_descriptor"
    #         eigen_pipeline = get_eigendecomposition_pipeline(input_ref, output_ref_name)
    #         task = SemanticTask(
    #             priority_class=PriorityClass.BACKGROUND,
    #             input_ref=input_ref,
    #             algorithm_pipeline=eigen_pipeline,
    #         )
    #         task.id = 1003

    #         task_plan = app_services.task_planner.plan_semantic_task(task)
    #         future = app_services.scheduler.run_task_plan(task_plan)
    #         future.result(timeout=10)

    #         listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
    #         storage_client = StorageClient(
    #             service=None,  # type: ignore[arg-type]
    #             service_address=listener_address,
    #             service_authkey=listener_authkey,
    #         )

    #         descriptor_ref = task_plan.bindings[output_ref_name]
    #         envelope_payload = storage_client.read_json_value(descriptor_ref)
    #         self.assertIn("eigen", envelope_payload)
    #         descriptor: EigenVectorsAndValues = envelope_payload["eigen"]
    #         self.assertIsInstance(descriptor, EigenVectorsAndValues)

    #         self.assertEqual(descriptor.count(), 2)
    #         eigen_values = np.array([descriptor.get_eigen_value(0), descriptor.get_eigen_value(1)])
    #         self.assertTrue(np.allclose(eigen_values, np.array([2.0, 1.0]), atol=1e-5))

    #         for i in range(descriptor.count()):
    #             vector_i = descriptor.get_eigen_vector(i)
    #             value_i = descriptor.get_eigen_value(i)
    #             self.assertTrue(np.allclose(matrix @ vector_i, value_i * vector_i, atol=1e-5))
    #     finally:
    #         if storage_client is not None:
    #             storage_client.close()
    #         app_services.scheduler.shutdown(wait=True)
    #         app_services.storage_service.close()

    # def test_whitening_matrix_stage_computes_lambda_inverse_sqrt_times_e_transpose(self) -> None:
    #     app_services = AppServices()
    #     storage_client = None
    #     try:
    #         process_storage_client = get_process_storage_client()
    #         vectors = np.array(
    #             [
    #                 [0.6, 0.8],
    #                 [-0.8, 0.6],
    #             ],
    #             dtype=np.float32,
    #         )
    #         values = np.array([9.0, 4.0], dtype=np.float32)

    #         vectors_ref = app_services.storage_service.allocate_data(
    #             AllocationRequest(
    #                 name="test_whiten_vectors",
    #                 kind="array",
    #                 residency="ram_cacheable",
    #                 size_est=vectors.size * vectors.dtype.itemsize,
    #                 shape=vectors.shape,
    #                 dtype=vectors.dtype,
    #             )
    #         )
    #         values_ref = app_services.storage_service.allocate_data(
    #             AllocationRequest(
    #                 name="test_whiten_values",
    #                 kind="array",
    #                 residency="ram_cacheable",
    #                 size_est=values.size * values.dtype.itemsize,
    #                 shape=values.shape,
    #                 dtype=values.dtype,
    #             )
    #         )
    #         descriptor_ref = app_services.storage_service.allocate_data(
    #             AllocationRequest(
    #                 name="test_whiten_descriptor",
    #                 kind="json",
    #                 residency="ram_cacheable",
    #                 size_est=1024,
    #             )
    #         )

    #         process_storage_client.write_data(vectors_ref, vectors)
    #         process_storage_client.write_data(values_ref, values)
    #         process_storage_client.write_json_value(
    #             descriptor_ref,
    #             {
    #                 "eigen": EigenVectorsAndValues(
    #                     eigen_vectors_ref=vectors_ref,
    #                     eigen_values_ref=values_ref,
    #                     num_vectors=2,
    #                     vector_dimension=2,
    #                 )
    #             },
    #         )

    #         output_ref_name = "whitening_matrix"
    #         stage = get_whitening_matrix_stage(descriptor_ref, output_ref_name)
    #         task = SemanticTask(
    #             priority_class=PriorityClass.BACKGROUND,
    #             input_ref=descriptor_ref,
    #             algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
    #         )
    #         task.id = 1004

    #         task_plan = app_services.task_planner.plan_semantic_task(task)
    #         future = app_services.scheduler.run_task_plan(task_plan)
    #         future.result(timeout=10)

    #         listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
    #         storage_client = StorageClient(
    #             service=None,  # type: ignore[arg-type]
    #             service_address=listener_address,
    #             service_authkey=listener_authkey,
    #         )
    #         output_ref = task_plan.bindings[output_ref_name]
    #         whitening_matrix, _ = storage_client.read_data(output_ref)

    #         expected = np.array(
    #             [
    #                 [0.2, 0.26666668],
    #                 [-0.4, 0.3],
    #             ],
    #             dtype=np.float32,
    #         )
    #         self.assertEqual(whitening_matrix.shape, vectors.shape)
    #         self.assertTrue(np.allclose(whitening_matrix, expected, atol=1e-6))
    #     finally:
    #         if storage_client is not None:
    #             storage_client.close()
    #         app_services.scheduler.shutdown(wait=True)
    #         app_services.storage_service.close()

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
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()
