import unittest
from unittest.mock import patch

import numpy as np
import pytest
import tests.context

from wiser.gui.app_services import AppServices
from wiser.utils.task_stages import (
    EigenVectorsAndValues,
    get_eigendecomposition_pipeline,
    get_noise_covariance_pipeline,
    get_spectral_mean_stage,
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
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()
