import unittest
from unittest.mock import patch

import numpy as np
import pytest
import tests.context

from wiser.gui.app_services import AppServices
from wiser.gui.mnf import SpectralMean, get_spectral_mean_stage
from wiser.raster.loader import RasterDataLoader
from wiser.utils.primitives import DataBinding, PriorityClass, SpatialTileScheme, SpectralBatchDatasetScheme
from wiser.utils.storage_client import StorageClient
from wiser.utils.storage_layer import ExternalRasterHandle
from wiser.utils.task_system import (
    AlgorithmPipeline,
    DatasetPlanMeta,
    ResourceModel,
    SemanticTask,
    SimpleChunkingPolicy,
)

pytestmark = [
    pytest.mark.integration,
]


class TestTaskStageFuncs(unittest.TestCase):
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

            stage_name = "spectral_mean"
            stage = get_spectral_mean_stage(input_ref, stage_name)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=input_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 1001

            task_plan = app_services.task_planner.plan_semantic_task(task)

            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=5)

            output_ref = task_plan.bindings[stage_name]

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
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()
