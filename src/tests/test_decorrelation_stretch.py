import unittest
from pathlib import Path

import numba as nb
import numpy as np
import pytest

import tests.context  # noqa: F401

from test_utils.memory_cleanup import release_kept_refs
from wiser.raster.decorrelation_stretch import (
    get_decorrelation_stretch_pipeline,
    compute_decorrelation_stretch,
    compute_decorrelation_stretch_numba,
    compute_decorrelation_stretch_numpy,
)
from wiser.utils.task_stage_utils import (
    EigenVectorsAndValues,
    get_decorrelation_stretch_stage,
)
from wiser.utils.primitives import (
    AllocationRequest,
    DataMeta,
    DeletePolicy,
    ExternalRasterHandle,
    PriorityClass,
)
from wiser.utils.storage_client import StorageClient
from wiser.utils.worker_runtime import get_process_storage_client
from wiser.utils.task_system import AlgorithmPipeline, SemanticTask
from test_utils.test_model import WiserTestModel


@nb.njit
def _apply_decorr_transform(flat_centered: np.ndarray, T: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Apply (x - mean) @ T + mean for all pixels in flat_centered."""
    return flat_centered @ T + mean


pytestmark = [
    pytest.mark.integration,
]


class TestDecorrelationStretchStage(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_full_decorrelation_pipeline_matches_envi_ground_truth(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_15_7_7.hdr"
        ).resolve()
        gt_path = (
            Path(__file__).resolve().parent
            / ".."
            / "test_utils"
            / "test_datasets"
            / "jpl_15_7_7_decor_envi_gt.hdr"
        ).resolve()

        dataset = self.test_model.load_dataset(str(dataset_path))
        gt_dataset = self.test_model.load_dataset(str(gt_path))

        app_services = self.test_model.app_services
        storage_client = None
        try:
            # GT band names: "Decor (Band 15), Decor (Band 12), Decor (Band 1)"
            # → 0-indexed bands [14, 11, 0]
            bands = [14, 11, 0]

            # --- Semantic task (WISER pipeline) output ---
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            output_ref_name = "decorr_gt_output"
            pipeline = get_decorrelation_stretch_pipeline(
                dataset_ref, bands=bands, output_ref_name=output_ref_name
            )
            pipeline.stages[-1].set_output_delete_policy(output_ref_name, DeletePolicy.KEEP)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=pipeline,
            )
            task.id = 3002

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=60)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_ref = task_plan.bindings[output_ref_name]
            our_output, _ = storage_client.read_data(output_ref)

            # --- Ground truth ---
            # get_image_data() returns [b][y][x]; transpose to [y][x][b]
            gt_yxb = np.asarray(gt_dataset.get_image_data(), dtype=np.float64).transpose(1, 2, 0)

            self.assertEqual(our_output.shape, gt_yxb.shape)
            self.assertTrue(np.allclose(our_output, gt_yxb, atol=1e-3))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_all_decorrelation_compute_functions_return_same_result(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_15_7_7.hdr"
        ).resolve()
        dataset = self.test_model.load_dataset(str(dataset_path))
        app_services = self.test_model.app_services
        app_state = self.test_model.app_state
        bands = (14, 11, 0)
        try:
            result_numba = compute_decorrelation_stretch_numba(dataset, bands)
            result_numpy = compute_decorrelation_stretch_numpy(dataset, bands)
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            result_pipeline = np.asarray(
                compute_decorrelation_stretch(
                    app_state=app_state,
                    app_services=app_services,
                    source_dataset=dataset,
                    input_ref=dataset_ref,
                    bands=bands,
                )
            )
            self.assertEqual(result_numba.shape, result_numpy.shape)
            self.assertEqual(result_numba.shape, result_pipeline.shape)
            self.assertTrue(np.allclose(result_numba, result_numpy, atol=1e-10))
            self.assertTrue(np.allclose(result_numba, result_pipeline, atol=1e-3))
        finally:
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()


if __name__ == "__main__":
    unittest.main()
