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
    decor_numba,
    decor_numpy,
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


class TestDecorrelationStretchNanResistance(unittest.TestCase):
    """The kernels must tolerate unmasked NaN/Inf pixels: one bad pixel must not
    poison the transform for every other pixel. Pre-fix, the covariance went
    non-finite and numba's eigh raised LinAlgError (aborting the GUI render);
    the NumPy path produced all-NaN output (black image)."""

    @staticmethod
    def _make_correlated_bands():
        rng = np.random.default_rng(0)
        height, width = 8, 9
        b0 = rng.standard_normal((height, width)) * 10.0 + 100.0
        # Correlate b1/b2 with b0 so the covariance is non-degenerate.
        b1 = 0.5 * b0 + rng.standard_normal((height, width)) * 5.0 + 50.0
        b2 = 0.3 * b0 + rng.standard_normal((height, width)) * 2.0 + 20.0
        return b0, b1, b2

    def test_kernels_tolerate_unmasked_nan_and_inf(self) -> None:
        b0, b1, b2 = self._make_correlated_bands()
        # Inject non-finite values at distinct pixels across the three bands.
        invalid_pixels = [(0, 0), (3, 4), (7, 8)]
        b0[0, 0] = np.nan
        b1[3, 4] = np.inf
        b2[7, 8] = -np.inf

        result_numpy = decor_numpy(b0, b1, b2)
        result_numba = decor_numba(b0, b1, b2)

        self.assertEqual(result_numpy.shape, (8, 9, 3))
        self.assertEqual(result_numba.shape, (8, 9, 3))

        # A pixel is valid iff all three input bands are finite there.
        valid = np.isfinite(b0) & np.isfinite(b1) & np.isfinite(b2)
        for y, x in invalid_pixels:
            self.assertFalse(valid[y, x])

        # Every valid pixel produced finite output (pre-fix: all-NaN / LinAlgError).
        self.assertTrue(np.all(np.isfinite(result_numpy[valid])))
        self.assertTrue(np.all(np.isfinite(result_numba[valid])))

        # numba and numpy agree at the valid pixels.
        self.assertTrue(np.allclose(result_numpy[valid], result_numba[valid], atol=1e-10))


if __name__ == "__main__":
    unittest.main()
