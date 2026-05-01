import unittest
from pathlib import Path

import numpy as np
import pytest

import tests.context  # noqa: F401 – sets up sys.path

from test_utils.memory_cleanup import release_kept_refs
from test_utils.test_model import WiserTestModel
from wiser.gui.mtmf import get_mtmf_pipeline
from wiser.raster.spectrum import SpectrumAtPoint
from wiser.utils.primitives import AllocationRequest, DeletePolicy, ExternalRasterHandle, PriorityClass
from wiser.utils.storage_client import StorageClient
from wiser.utils.task_system import SemanticTask
from wiser.utils.worker_runtime import get_process_storage_client

pytestmark = [
    pytest.mark.integration,
]

_JPL_PATH = (
    Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_425_7_7.hdr"
).resolve()

# Pixel (x=4, y=4) in image-coordinate convention used by SpectrumAtPoint
_TARGET_POINT = (4, 4)


class TestMTMF(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _run_mtmf_pipeline(self, dataset, *, output_ref_name: str, task_id: int):
        """Register dataset as both source and noise, use center pixel as target.

        Returns ``(scores_array, storage_client, app_services)`` where
        ``scores_array`` has shape ``(H, W, 1)`` (one target).
        The caller is responsible for closing ``storage_client`` and
        calling ``release_kept_refs`` / shutdown.
        """
        app_services = self.test_model.app_services

        source_ref = app_services.storage_service.register_external(ExternalRasterHandle(dataset_obj=dataset))
        noise_ref = app_services.storage_service.register_external(ExternalRasterHandle(dataset_obj=dataset))

        # Build a single-pixel target spectrum from the center pixel (4, 4).
        target_spectrum = SpectrumAtPoint(dataset, _TARGET_POINT)
        target_arr = np.asarray(target_spectrum.get_spectrum(), dtype=np.float32)  # (B,)
        target_2d = target_arr[np.newaxis, :]  # (1, B)

        target_ref = app_services.storage_service.allocate_data(
            AllocationRequest(
                name="mtmf_test_target",
                kind="array",
                residency="ram_cacheable",
                size_est=int(target_2d.size * target_2d.dtype.itemsize),
                shape=target_2d.shape,
                dtype=target_2d.dtype,
            )
        )
        get_process_storage_client().write_data(target_ref, target_2d)

        pipeline = get_mtmf_pipeline(
            source_ref=source_ref,
            noise_ref=noise_ref,
            target_spectra_ref=target_ref,
            output_ref_name=output_ref_name,
        )

        # Keep the final output alive so we can read it back.
        pipeline.stages[-1].set_output_delete_policy(output_ref_name, DeletePolicy.KEEP)

        task = SemanticTask(
            priority_class=PriorityClass.BACKGROUND,
            input_ref=source_ref,
            algorithm_pipeline=pipeline,
            extra_plan_bindings={"mtmf_noise_ref": noise_ref},
        )
        task.id = task_id

        task_plan = app_services.task_planner.plan_semantic_task(task)
        future = app_services.scheduler.run_task_plan(task_plan)
        future.result(timeout=180)

        listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
        storage_client = StorageClient(
            service=None,  # type: ignore[arg-type]
            service_address=listener_address,
            service_authkey=listener_authkey,
        )
        output_ref = task_plan.bindings[output_ref_name]
        scores, _ = storage_client.read_data(output_ref, filter_data=False)
        return np.asarray(scores, dtype=np.float32), storage_client, app_services

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------

    def test_mtmf_pipeline_runs_without_error_on_jpl_fixture(self) -> None:
        """Pipeline completes and output binding is present."""
        dataset = self.test_model.load_dataset(str(_JPL_PATH))
        app_services = self.test_model.app_services
        storage_client = None
        try:
            scores, storage_client, _ = self._run_mtmf_pipeline(
                dataset,
                output_ref_name="mtmf_smoke",
                task_id=5001,
            )
            self.assertIsNotNone(scores)
            self.assertEqual(scores.ndim, 3)
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    # def test_mtmf_output_shape_matches_input_spatial_dims_and_one_target(self) -> None:
    #     """Output shape is (H, W, 1) for one target spectrum."""
    #     dataset = self.test_model.load_dataset(str(_JPL_PATH))
    #     app_services = self.test_model.app_services
    #     storage_client = None
    #     try:
    #         scores, storage_client, _ = self._run_mtmf_pipeline(
    #             dataset,
    #             output_ref_name="mtmf_shape",
    #             task_id=5002,
    #         )

    #         # Dataset is 7×7 spatially; one target → (7, 7, 1)
    #         image_by_band = np.asarray(
    #             dataset.get_image_data(filter_data_ignore_value=False)
    #         )
    #         _bands, h, w = image_by_band.shape
    #         self.assertEqual(scores.shape, (h, w, 1))
    #     finally:
    #         if storage_client is not None:
    #             storage_client.close()
    #         release_kept_refs(app_services)
    #         app_services.scheduler.shutdown(wait=True)
    #         app_services.storage_service.close()

    # def test_mtmf_center_pixel_score_is_close_to_one(self) -> None:
    #     """The matched-filter score at the target pixel should be ≈ 1.

    #     When the target spectrum equals a pixel, the matched-filter formula
    #     T(x) = (t-μ)ᵀΓ⁻¹(x-μ) / (t-μ)ᵀΓ⁻¹(t-μ) evaluates to exactly 1
    #     for that pixel (modulo floating-point rounding).
    #     """
    #     dataset = self.test_model.load_dataset(str(_JPL_PATH))
    #     app_services = self.test_model.app_services
    #     storage_client = None
    #     try:
    #         scores, storage_client, _ = self._run_mtmf_pipeline(
    #             dataset,
    #             output_ref_name="mtmf_center",
    #             task_id=5003,
    #         )

    #         x, y = _TARGET_POINT
    #         center_score = float(scores[y, x, 0])
    #         self.assertAlmostEqual(
    #             center_score,
    #             1.0,
    #             delta=0.05,
    #             msg=f"Expected center-pixel score ≈ 1.0, got {center_score:.6f}",
    #         )
    #     finally:
    #         if storage_client is not None:
    #             storage_client.close()
    #         release_kept_refs(app_services)
    #         app_services.scheduler.shutdown(wait=True)
    #         app_services.storage_service.close()


if __name__ == "__main__":
    t = TestMTMF()
    t.setUp()
    try:
        t.test_mtmf_pipeline_runs_without_error_on_jpl_fixture()
        t.test_mtmf_output_shape_matches_input_spatial_dims_and_one_target()
        t.test_mtmf_center_pixel_score_is_close_to_one()
    finally:
        t.tearDown()
