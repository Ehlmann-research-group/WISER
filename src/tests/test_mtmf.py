from typing import Optional
import unittest
from pathlib import Path

import numpy as np
import pytest

import tests.context  # noqa: F401 – sets up sys.path

from test_utils.memory_cleanup import release_kept_refs
from test_utils.test_model import WiserTestModel
from wiser.gui.mtmf import get_mnf_mtmf_pipeline, get_mtmf_pipeline
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

_CALTECH_PATH = (
    Path(__file__).resolve().parent
    / ".."
    / "test_utils"
    / "test_datasets"
    / "caltech_425_6_6_data_ignore.hdr"
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

    def _run_mnf_mtmf_pipeline(
        self,
        dataset,
        *,
        output_ref_name: str,
        task_id: int,
        keep_ref_names: Optional[list] = None,
    ):
        """Run get_mnf_mtmf_pipeline and keep selected intermediate refs alive.

        Returns ``(source_ref, task_plan, storage_client, app_services)``.
        The caller is responsible for closing ``storage_client`` and
        calling ``release_kept_refs`` / shutdown.
        """
        if keep_ref_names is None:
            keep_ref_names = []

        app_services = self.test_model.app_services

        source_ref = app_services.storage_service.register_external(ExternalRasterHandle(dataset_obj=dataset))

        target_spectrum = SpectrumAtPoint(dataset, _TARGET_POINT)
        target_arr = np.asarray(target_spectrum.get_spectrum(), dtype=np.float32)
        target_2d = target_arr[np.newaxis, :]

        target_ref = app_services.storage_service.allocate_data(
            AllocationRequest(
                name="mnf_mtmf_test_target",
                kind="array",
                residency="ram_cacheable",
                size_est=int(target_2d.size * target_2d.dtype.itemsize),
                shape=target_2d.shape,
                dtype=target_2d.dtype,
            )
        )
        get_process_storage_client().write_data(target_ref, target_2d)

        pipeline = get_mnf_mtmf_pipeline(
            dataset_ref=source_ref,
            target_spectra_ref=target_ref,
            output_ref_name=output_ref_name,
        )

        keep_set = set(keep_ref_names) | {output_ref_name}
        for stage in pipeline.stages:
            for ob in stage.output_bindings:
                if ob.name in keep_set:
                    stage.set_output_delete_policy(ob.name, DeletePolicy.KEEP)

        task = SemanticTask(
            priority_class=PriorityClass.BACKGROUND,
            input_ref=source_ref,
            algorithm_pipeline=pipeline,
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

        return source_ref, task_plan, storage_client, app_services

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------

    def test_mnf_data_mean_is_near_zero(self) -> None:
        """MNF output is mean-centered, so its per-band mean over valid pixels should be ≈ 0."""
        dataset = self.test_model.load_dataset(str(_JPL_PATH))
        app_services = self.test_model.app_services
        storage_client = None
        try:
            mnf_data_ref_name = "mtmf_mnf_data"
            _, task_plan, storage_client, _ = self._run_mnf_mtmf_pipeline(
                dataset,
                output_ref_name="mnf_mtmf_mean_test",
                task_id=5010,
                keep_ref_names=[mnf_data_ref_name],
            )

            mnf_ref = task_plan.bindings[mnf_data_ref_name]
            mnf_data, _ = storage_client.read_data(mnf_ref, filter_data=False)
            mnf_arr = np.asarray(mnf_data, dtype=np.float32)  # (H, W, num_features)

            # Compute per-band mean across all pixels
            band_means = mnf_arr.reshape(-1, mnf_arr.shape[2]).mean(axis=0)
            np.testing.assert_allclose(
                band_means,
                0.0,
                atol=1e-3,
                err_msg="MNF output should be mean-centered; per-band mean must be near zero",
            )
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_transform_matrix_applied_to_input_matches_mnf_data(self) -> None:
        """Manually applying the MNF transform matrix to mean-centered input reproduces the MNF cube.

        The pipeline stores T = A @ W where A are the whitened eigenvectors and W is the noise
        whitening matrix.  For every valid pixel x (bad bands stripped):
            mnf(x)  ==  T @ (x - μ)
        """
        # Note: test currently doens't work when using _CALTECH_PATH (but does with _JPL_PATH)
        # this is likely due to a difference in the cleaning of bad bands and data ignore values
        dataset = self.test_model.load_dataset(str(_CALTECH_PATH))
        app_services = self.test_model.app_services
        storage_client = None
        try:
            mnf_data_ref_name = "mtmf_mnf_data"
            transform_matrix_ref_name = "mtmf_mnf_transform_matrix"
            mnf_input_mean_ref_name = "mtmf_mnf_input_spectral_mean"

            source_ref, task_plan, storage_client, _ = self._run_mnf_mtmf_pipeline(
                dataset,
                output_ref_name="mnf_mtmf_transform_test",
                task_id=5011,
                keep_ref_names=[mnf_data_ref_name, transform_matrix_ref_name, mnf_input_mean_ref_name],
            )

            # --- read raw input data ---
            source_data, _ = storage_client.read_data(source_ref, filter_data=False)
            source_arr = np.asarray(source_data, dtype=np.float32)  # (H, W, B_total)
            source_meta = storage_client.get_meta(source_ref)

            # Strip bad bands to reach (H, W, num_features)
            if source_meta.bad_bands is not None:
                good_band_indices = np.where(np.asarray(source_meta.bad_bands) != 0)[0]
                source_good = source_arr[:, :, good_band_indices]
            else:
                source_good = source_arr

            # Build valid-pixel mask (exclude nodata pixels)
            if source_meta.nodata is not None:
                valid_mask = ~np.all(source_arr == source_meta.nodata, axis=2)
            else:
                valid_mask = np.ones(source_arr.shape[:2], dtype=bool)

            x_valid = source_good[valid_mask]  # (N_valid, num_features)

            # --- read pipeline intermediates ---
            T_raw, _ = storage_client.read_data(
                task_plan.bindings[transform_matrix_ref_name], filter_data=False
            )
            T = np.asarray(T_raw, dtype=np.float32)  # (num_features, num_features)

            mean_raw, _ = storage_client.read_data(
                task_plan.bindings[mnf_input_mean_ref_name], filter_data=False
            )
            mean = np.asarray(mean_raw, dtype=np.float32).reshape(-1)  # (num_features,)

            mnf_raw, _ = storage_client.read_data(task_plan.bindings[mnf_data_ref_name], filter_data=False)
            mnf_arr = np.asarray(mnf_raw, dtype=np.float32)  # (H, W, num_features)
            mnf_valid = mnf_arr[valid_mask]  # (N_valid, num_features)

            # --- apply transform manually: mnf = (x - μ) @ T.T ---
            x_centered = x_valid - mean
            result = x_centered @ T.T

            np.testing.assert_allclose(
                result,
                mnf_valid,
                rtol=1e-4,
                atol=1e-4,
                err_msg="Manually applied transform matrix must match the MNF data cube",
            )
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

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
