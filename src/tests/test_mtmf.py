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

_CALTECH_ENVI_MTMF_GT_PATH = (
    Path(__file__).resolve().parent
    / ".."
    / "test_utils"
    / "test_datasets"
    / "caltech_15_20_22_envi_mtmf_gt.hdr"
).resolve()

_CALTECH_BB_PATH = (
    Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "caltech_15_20_22_bb.hdr"
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

    def test_mnf_residual_orthogonal_to_target(self) -> None:
        """r = x_mnf - a·t_mnf is Λ⁻¹-orthogonal to t_mnf for every valid pixel and target.

        The MNF matched-filter score a is computed as a Mahalanobis projection:
            a_j = (t_j^T Λ⁻¹ x) / (t_j^T Λ⁻¹ t_j)

        By definition of orthogonal projection in the Λ⁻¹-weighted inner product,
        the residual r = x_mnf - a·t_mnf satisfies:
            r^T Λ⁻¹ t_j = 0  for every valid pixel and target j.
        """
        mnf_data_ref_name = "mtmf_mnf_data"
        target_mnf_ref_name = "mtmf_target_mnf"
        inv_lambda_ref_name = "mtmf_inv_lambda"
        mf_scores_ref_name = "mnf_mtmf_ortho_mf_scores"

        dataset = self.test_model.load_dataset(str(_JPL_PATH))
        app_services = self.test_model.app_services
        storage_client = None
        try:
            _, task_plan, storage_client, _ = self._run_mnf_mtmf_pipeline(
                dataset,
                output_ref_name=mf_scores_ref_name,
                task_id=5013,
                keep_ref_names=[mnf_data_ref_name, target_mnf_ref_name, inv_lambda_ref_name],
            )

            mnf_raw, _ = storage_client.read_data(task_plan.bindings[mnf_data_ref_name], filter_data=False)
            t_raw, _ = storage_client.read_data(task_plan.bindings[target_mnf_ref_name], filter_data=False)
            alpha_raw, _ = storage_client.read_data(task_plan.bindings[mf_scores_ref_name], filter_data=False)
            inv_lam_raw, _ = storage_client.read_data(
                task_plan.bindings[inv_lambda_ref_name], filter_data=False
            )

            x_mnf = np.asarray(np.ma.getdata(mnf_raw), dtype=np.float64)  # (H, W, F)
            t_mnf = np.asarray(np.ma.getdata(t_raw), dtype=np.float64)  # (N_targets, F)
            alpha = np.asarray(np.ma.getdata(alpha_raw), dtype=np.float64)  # (H, W, N_targets)
            inv_lam = np.asarray(np.ma.getdata(inv_lam_raw), dtype=np.float64)  # (F, F) diagonal
            if inv_lam.ndim == 3:
                inv_lam = np.squeeze(inv_lam, axis=2)
            if t_mnf.ndim == 1:
                t_mnf = t_mnf[np.newaxis, :]

            h, w, f = x_mnf.shape
            n_targets = t_mnf.shape[0]

            # Valid-pixel mask: exclude nodata rows (any band == nodata or NaN)
            mnf_meta = storage_client.get_meta(task_plan.bindings[mnf_data_ref_name])
            nodata = mnf_meta.nodata
            flat_x = x_mnf.reshape(-1, f)
            flat_alpha = alpha.reshape(-1, n_targets)
            if nodata is not None:
                if np.isnan(nodata):
                    valid_mask = ~np.any(np.isnan(flat_x), axis=1)
                else:
                    valid_mask = ~np.any(flat_x == nodata, axis=1)
            else:
                valid_mask = np.ones(h * w, dtype=bool)

            x_valid = flat_x[valid_mask]  # (N_valid, F)
            alpha_valid = flat_alpha[valid_mask]  # (N_valid, N_targets)

            # Precompute Λ⁻¹ t_j for each target: (N_targets, F)
            inv_lam_t = (inv_lam @ t_mnf.T).T  # (N_targets, F)

            for j in range(n_targets):
                # r = x_mnf - a_j * t_j  →  (N_valid, F)
                residual = x_valid - alpha_valid[:, j : j + 1] * t_mnf[j]

                # --- Λ⁻¹-weighted inner product: r^T Λ⁻¹ t_j  →  (N_valid,) ---
                dots = residual @ inv_lam_t[j]
                np.testing.assert_allclose(
                    dots,
                    0.0,
                    atol=1e-4,
                    err_msg=f"Residual r = x_mnf - a·t_mnf must be Λ⁻¹-orthogonal to t_mnf[{j}]",
                )
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_mnf_whitened_quadratic_form_equals_diagonal_eigenvalues(self) -> None:
        """(Σ_N^{-1/2} U)^T Σ_b (Σ_N^{-1/2} U) = diag(λ) using pipeline artifacts.

        Σ_N^{-1/2} is the noise whitening matrix ``W``, ``U`` has orthonormal columns from the
        whitened eigendecomposition (storage uses eigenvectors as rows ``A_row``, so ``U = A_row.T``),
        ``Σ_b`` is the input covariance, and ``λ`` are the whitened eigenvalues. With symmetric ``W``,
        this equals ``U^T (W Σ_b W) U``, i.e. diagonalization of the whitened covariance.
        """
        whitening_ref_name = "mtmf_mnf_noise_whitening_matrix"
        input_cov_ref_name = "mtmf_mnf_input_covariance"
        whitened_vectors_ref_name = "mtmf_mnf_whitened_eigen_vectors"
        whitened_values_ref_name = "mtmf_mnf_whitened_eigen_values"

        dataset = self.test_model.load_dataset(str(_JPL_PATH))
        app_services = self.test_model.app_services
        storage_client = None
        try:
            _, task_plan, storage_client, _ = self._run_mnf_mtmf_pipeline(
                dataset,
                output_ref_name="mnf_mtmf_quadratic_form_test",
                task_id=5012,
                keep_ref_names=[
                    whitening_ref_name,
                    input_cov_ref_name,
                    whitened_vectors_ref_name,
                    whitened_values_ref_name,
                ],
            )

            w_raw, _ = storage_client.read_data(task_plan.bindings[whitening_ref_name], filter_data=False)
            sigma_b_raw, _ = storage_client.read_data(
                task_plan.bindings[input_cov_ref_name], filter_data=False
            )
            a_raw, _ = storage_client.read_data(
                task_plan.bindings[whitened_vectors_ref_name], filter_data=False
            )
            lam_raw, _ = storage_client.read_data(
                task_plan.bindings[whitened_values_ref_name], filter_data=False
            )

            w = np.asarray(np.ma.getdata(w_raw), dtype=np.float64)
            sigma_b = np.asarray(np.ma.getdata(sigma_b_raw), dtype=np.float64)
            a_row = np.asarray(np.ma.getdata(a_raw), dtype=np.float64)
            lam = np.asarray(np.ma.getdata(lam_raw), dtype=np.float64).reshape(-1)

            if w.ndim == 3:
                w = np.squeeze(w, axis=2)
            if sigma_b.ndim == 3:
                sigma_b = np.squeeze(sigma_b, axis=2)
            if a_row.ndim == 3:
                a_row = np.squeeze(a_row, axis=2)

            assert w.ndim == 2 and sigma_b.ndim == 2 and a_row.ndim == 2
            u = a_row.T
            w_u = w @ u
            quadratic = w_u.T @ sigma_b @ w_u
            quadratic = np.diagonal(quadratic)
            expected = lam

            np.testing.assert_allclose(
                quadratic,
                expected,
                rtol=5e-4,
                atol=5e-5,
                err_msg="(W U)^T Σ_b (W U) should equal diag(whitened eigenvalues)",
            )
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_envi_mtmf_infeasibility_ranking(self) -> None:
        """Read ENVI MTMF ground-truth, extract infeasibility (band 1), rank pixels.

        Band 0 = MF score, band 1 = infeasibility.
        get_image_data() returns shape [b][y][x], so infeasibility is [1].
        Prints pixels ranked from highest to lowest infeasibility as
        (rank, y, x, infeasibility_value) tuples.  No assertion yet.
        """
        gt_dataset = self.test_model.load_dataset(str(_CALTECH_ENVI_MTMF_GT_PATH))

        # shape: [bands][lines][samples] = [2][22][20]
        image_data = np.asarray(gt_dataset.get_image_data(filter_data_ignore_value=False))
        infeasibility = image_data[1]  # (22, 20) — lines × samples

        h, w = infeasibility.shape
        # Build flat list of (y, x, value) then sort descending by value
        pixels = [(int(y), int(x), float(infeasibility[y, x])) for y in range(h) for x in range(w)]
        ranked = sorted(pixels, key=lambda p: p[2], reverse=True)

        print("\n!@# ENVI MTMF infeasibility ranking (rank, y, x, value):")
        for rank, (y, x, val) in enumerate(ranked):
            print(f"!@#   rank={rank:4d}  y={y:3d}  x={x:3d}  infeasibility={val:.6f}")

        # -----------------------------------------------------------------
        # Run our MNF-MTMF pipeline on caltech_15_20_22_bb with the
        # top-left pixel (x=0, y=0) as the target spectrum, then rank
        # the resulting infeasibility output the same way.
        # -----------------------------------------------------------------
        bb_dataset = self.test_model.load_dataset(str(_CALTECH_BB_PATH))
        app_services = self.test_model.app_services
        storage_client = None
        try:
            target_spectrum = SpectrumAtPoint(bb_dataset, (0, 0))
            target_arr = np.asarray(target_spectrum.get_spectrum(), dtype=np.float32)  # (B,)
            target_2d = target_arr[np.newaxis, :]  # (1, B)

            source_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=bb_dataset)
            )
            target_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="caltech_bb_target",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=int(target_2d.size * target_2d.dtype.itemsize),
                    shape=target_2d.shape,
                    dtype=target_2d.dtype,
                )
            )
            get_process_storage_client().write_data(target_ref, target_2d)

            mf_scores_ref_name = "caltech_bb_mf_scores"
            infeasibility_ref_name = f"{mf_scores_ref_name}_infeasibility"
            eigenvalues_ref_name = "mtmf_mnf_whitened_eigen_values"
            noise_eigen_vectors_ref_name = "mtmf_mnf_noise_eigen_vectors"
            noise_eigen_values_ref_name = "mtmf_mnf_noise_eigen_values"

            pipeline = get_mnf_mtmf_pipeline(
                dataset_ref=source_ref,
                target_spectra_ref=target_ref,
                output_ref_name=mf_scores_ref_name,
            )

            keep_set = {
                mf_scores_ref_name,
                infeasibility_ref_name,
                eigenvalues_ref_name,
                noise_eigen_vectors_ref_name,
                noise_eigen_values_ref_name,
            }
            for stage in pipeline.stages:
                for ob in stage.output_bindings:
                    if ob.name in keep_set:
                        stage.set_output_delete_policy(ob.name, DeletePolicy.KEEP)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=source_ref,
                algorithm_pipeline=pipeline,
            )
            task.id = 5020

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=180)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )

            infeas_raw, _ = storage_client.read_data(
                task_plan.bindings[infeasibility_ref_name], filter_data=False
            )
            infeas_arr = np.asarray(infeas_raw, dtype=np.float32)  # (H, W, 1)
            infeas_2d = infeas_arr[:, :, 0]  # (H, W)

            h, w = infeas_2d.shape
            our_pixels = [(int(y), int(x), float(infeas_2d[y, x])) for y in range(h) for x in range(w)]
            our_ranked = sorted(our_pixels, key=lambda p: p[2], reverse=True)

            print("\n!@# WISER MTMF infeasibility ranking (rank, y, x, value):")
            for rank, (y, x, val) in enumerate(our_ranked):
                print(f"!@#   rank={rank:4d}  y={y:3d}  x={x:3d}  infeasibility={val:.6f}")

            assert len(ranked) == len(
                our_ranked
            ), f"Ranking length mismatch: ENVI={len(ranked)}, OURS={len(our_ranked)}"
            print("\n!@# Side-by-side infeasibility ranking:")
            print(f"!@# {'ENVI (y, x, val)':>35} | {'OURS (y, x, val)':>35}\t rank")
            for rank, ((ey, ex, eval_), (oy, ox, oval)) in enumerate(zip(ranked, our_ranked)):
                print(
                    f"!@# ({ey:3d}, {ex:3d}, {eval_:10.6f}){' ':>14}"
                    f" | ({oy:3d}, {ox:3d}, {oval:10.6f}){' ':>14}\t {rank}"
                )

            # -----------------------------------------------------------------
            # Rank-distance and infeasibility-difference statistics
            # Build dict: (y, x) -> (rank, infeasibility) for each method,
            # then compare over the shared set of pixels.
            # -----------------------------------------------------------------
            envi_rank_map: dict[tuple[int, int], tuple[int, float]] = {
                (y, x): (rank, val) for rank, (y, x, val) in enumerate(ranked)
            }
            our_rank_map: dict[tuple[int, int], tuple[int, float]] = {
                (y, x): (rank, val) for rank, (y, x, val) in enumerate(our_ranked)
            }

            rank_diffs: list[float] = []
            infeas_diffs: list[float] = []
            for pixel, (envi_rank, envi_val) in envi_rank_map.items():
                if pixel not in our_rank_map:
                    continue
                our_rank, our_val = our_rank_map[pixel]
                rank_diffs.append(abs(envi_rank - our_rank))
                infeas_diffs.append(abs(envi_val - our_val))

            rank_diffs_arr = np.asarray(rank_diffs, dtype=np.float64)
            infeas_diffs_arr = np.asarray(infeas_diffs, dtype=np.float64)

            print(f"\n!@# Infeasibility rank-difference statistics (n={len(rank_diffs)}):")
            print(f"!@#   min  = {rank_diffs_arr.min():.4f}")
            print(f"!@#   max  = {rank_diffs_arr.max():.4f}")
            print(f"!@#   mean = {rank_diffs_arr.mean():.4f}")
            print(f"!@#   std  = {rank_diffs_arr.std():.4f}")

            print(f"\n!@# Infeasibility score-difference statistics (n={len(infeas_diffs)}):")
            print(f"!@#   min  = {infeas_diffs_arr.min():.6f}")
            print(f"!@#   max  = {infeas_diffs_arr.max():.6f}")
            print(f"!@#   mean = {infeas_diffs_arr.mean():.6f}")
            print(f"!@#   std  = {infeas_diffs_arr.std():.6f}")

            # -----------------------------------------------------------------
            # Compare matched-filter scores (band 0 of ENVI GT vs our output)
            # -----------------------------------------------------------------
            envi_mf = image_data[0]  # (H, W) — MF score from ENVI GT

            mf_raw, _ = storage_client.read_data(task_plan.bindings[mf_scores_ref_name], filter_data=False)
            our_mf_arr = np.asarray(mf_raw, dtype=np.float32)  # (H, W, 1)
            our_mf = our_mf_arr[:, :, 0]  # (H, W)

            all_equal = True
            mismatches = []
            for y in range(envi_mf.shape[0]):
                for x in range(envi_mf.shape[1]):
                    ev = float(envi_mf[y, x])
                    ov = float(our_mf[y, x])
                    if not np.isclose(ev, ov, rtol=1e-1, atol=1e-8):
                        all_equal = False
                        mismatches.append((y, x, ev, ov))

            if all_equal:
                print("\n!@# MF scores: all pixels match ENVI")
            else:
                print(f"\n!@# MF scores: {len(mismatches)} pixel(s) differ from ENVI")
                print(f"!@# {'ENVI (y, x, mf)':>35} | {'OURS (y, x, mf)':>35}")
                for y, x, ev, ov in mismatches:
                    print(f"!@# ({y:3d}, {x:3d}, {ev:10.6f}){' ':>14}" f" | ({y:3d}, {x:3d}, {ov:10.6f})")

            # -----------------------------------------------------------------
            # Print the diagonal of the whitened eigenvalue matrix (1-D vector)
            # -----------------------------------------------------------------
            eigenvalues_raw, _ = storage_client.read_data(
                task_plan.bindings[eigenvalues_ref_name], filter_data=False
            )
            eigenvalues = np.asarray(np.ma.getdata(eigenvalues_raw), dtype=np.float32).ravel()
            print(f"\n!@# Whitened eigenvalues (diagonal of Λ), length={len(eigenvalues)}:")
            print(f"!@# {eigenvalues}")

            # -----------------------------------------------------------------
            # Print the noise eigen vectors and values
            # -----------------------------------------------------------------
            noise_vectors_raw, _ = storage_client.read_data(
                task_plan.bindings[noise_eigen_vectors_ref_name], filter_data=False
            )
            noise_vectors = np.asarray(np.ma.getdata(noise_vectors_raw), dtype=np.float32)
            print(f"\n!@# Noise eigen vectors shape={noise_vectors.shape}:")
            print(f"!@# {noise_vectors}")

            noise_values_raw, _ = storage_client.read_data(
                task_plan.bindings[noise_eigen_values_ref_name], filter_data=False
            )
            noise_values = np.asarray(np.ma.getdata(noise_values_raw), dtype=np.float32).ravel()
            print(f"\n!@# Noise eigen values (1-D diagonal), length={len(noise_values)}:")
            print(f"!@# {noise_values}")

            # Test is still in progress, so failing so we don't forget about it
            self.assertTrue(False)
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
