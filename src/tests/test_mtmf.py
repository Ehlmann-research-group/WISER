from typing import Optional
import sys
import unittest
from pathlib import Path

import numpy as np

import pytest

import tests.context  # noqa: F401 – sets up sys.path

from test_utils.memory_cleanup import release_kept_refs
from test_utils.test_model import WiserTestModel
from wiser.gui.mtmf import get_mnf_mtmf_pipeline
from wiser.raster.loader import RasterDataLoader
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

_JPL_SMALL_PATH = (
    Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_15_7_7.hdr"
).resolve()

_JPL_SMALL_ENVI_MTMF_GT_PATH = (
    Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_15_7_7_envi_mtmf_gt.hdr"
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

_MTMF_TESTING_DIR = Path(r"C:\Users\jgarc\OneDrive\Documents\Data\MTMF_testing\full_jpl_mnf_v1.hdr")

# Pixel (x=4, y=4) in image-coordinate convention used by SpectrumAtPoint
_TARGET_POINT = (4, 4)


def compare_datasets(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Compare two (H, W, B) arrays: means, covariances, and eigendecompositions.

    Prints:
      - Per-band means side-by-side and their difference statistics.
      - Eigenvalues of each covariance matrix side-by-side with difference.
      - First 5 rows of each covariance matrix side-by-side, one comparison
        column per entry.
    """
    n_bands = arr_a.shape[-1]
    pix_a = arr_a.reshape(-1, n_bands)  # (N, B)
    pix_b = arr_b.reshape(-1, n_bands)

    mean_a = pix_a.mean(axis=0)  # (B,)
    mean_b = pix_b.mean(axis=0)

    mean_diff = mean_a - mean_b
    print(f"\n === compare_datasets: {label_a!r} vs {label_b!r} ===")
    print(f" Spectral means — {'idx':>5}  {label_a:>20}  {label_b:>20}  {'diff':>20}")
    for i in range(n_bands):
        print(f"                  {i:>5}  {mean_a[i]:>20.8f}  {mean_b[i]:>20.8f}  {mean_diff[i]:>20.8f}")
    print(f"\n Mean-difference statistics (n={n_bands} bands):")
    print(f"   min  = {mean_diff.min():.8f}")
    print(f"   max  = {mean_diff.max():.8f}")
    print(f"   mean = {mean_diff.mean():.8f}")
    print(f"   std  = {mean_diff.std():.8f}")

    # np.cov expects (features, observations)
    cov_a = np.cov(pix_a.T)  # (B, B)
    cov_b = np.cov(pix_b.T)

    # eigh → ascending; reverse for greatest→smallest
    evals_a_asc, evecs_a_asc = np.linalg.eigh(cov_a)
    evals_b_asc, evecs_b_asc = np.linalg.eigh(cov_b)
    evals_a = evals_a_asc[::-1]
    evals_b = evals_b_asc[::-1]

    n_show = min(n_bands, 20)
    print(f"\n Eigenvalues (greatest→smallest, first {n_show}):")
    print(f" {'idx':>5}  {label_a:>20}  {label_b:>20}  {'diff':>20}")
    for i in range(n_show):
        diff = float(evals_a[i]) - float(evals_b[i])
        print(f" {i:>5}  {evals_a[i]:>20.6f}  {evals_b[i]:>20.6f}  {diff:>20.6f}")

    # Each printed line = one column index j, showing that entry across the first
    # n_rows rows side-by-side: | cov_a[0][j] cov_b[0][j] diff[0][j] | cov_a[1][j] … |
    n_rows = min(5, n_bands)
    col_w = 12
    print(f"\n Covariance — first {n_rows} rows, one line per column index")
    print(f" Each cell: | {label_a[:col_w]:>{col_w}} {label_b[:col_w]:>{col_w}} {'diff':>{col_w}} |")
    for j in range(n_bands):
        cells = "".join(
            f"| {cov_a[r, j]:>{col_w}.6f} {cov_b[r, j]:>{col_w}.6f}"
            f" {cov_a[r, j]-cov_b[r, j]:>{col_w}.6f} "
            for r in range(n_rows)
        )
        print(f" col[{j:>4}]: {cells}|")


class TestMTMF(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    # Tests
    # ------------------------------------------------------------------

    def test_mnf_data_mean_is_near_zero(self) -> None:
        """MNF output is mean-centered, so its per-band mean over valid pixels should be ≈ 0."""
        dataset = self.test_model.load_dataset(str(_CALTECH_PATH))
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
            mnf_arr = np.asarray(np.ma.getdata(mnf_data), dtype=np.float64)  # (H, W, num_features)

            mnf_meta = storage_client.get_meta(mnf_ref)

            # Build a per-pixel valid mask using the nodata value.
            if mnf_meta.nodata is not None:
                nodata = mnf_meta.nodata
                if np.isnan(nodata):
                    valid_mask = ~np.any(np.isnan(mnf_arr), axis=2)
                else:
                    valid_mask = ~np.all(mnf_arr == nodata, axis=2)
            else:
                valid_mask = np.ones(mnf_arr.shape[:2], dtype=bool)

            valid_pixels = mnf_arr[valid_mask]  # (N_valid, num_features)

            # Strip bad bands from the feature axis.
            if mnf_meta.bad_bands is not None:
                good_band_indices = np.where(np.asarray(mnf_meta.bad_bands) != 0)[0]
                valid_pixels = valid_pixels[:, good_band_indices]

            # Compute per-band mean over valid, good-band pixels only.
            band_means = valid_pixels.mean(axis=0)

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

    @unittest.skip(
        "Should be used to validate the infeasibility score differences between ENVI's infeasibility "
        "score our ours. This is describe in the WISER Requirements -> Testing section here: "
        "https://docs.google.com/document/d/15tzSVhaqrEyCaV-NaXwp8G-caW9UUkeRii8-ql0W0YY/edit?usp=sharing"
    )
    def test_compare_envi_mnf_mf_and_infeasibility(self) -> None:
        """Read ENVI MTMF ground-truth, extract infeasibility (band 1), rank pixels.
        Compare ENVI matched filter result with outs. Compare ENVI's MNF result with ours.
        Use this for either debugging the differences in ours and ENVI's MTMF.

        Band 0 = MF score, band 1 = infeasibility.
        get_image_data() returns shape [b][y][x], so infeasibility is [1].
        Prints pixels ranked from highest to lowest infeasibility as
        (rank, y, x, infeasibility_value) tuples.  No assertion yet.
        """
        gt_dataset = self.test_model.load_dataset(
            Path(r"<path to ENVI's mtmf result for ang20160910t185702_rdn_v2n2_clip_subset").resolve()
        )

        # shape: [bands][lines][samples] = [2][22][20]
        image_data = np.asarray(gt_dataset.get_image_data(filter_data_ignore_value=False))
        infeasibility = image_data[1]  # (22, 20) — lines × samples

        h, w = infeasibility.shape
        # Build flat list of (y, x, value) then sort descending by value
        pixels = [(int(y), int(x), float(infeasibility[y, x])) for y in range(h) for x in range(w)]
        ranked = sorted(pixels, key=lambda p: p[2], reverse=True)

        # -----------------------------------------------------------------
        # Run our MNF-MTMF pipeline on the testing dataset with the
        # top-left pixel (x=0, y=0) as the target spectrum, then rank
        # the resulting infeasibility output the same way.
        # -----------------------------------------------------------------
        bb_dataset = self.test_model.load_dataset(
            Path(r"<path to ang20160910t185702_rdn_v2n2_clip_subset.hdr>").resolve()
        )
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
                    name="bb_target",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=int(target_2d.size * target_2d.dtype.itemsize),
                    shape=target_2d.shape,
                    dtype=target_2d.dtype,
                )
            )
            get_process_storage_client().write_data(target_ref, target_2d)

            mf_scores_ref_name = "bb_mf_scores"
            infeasibility_ref_name = f"{mf_scores_ref_name}_infeasibility"
            eigenvalues_ref_name = "mtmf_mnf_whitened_eigen_values"
            noise_eigen_vectors_ref_name = "mtmf_mnf_noise_eigen_vectors"
            noise_eigen_values_ref_name = "mtmf_mnf_noise_eigen_values"
            noise_cube_ref_name = "mtmf_mnf_shift_y_noise"
            mnf_data_ref_name = "mtmf_mnf_data"
            input_mean_ref_name = "mtmf_mnf_input_spectral_mean"
            transform_matrix_ref_name = "mtmf_mnf_transform_matrix"

            pipeline = get_mnf_mtmf_pipeline(
                dataset_ref=source_ref,
                target_spectra_ref=target_ref,
                output_ref_name=mf_scores_ref_name,
            )

            whitened_covariance_ref_name = "mtmf_mnf_whitened_covariance"
            input_covariance_ref_name = "mtmf_mnf_input_covariance"
            noise_whitening_matrix_ref_name = "mtmf_mnf_noise_whitening_matrix"
            keep_set = {
                mf_scores_ref_name,
                infeasibility_ref_name,
                eigenvalues_ref_name,
                noise_eigen_vectors_ref_name,
                noise_eigen_values_ref_name,
                noise_cube_ref_name,
                mnf_data_ref_name,
                input_mean_ref_name,
                transform_matrix_ref_name,
                whitened_covariance_ref_name,
                input_covariance_ref_name,
                noise_whitening_matrix_ref_name,
                "mtmf_mnf_whitened_eigen_vectors",
                "mtmf_mnf_noise_eigen_covariance",
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

            print(f"\n Infeasibility rank-difference statistics (n={len(rank_diffs)}):")
            print(f"   min  = {rank_diffs_arr.min():.4f}")
            print(f"   max  = {rank_diffs_arr.max():.4f}")
            print(f"   mean = {rank_diffs_arr.mean():.4f}")
            print(f"   std  = {rank_diffs_arr.std():.4f}")

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            _hist_path = (
                Path(__file__).resolve().parent.parent / "test_utils" / "infeas_rank_diff_histogram.png"
            )
            pval = np.percentile(rank_diffs_arr, 99.73)
            rank_diffs_trimmed = rank_diffs_arr[rank_diffs_arr <= pval]
            bin_edges = np.linspace(0, pval, 201)  # 200 bins over [0, pval]

            fig, ax = plt.subplots()
            ax.hist(rank_diffs_trimmed, bins=bin_edges)
            ax.set_xlabel("Absolute infeasibility rank difference (ENVI vs WISER)")
            ax.set_ylabel("Pixel count")
            ax.set_title(f"Infeasibility ranking difference histogram (200 bins, ≤99.73th pct={pval:.1f})")
            fig.savefig(str(_hist_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\n Rank-difference histogram saved to {_hist_path}")

            print(f"\n Infeasibility score-difference statistics (n={len(infeas_diffs)}):")
            print(f"   min  = {infeas_diffs_arr.min():.6f}")
            print(f"   max  = {infeas_diffs_arr.max():.6f}")
            print(f"   mean = {infeas_diffs_arr.mean():.6f}")
            print(f"   std  = {infeas_diffs_arr.std():.6f}")

            # -----------------------------------------------------------------
            # Compare matched-filter scores (band 0 of ENVI GT vs our output)
            # -----------------------------------------------------------------
            envi_mf = image_data[0]  # (H, W) — MF score from ENVI GT

            mf_raw, _ = storage_client.read_data(task_plan.bindings[mf_scores_ref_name], filter_data=False)
            our_mf_arr = np.asarray(mf_raw, dtype=np.float32)  # (H, W, 1)
            our_mf = our_mf_arr[:, :, 0]  # (H, W)

            _mf_rtol = 1e-1
            _mf_atol = 1e-8

            all_equal = True
            mismatches = []
            for y in range(envi_mf.shape[0]):
                for x in range(envi_mf.shape[1]):
                    ev = float(envi_mf[y, x])
                    ov = float(our_mf[y, x])
                    if not np.isclose(ev, ov, rtol=_mf_rtol, atol=_mf_atol):
                        all_equal = False
                        mismatches.append((y, x, ev, ov))

            if all_equal:
                print("\n MF scores: all pixels match ENVI")
            else:
                print(f"\n MF scores: {len(mismatches)} pixel(s) differ from ENVI")

            mf_diffs_arr = np.abs(envi_mf.astype(np.float64) - our_mf.astype(np.float64)).ravel()
            print(f"\n Pipeline vs ENVI MF score difference statistics (n={mf_diffs_arr.size}):")
            print(f"   min  = {mf_diffs_arr.min():.6f}")
            print(f"   max  = {mf_diffs_arr.max():.6f}")
            print(f"   mean = {mf_diffs_arr.mean():.6f}")
            print(f"   std  = {mf_diffs_arr.std():.6f}")

            envi_mf_flat = envi_mf.astype(np.float64).ravel()
            our_mf_flat = our_mf.astype(np.float64).ravel()
            print(f"\n ENVI MF score statistics (n={envi_mf_flat.size}):")
            print(f"   min  = {envi_mf_flat.min():.6f}")
            print(f"   max  = {envi_mf_flat.max():.6f}")
            print(f"   mean = {envi_mf_flat.mean():.6f}")
            print(f"   std  = {envi_mf_flat.std():.6f}")
            print(f"\n WISER MF score statistics (n={our_mf_flat.size}):")
            print(f"   min  = {our_mf_flat.min():.6f}")
            print(f"   max  = {our_mf_flat.max():.6f}")
            print(f"   mean = {our_mf_flat.mean():.6f}")
            print(f"   std  = {our_mf_flat.std():.6f}")

            # -----------------------------------------------------------------
            # Manual MNF calculation
            # -----------------------------------------------------------------
            import gc  # Memory can spike here, so we want to release memory quickly

            noise_cube_raw, _ = storage_client.read_data(
                task_plan.bindings[noise_cube_ref_name], filter_data=False
            )
            noise_cube_arr = np.asarray(np.ma.getdata(noise_cube_raw), dtype=np.float32)
            cov_our_noise = np.cov(noise_cube_arr.reshape(-1, noise_cube_arr.shape[-1]).astype(np.float64).T)
            del noise_cube_arr
            gc.collect()
            input_raw, _ = storage_client.read_data(source_ref, filter_data=False)
            input_arr = np.asarray(np.ma.getdata(input_raw), dtype=np.float64)
            input_data_mean = input_arr.reshape(-1, input_arr.shape[-1]).mean(axis=0, dtype=np.float64)
            noise_evals_w, noise_evecs_w = np.linalg.eigh(cov_our_noise)
            noise_evals_w = np.maximum(noise_evals_w, 1e-12)
            W = noise_evecs_w @ np.diag(noise_evals_w**-0.5) @ noise_evecs_w.T
            h_in, w_in, b_in = input_arr.shape
            centered_2d = (input_arr - input_data_mean).reshape(-1, b_in)
            del input_arr
            gc.collect()
            whitened_2d = centered_2d @ W.T
            A_evals_asc, A_evecs_asc = np.linalg.eigh(np.cov(whitened_2d.T))
            desc_idx = np.argsort(A_evals_asc)[::-1]
            A = A_evecs_asc[:, desc_idx].T
            T = A @ W
            mnf_manual = (centered_2d @ T.T).reshape(h_in, w_in, T.shape[0])
            del centered_2d
            gc.collect()
            lambda_vals = A_evals_asc[desc_idx]  # variance per MNF band (diag of C_mnf)

            # -----------------------------------------------------------------
            # Manual matched filter and infeasibility
            # MF: alpha = (t_mnf^T Λ⁻¹ x_mnf) / (t_mnf^T Λ⁻¹ t_mnf)
            # Infeasibility: r = x_mnf - alpha t_mnf;
            #                d = sqrt(λ)(1-alpha) + alpha;
            #                I = ||r / d||_2
            # -----------------------------------------------------------------
            t_mnf = (target_arr.astype(np.float64) - input_data_mean) @ T.T  # (B,)
            inv_lambda = 1.0 / np.where(lambda_vals == 0.0, np.finfo(np.float64).eps, lambda_vals)
            mnf_flat = mnf_manual.reshape(-1, mnf_manual.shape[-1])  # (N, B)
            numerator = mnf_flat @ (inv_lambda * t_mnf)  # (N,)
            denominator = float(np.sum(t_mnf * inv_lambda * t_mnf))
            denominator = denominator if denominator != 0.0 else np.finfo(np.float64).eps
            alpha_flat = numerator / denominator  # (N,)
            our_mf_manual = alpha_flat.reshape(h_in, w_in)

            sqrt_lambda = np.sqrt(np.maximum(lambda_vals, 0.0))
            r = mnf_flat - alpha_flat[:, np.newaxis] * t_mnf[np.newaxis, :]  # (N, B)
            d = sqrt_lambda[np.newaxis, :] * (1.0 - alpha_flat[:, np.newaxis]) + alpha_flat[:, np.newaxis]
            d = np.where(d == 0.0, np.finfo(np.float64).eps, d)
            mt = r / d
            our_infeas_manual = np.sqrt(np.sum(mt**2, axis=1)).reshape(h_in, w_in)

            # -----------------------------------------------------------------
            # Compare manual MF / infeasibility against ENVI
            # -----------------------------------------------------------------
            envi_infeas = image_data[1].astype(np.float64)
            envi_mf_f64 = image_data[0].astype(np.float64)

            mf_man_diff = (our_mf_manual.astype(np.float64) - envi_mf_f64).ravel()
            print(f"\n Manual MF vs ENVI MF difference statistics (n={mf_man_diff.size}):")
            print(f"   min  = {mf_man_diff.min():.6f}")
            print(f"   max  = {mf_man_diff.max():.6f}")
            print(f"   mean = {mf_man_diff.mean():.6f}")
            print(f"   std  = {mf_man_diff.std():.6f}")

            infeas_man_diff = (our_infeas_manual.astype(np.float64) - envi_infeas).ravel()
            print(
                "\n Manual infeasibility vs ENVI infeasibility difference statistics "
                f"(n={infeas_man_diff.size}):"
            )
            print(f"   min  = {infeas_man_diff.min():.6f}")
            print(f"   max  = {infeas_man_diff.max():.6f}")
            print(f"   mean = {infeas_man_diff.mean():.6f}")
            print(f"   std  = {infeas_man_diff.std():.6f}")

            del mnf_flat
            gc.collect()

            # -----------------------------------------------------------------
            # Compare our MNF data cube with the ENVI MNF reference
            # -----------------------------------------------------------------

            envi_mnf_ds = self.test_model.load_dataset(
                str(Path(r"<path to intermediate MNF result from ENVI (.hdr)"))
            )
            envi_mnf_source_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=envi_mnf_ds)
            )
            envi_mnf_data_raw, _ = storage_client.read_data(envi_mnf_source_ref, filter_data=False)
            # read_data returns [y][x][b] for dataset refs — no axis reordering needed.
            envi_mnf = np.asarray(np.ma.getdata(envi_mnf_data_raw), dtype=np.float64)  # (H, W, B)

            our_mnf_raw, _ = storage_client.read_data(
                task_plan.bindings[mnf_data_ref_name], filter_data=False
            )
            our_mnf = np.asarray(np.ma.getdata(our_mnf_raw), dtype=np.float64)  # (H, W, B)

            compare_datasets(our_mnf, envi_mnf, label_a="pipeline MNF", label_b="ENVI MNF")
            compare_datasets(mnf_manual, envi_mnf, label_a="manual MNF", label_b="ENVI MNF")

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
            scores, storage_client, _ = self._run_mnf_mtmf_pipeline(
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
    #         scores, storage_client, _ = self._run_mnf_mtmf_pipeline(
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
    #         scores, storage_client, _ = self._run_mnf_mtmf_pipeline(
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
