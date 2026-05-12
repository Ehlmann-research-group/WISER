"""Matched target / matched filter (MTMF) task stage for hyperspectral cubes."""

import datetime
from enum import IntEnum
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide2.QtCore import QObject, Qt, Signal, Slot
from PySide2.QtWidgets import QDialog, QMessageBox

from wiser.gui.app_services import AppServices
from wiser.gui.app_state import ApplicationState
from wiser.gui.generated.mtmf_dialog_ui import Ui_MTMF_Dialog
from wiser.gui.mnf import get_mnf_pipeline, ShiftDiffNoiseDirection
from wiser.utils.primitives import (
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    ExternalRasterHandle,
    NoChunkingScheme,
    PriorityClass,
    SpectraListPlanMeta,
    SpatialTileScheme,
)
from wiser.utils.task_system import (
    AlgorithmPipeline,
    BasePlanMeta,
    DatasetPlanMeta,
    MapStage,
    ResourceModel,
    SemanticTask,
    SequentialStage,
    WriteSpec,
)
from wiser.utils.task_stage_utils import (
    CalcCovMatrixStage,
    DiagonalMatrixFromValuesStage,
    get_good_band_runs,
    get_spectral_mean_stage,
    MatrixMultiplicationStage,
    PosSemiDefMatrixInverse,
    split_dataset_tile_by_good_band_runs,
)
from wiser.utils.worker_runtime import get_process_storage_client

from wiser.raster.dataset import RasterDataSet
from wiser.raster.spectrum import Spectrum


def _run_transform_targets_to_mnf(
    input_ref: DataRef,
    input_region: DataRegion,
    output_ref: DataRef,
    target_spectra_ref: DataRef,
    input_mean_ref: DataRef,
    good_band_runs: tuple,
) -> None:
    """Worker: transform target spectra into MNF space.

    Computes t_mnf = T_mnf x (t - μ_b) for each target, stripping bad bands
    from the full-band target spectra before applying the transform.
    """
    _ = input_region
    client = get_process_storage_client()

    transform_raw, _ = client.read_data(input_ref)
    transform = np.asarray(np.ma.getdata(transform_raw), dtype=np.float64)
    if transform.ndim == 3:
        transform = np.squeeze(transform, axis=2)

    mean_raw, _ = client.read_data(input_mean_ref)
    mean_arr = np.asarray(np.ma.getdata(mean_raw), dtype=np.float64).ravel()

    targets_raw, _ = client.read_data(target_spectra_ref)
    targets_full = np.asarray(np.ma.getdata(targets_raw), dtype=np.float64)
    if targets_full.ndim == 1:
        targets_full = targets_full[np.newaxis, :]

    targets = np.concatenate([targets_full[:, start:end] for start, end in good_band_runs], axis=1)

    targets_centered = targets - mean_arr
    t_mnf = (transform @ targets_centered.T).T

    client.write_data(output_ref, t_mnf.astype(np.float32, copy=False))


@dataclass
class TransformTargetsToMNFStage(SequentialStage):
    """Transform target spectra from full-band space into MNF space.

    Computes ``t_mnf = T_mnf x (t - μ_b)`` for each target spectrum.
    Bad bands are stripped from the full-band target spectra before the
    transform is applied.  The primary input is the T_mnf matrix.

    Output shape: ``(N_targets, num_features)``.
    """

    _output_ref_name: str = "t_mnf"
    _target_spectra_ref_name: str = "target_spectra"
    _input_mean_ref_name: str = "input_mean"
    _good_band_runs: tuple = field(default_factory=tuple)
    _n_targets: int = 1
    _num_features: int = 0
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = NoChunkingScheme

    def __post_init__(self) -> None:
        for name in (self._target_spectra_ref_name, self._input_mean_ref_name):
            if name not in self.broadcast_input:
                raise ValueError(f"TransformTargetsToMNFStage requires '{name}' in broadcast_input")
        self.output_bindings = list(self.output_bindings) + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        return None

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> list[AllocationRequest]:
        _ = (input_meta, chosen_scheme)
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=self._n_targets * self._num_features * np.dtype(np.float32).itemsize,
                shape=(self._n_targets, self._num_features),
                dtype=np.dtype(np.float32),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            )
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        output_write = output_writes[self._output_ref_name]
        target_spectra_ref = broadcast_inputs[self._target_spectra_ref_name]
        input_mean_ref = broadcast_inputs[self._input_mean_ref_name]
        return partial(
            _run_transform_targets_to_mnf,
            input_ref,
            input_region,
            output_write.ref,
            target_spectra_ref,
            input_mean_ref,
            self._good_band_runs,
        )


def _run_mnf_mahalanobis_tile(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    inv_cov_ref: DataRef,
    t_mnf_ref: DataRef,
    normalize: bool,
) -> None:
    """Worker: compute Mahalanobis matched-filter scores for one spatial tile.

    For each pixel x_mnf in the tile computes:
        score = (t_mnf^T Λ⁻¹ x_mnf) [/ (t_mnf^T Λ⁻¹ t_mnf) when normalize=True]

    The data is already in MNF space and mean-centered, so no further
    centering or bad-band stripping is needed.
    """
    client = get_process_storage_client()

    data_tile, region_meta = client.read_region(input_ref, input_region, filter_data=False)
    data_arr = np.asarray(np.ma.getdata(data_tile), dtype=np.float64)  # (h, w, num_features)
    chunk_h, chunk_w, num_features = data_arr.shape

    inv_cov_raw, _ = client.read_data(inv_cov_ref)
    inv_cov = np.asarray(np.ma.getdata(inv_cov_raw), dtype=np.float64)
    if inv_cov.ndim == 3:
        inv_cov = np.squeeze(inv_cov, axis=2)
    t_mnf_raw, _ = client.read_data(t_mnf_ref)
    t_mnf = np.asarray(np.ma.getdata(t_mnf_raw), dtype=np.float64)
    if t_mnf.ndim == 1:
        t_mnf = t_mnf[np.newaxis, :]
    n_targets = t_mnf.shape[0]

    flat = data_arr.reshape(chunk_h * chunk_w, num_features)  # (n_pixels, num_features)

    nodata = region_meta.nodata
    if nodata is not None:
        if np.isnan(nodata):
            nodata_mask = np.any(np.isnan(flat), axis=1)
        else:
            nodata_mask = np.any(flat == nodata, axis=1)
        valid_indices = np.where(~nodata_mask)[0]
    else:
        valid_indices = np.arange(chunk_h * chunk_w)

    flat_valid = flat[valid_indices]  # (n_valid, num_features)

    # Λ⁻¹ x t_mnf^T → (num_features, N_targets)
    inv_cov_t = inv_cov @ t_mnf.T
    # inv_cov_t = t_mnf.T

    # Numerator: x_mnf @ (Λ⁻¹ t_mnf^T) → (n_valid, N_targets)
    numerators = flat_valid @ inv_cov_t

    if normalize:
        # Denominator: t_mnf^T Λ⁻¹ t_mnf — one scalar per target (diagonal of t_mnf @ inv_cov_t)
        denominators = np.sum(t_mnf * inv_cov_t.T, axis=1)  # (N_targets,)
        denominators = np.where(denominators == 0.0, np.finfo(np.float64).eps, denominators)
        scores_valid = numerators / denominators
    else:
        scores_valid = numerators

    scores_flat = np.full((chunk_h * chunk_w, n_targets), fill_value=np.nan, dtype=np.float32)
    scores_flat[valid_indices] = scores_valid.astype(np.float32)
    client.write_spec(output_write, scores_flat.reshape(chunk_h, chunk_w, n_targets))


@dataclass
class MNFMatchedFilterStage(MapStage):
    """Compute matched-filter scores per pixel in MNF space.

    Implements the (optionally normalized) Mahalanobis inner product:

    .. code-block:: text

        score = (t_mnf^T Λ⁻¹ x_mnf) [/ (t_mnf^T Λ⁻¹ t_mnf) when normalize=True]

    With ``normalize=True`` (the default) this is the standard matched-filter
    abundance estimate a.  The primary input is the MNF-transformed data cube
    (H, W, num_features); the data must already be mean-centered.

    Output shape: ``(H, W, N_targets)`` float32.
    """

    _output_ref_name: str = "mnf_matched_filter"
    _inv_cov_ref_name: str = "inv_cov"
    _t_mnf_ref_name: str = "t_mnf"
    _n_targets: int = 1
    _normalize: bool = True
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=3,
            bytes_per_scalar_out=0,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

    def __post_init__(self) -> None:
        for name in (self._inv_cov_ref_name, self._t_mnf_ref_name):
            if name not in self.broadcast_input:
                raise ValueError(f"MNFMatchedFilterStage requires '{name}' in broadcast_input")
        self.output_bindings = list(self.output_bindings) + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        if not isinstance(input_region, DatasetRegionRef):
            raise TypeError("MNFMatchedFilterStage expects a DatasetRegionRef input region")
        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1,
            x0=input_region.x0,
            x1=input_region.x1,
            b0=0,
            b1=self._n_targets,
        )

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> List[AllocationRequest]:
        _ = chosen_scheme
        assert isinstance(input_meta, DatasetPlanMeta), "MNFMatchedFilterStage requires DatasetPlanMeta"
        size_est = input_meta.height * input_meta.width * self._n_targets * np.dtype(np.float32).itemsize
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="dataset",
                residency="ram_cacheable",
                size_est=size_est,
                shape=(input_meta.height, input_meta.width, self._n_targets),
                dtype=np.dtype(np.float32),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            )
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        output_write = output_writes[self._output_ref_name]
        inv_cov_ref = broadcast_inputs[self._inv_cov_ref_name]
        t_mnf_ref = broadcast_inputs[self._t_mnf_ref_name]
        return partial(
            _run_mnf_mahalanobis_tile,
            input_ref,
            input_region,
            output_write,
            inv_cov_ref,
            t_mnf_ref,
            self._normalize,
        )


def _run_infeasibility_tile(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    lambda_vals_ref: DataRef,
    t_mnf_ref: DataRef,
    mf_scores_ref: DataRef,
    n_targets: int,
) -> None:
    """Worker: compute MTMF infeasibility scores for one spatial tile.

    Replicates the mixture-tuning step from the reference MATLAB implementation:

        adm   = alpha x t_mnf                         (target contribution per pixel)
        r     = x_mnf - adm                           (residual)
        d_k   = sqrt(lambda_k) x (1 - alpha) + alpha  (adaptive per-feature denominator)
        MT_k  = r_k / d_k                             (mixture-tuned residual)
        I     = ||MT||_2                              (infeasibility score)

    The denominator blends between sqrt(lambda_k) at alpha=0 (background pixel,
    tight threshold governed by background variance) and 1 at alpha=1 (pure
    target pixel, raw residual length), so the feasibility boundary scales with
    the expected variance of a mixture at that abundance level.
    """
    client = get_process_storage_client()

    # Primary: MNF data tile (h, w, num_features)
    data_tile, region_meta = client.read_region(input_ref, input_region, filter_data=False)
    data_arr = np.asarray(np.ma.getdata(data_tile), dtype=np.float64)
    chunk_h, chunk_w, num_features = data_arr.shape

    # Read the matched-filter scores for the same spatial region
    assert isinstance(input_region, DatasetRegionRef)
    mf_region = DatasetRegionRef(
        y0=input_region.y0,
        y1=input_region.y1,
        x0=input_region.x0,
        x1=input_region.x1,
        b0=0,
        b1=n_targets,
    )
    mf_tile, _ = client.read_region(mf_scores_ref, mf_region, filter_data=False)
    alpha = np.asarray(np.ma.getdata(mf_tile), dtype=np.float64)  # (h, w, N_targets)

    lambda_raw, _ = client.read_data(lambda_vals_ref)
    lambda_vals = np.asarray(np.ma.getdata(lambda_raw), dtype=np.float64).ravel()  # (num_features,)
    sqrt_lambda = np.sqrt(np.maximum(lambda_vals, 0.0))  # guard against tiny negatives from numerics

    t_mnf_raw, _ = client.read_data(t_mnf_ref)
    t_mnf = np.asarray(np.ma.getdata(t_mnf_raw), dtype=np.float64)  # (N_targets, num_features)
    if t_mnf.ndim == 1:
        t_mnf = t_mnf[np.newaxis, :]

    flat = data_arr.reshape(chunk_h * chunk_w, num_features)  # (n_pixels, num_features)
    alpha_flat = alpha.reshape(chunk_h * chunk_w, n_targets)  # (n_pixels, N_targets)

    nodata = region_meta.nodata
    if nodata is not None:
        if np.isnan(nodata):
            nodata_mask = np.any(np.isnan(flat), axis=1)
        else:
            nodata_mask = np.any(flat == nodata, axis=1)
        valid_indices = np.where(~nodata_mask)[0]
    else:
        valid_indices = np.arange(chunk_h * chunk_w)

    flat_valid = flat[valid_indices]  # (n_valid, num_features)
    alpha_valid = alpha_flat[valid_indices]  # (n_valid, N_targets)

    # Residual r[n_valid, N_targets, num_features]: r_ijk = x_ik - alpha_ij * t_jk
    r = flat_valid[:, np.newaxis, :] - alpha_valid[:, :, np.newaxis] * t_mnf[np.newaxis, :, :]

    # Adaptive denominator d[n_valid, N_targets, num_features]:
    #   d_ijk = sqrt(lambda_k) * (1 - alpha_ij) + alpha_ij
    d = (
        sqrt_lambda[np.newaxis, np.newaxis, :] * (1.0 - alpha_valid[:, :, np.newaxis])
        + alpha_valid[:, :, np.newaxis]
    )
    d = np.where(d == 0.0, np.finfo(np.float64).eps, d)

    # Mixture-tuned residual; infeasibility = L2 norm over features
    MT = r / d  # (n_valid, N_targets, num_features)
    infeasibility_valid = np.sqrt(np.sum(MT**2, axis=2))  # (n_valid, N_targets)

    infeasibility_flat = np.full((chunk_h * chunk_w, n_targets), fill_value=np.nan, dtype=np.float32)
    infeasibility_flat[valid_indices] = infeasibility_valid.astype(np.float32)
    client.write_spec(output_write, infeasibility_flat.reshape(chunk_h, chunk_w, n_targets))


@dataclass
class MNFInfeasibilityStage(MapStage):
    """Compute MTMF infeasibility scores per pixel in MNF space.

    For each pixel ``x_mnf`` and its matched-filter abundance ``alpha`` computes
    the mixture-tuned residual and its L2 norm:

    .. code-block:: text

        r_k   = x_mnf_k - alpha * t_mnf_k
        d_k   = sqrt(lambda_k) * (1 - alpha) + alpha
        MT_k  = r_k / d_k
        I     = ||MT||_2

    The adaptive denominator ``d_k`` scales from ``sqrt(lambda_k)`` at alpha=0
    (background pixel) down to 1 at alpha=1 (pure target pixel), so the
    feasibility boundary tracks the expected variance of a mixture at that
    abundance level.

    The primary input is the MNF-transformed data cube (H, W, num_features).
    The matched-filter scores ``(H, W, N_targets)`` are read by spatial region
    from a broadcast ``DataRef`` so the two cubes stay tile-synchronized.

    Output shape: ``(H, W, N_targets)`` float32.
    """

    _output_ref_name: str = "mnf_infeasibility"
    _lambda_vals_ref_name: str = "lambda_vals"
    _t_mnf_ref_name: str = "t_mnf"
    _mf_scores_ref_name: str = "mf_scores"
    _n_targets: int = 1
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=6,
            bytes_per_scalar_out=0,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

    def __post_init__(self) -> None:
        for name in (self._lambda_vals_ref_name, self._t_mnf_ref_name, self._mf_scores_ref_name):
            if name not in self.broadcast_input:
                raise ValueError(f"MNFInfeasibilityStage requires '{name}' in broadcast_input")
        self.output_bindings = list(self.output_bindings) + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        if not isinstance(input_region, DatasetRegionRef):
            raise TypeError("MNFInfeasibilityStage expects a DatasetRegionRef input region")
        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1,
            x0=input_region.x0,
            x1=input_region.x1,
            b0=0,
            b1=self._n_targets,
        )

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> List[AllocationRequest]:
        _ = chosen_scheme
        assert isinstance(input_meta, DatasetPlanMeta), "MNFInfeasibilityStage requires DatasetPlanMeta"
        size_est = input_meta.height * input_meta.width * self._n_targets * np.dtype(np.float32).itemsize
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="dataset",
                residency="ram_cacheable",
                size_est=size_est,
                shape=(input_meta.height, input_meta.width, self._n_targets),
                dtype=np.dtype(np.float32),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            )
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        output_write = output_writes[self._output_ref_name]
        lambda_vals_ref = broadcast_inputs[self._lambda_vals_ref_name]
        t_mnf_ref = broadcast_inputs[self._t_mnf_ref_name]
        mf_scores_ref = broadcast_inputs[self._mf_scores_ref_name]
        return partial(
            _run_infeasibility_tile,
            input_ref,
            input_region,
            output_write,
            lambda_vals_ref,
            t_mnf_ref,
            mf_scores_ref,
            self._n_targets,
        )


def get_mnf_mtmf_pipeline(
    dataset_ref: DataRef,
    target_spectra_ref: DataRef,
    output_ref_name: str,
    mnf_noise_ref_name: str = "mtmf_mnf_shift_y_noise",
    mnf_noise_eigen_ref_name: str = "mtmf_mnf_noise_eigen",
    mnf_noise_whitening_matrix_ref_name: str = "mtmf_mnf_noise_whitening_matrix",
    mnf_input_mean_ref_name: str = "mtmf_mnf_input_spectral_mean",
    mnf_input_total_ref_name: str = "mtmf_mnf_input_valid_pixel_total",
    mnf_input_covariance_ref_name: str = "mtmf_mnf_input_covariance",
    mnf_whitened_covariance_ref_name: str = "mtmf_mnf_whitened_covariance",
    mnf_whitened_eigen_ref_name: str = "mtmf_mnf_whitened_eigen",
    mnf_data_ref_name: str = "mtmf_mnf_data",
    data_variance_factor: float = 2,
    shift_diff_noise_direction: ShiftDiffNoiseDirection = ShiftDiffNoiseDirection.DOWN,
) -> AlgorithmPipeline:
    """Build the MNF-based MTMF AlgorithmPipeline.

    Runs a full MNF transform (all components) on the dataset, then uses the
    resulting MNF-space data, transformation matrix, and eigenvalue-diagonal
    covariance to run the matched filter and infeasibility stages.

    Args:
        dataset_ref:                         DataRef for the input hyperspectral data cube (H, W, B).
        target_spectra_ref:                  DataRef for the target spectra (N_targets, B) or (B,).
        output_ref_name:                     Name for the final output allocation.
        mnf_noise_ref_name:                  Ref name for the shift-Y noise cube.
        mnf_noise_eigen_ref_name:            Ref name for the noise eigen-decomposition.
        mnf_noise_whitening_matrix_ref_name: Ref name for the noise whitening matrix.
        mnf_input_mean_ref_name:             Ref name for the input spectral mean vector.
        mnf_input_total_ref_name:            Ref name for the valid-pixel count accumulator.
        mnf_input_covariance_ref_name:       Ref name for the input covariance matrix.
        mnf_whitened_covariance_ref_name:    Ref name for the whitened covariance matrix.
        mnf_whitened_eigen_ref_name:         Ref name for the whitened eigen-decomposition.
        mnf_data_ref_name:                   Ref name for the MNF-transformed data cube output.
        shift_diff_noise_direction:          Spatial direction used by the internal shift-difference
                                             noise stage.  Defaults to ``ShiftDiffNoiseDirection.DOWN``.

    Returns:
        An AlgorithmPipeline containing the MNF stages followed by the
        MTMF-specific stages.
    """

    storage_client = get_process_storage_client()
    data_meta = storage_client.get_meta(dataset_ref)
    if data_meta.bad_bands is not None:
        num_features = int(np.sum(data_meta.bad_bands))
        good_band_runs = tuple(get_good_band_runs(np.asarray(data_meta.bad_bands)))
    else:
        num_features = int(data_meta.shape[2])
        good_band_runs = ((0, num_features),)

    target_meta = storage_client.get_meta(target_spectra_ref)
    if len(target_meta.shape) == 1:
        n_targets = 1
    elif len(target_meta.shape) == 2:
        n_targets = int(target_meta.shape[0])
    else:
        raise ValueError(f"Target spectra must be 1-D (B,) or 2-D (N_targets, B), got {target_meta.shape}")

    mnf_pipeline = get_mnf_pipeline(
        dataset_ref,
        num_components=None,
        output_ref_name=mnf_data_ref_name,
        noise_ref_name=mnf_noise_ref_name,
        noise_eigen_ref_name=mnf_noise_eigen_ref_name,
        noise_whitening_matrix_ref_name=mnf_noise_whitening_matrix_ref_name,
        input_mean_ref_name=mnf_input_mean_ref_name,
        input_total_ref_name=mnf_input_total_ref_name,
        input_covariance_ref_name=mnf_input_covariance_ref_name,
        whitened_covariance_ref_name=mnf_whitened_covariance_ref_name,
        whitened_eigen_ref_name=mnf_whitened_eigen_ref_name,
        data_variance_factor=data_variance_factor,
        shift_diff_noise_direction=shift_diff_noise_direction,
    )

    # Step 3 — build full MNF transformation matrix T = A x W
    # A = whitened eigenvectors (num_features x num_features)
    # W = noise whitening matrix (num_features x num_features)
    # T = A x W maps a mean-centered input spectrum directly into MNF space
    mnf_transform_matrix_ref_name = "mtmf_mnf_transform_matrix"
    mnf_eigenvectors_ref_name = f"{mnf_whitened_eigen_ref_name}_vectors"
    transform_matrix_stage = MatrixMultiplicationStage(
        _output_ref_name=mnf_transform_matrix_ref_name,
        _matrix_input_names=(mnf_eigenvectors_ref_name, mnf_noise_whitening_matrix_ref_name),
        _output_shape=(num_features, num_features),
        _output_dtype=np.dtype(np.float32),
        default_executor="process",
        input_binding=DataBinding(mnf_eigenvectors_ref_name),
        input_plan_meta=SpectraListPlanMeta(
            num_spectra=num_features,
            spectrum_length=num_features,
            dtype=np.dtype(np.float32),
        ),
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        broadcast_input={
            mnf_eigenvectors_ref_name: DataBinding(mnf_eigenvectors_ref_name),
            mnf_noise_whitening_matrix_ref_name: DataBinding(mnf_noise_whitening_matrix_ref_name),
        },
    )

    # Step 4 — construct Λ (diagonal covariance of MNF data) from eigenvalues
    # The covariance of MNF-transformed data equals the diagonal matrix of
    # the whitened eigenvalues, i.e. Λ = diag(λ₁, λ₂, ..., λ_N)
    mnf_eigenvalues_ref_name = f"{mnf_whitened_eigen_ref_name}_values"
    mnf_lambda_ref_name = "mtmf_mnf_lambda"
    lambda_stage = DiagonalMatrixFromValuesStage(
        _output_ref_name=mnf_lambda_ref_name,
        _n=num_features,
        default_executor="process",
        input_binding=DataBinding(mnf_eigenvalues_ref_name),
        input_plan_meta=SpectraListPlanMeta(
            num_spectra=1,
            spectrum_length=num_features,
            dtype=np.dtype(np.float32),
        ),
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
    )

    # Stage 1 — transform target spectra into MNF space
    # t_mnf = T_mnf x (t - μ_b), with bad bands stripped from t first
    target_mnf_ref_name = "mtmf_target_mnf"
    _target_spectra_bc = "mtmf_bc_target_spectra"
    _input_mean_bc = "mtmf_bc_input_mean"
    transform_targets_stage = TransformTargetsToMNFStage(
        _output_ref_name=target_mnf_ref_name,
        _target_spectra_ref_name=_target_spectra_bc,
        _input_mean_ref_name=_input_mean_bc,
        _good_band_runs=good_band_runs,
        _n_targets=n_targets,
        _num_features=num_features,
        default_executor="process",
        input_binding=DataBinding(mnf_transform_matrix_ref_name),
        input_plan_meta=SpectraListPlanMeta(
            num_spectra=num_features,
            spectrum_length=num_features,
            dtype=np.dtype(np.float32),
        ),
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        broadcast_input={
            _target_spectra_bc: target_spectra_ref,
            _input_mean_bc: DataBinding(mnf_input_mean_ref_name),
        },
    )

    # Step 2 — invert Λ to get Λ⁻¹ for the matched filter
    inv_lambda_ref_name = "mtmf_inv_lambda"
    inv_lambda_stage = PosSemiDefMatrixInverse(
        _output_ref_name=inv_lambda_ref_name,
        default_executor="process",
        input_binding=DataBinding(mnf_lambda_ref_name),
        input_plan_meta=SpectraListPlanMeta(
            num_spectra=num_features,
            spectrum_length=num_features,
            dtype=np.dtype(np.float32),
        ),
        resource_model=ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=NoChunkingScheme,
    )

    # Step 3 — matched filter: score = (t_mnf^T Λ⁻¹ x_mnf) / (t_mnf^T Λ⁻¹ t_mnf)
    # Primary input: MNF data cube (H, W, num_features), already mean-centered.
    # Broadcast: Λ⁻¹ and t_mnf.  Output: (H, W, N_targets) float32.
    _inv_lambda_bc = "mtmf_bc_inv_lambda"
    _target_mnf_bc = "mtmf_bc_target_mnf"
    matched_filter_ref_name = output_ref_name
    height = int(data_meta.shape[0])
    width = int(data_meta.shape[1])
    matched_filter_stage = MNFMatchedFilterStage(
        _output_ref_name=matched_filter_ref_name,
        _inv_cov_ref_name=_inv_lambda_bc,
        _t_mnf_ref_name=_target_mnf_bc,
        _n_targets=n_targets,
        _normalize=True,
        default_executor="process",
        input_binding=DataBinding(mnf_data_ref_name),
        input_plan_meta=DatasetPlanMeta(
            shape=(height, width, num_features),
            dtype=np.dtype(np.float32),
        ),
        broadcast_input={
            _inv_lambda_bc: DataBinding(inv_lambda_ref_name),
            _target_mnf_bc: DataBinding(target_mnf_ref_name),
        },
    )

    # Step 4 — infeasibility via mixture-tuning residual norm:
    #   d_k = sqrt(lambda_k) * (1 - alpha) + alpha
    #   I   = ||( x_mnf - alpha * t_mnf ) / d||_2
    # Uses the raw eigenvalue vector (diagonal of Λ), not Λ⁻¹.
    # Matched-filter scores are read by spatial region inside the worker.
    infeasibility_ref_name = f"{output_ref_name}_infeasibility"
    _lambda_vals_bc = "mtmf_bc_lambda_vals"
    _mf_scores_bc = "mtmf_bc_mf_scores"
    infeasibility_stage = MNFInfeasibilityStage(
        _output_ref_name=infeasibility_ref_name,
        _lambda_vals_ref_name=_lambda_vals_bc,
        _t_mnf_ref_name=_target_mnf_bc,
        _mf_scores_ref_name=_mf_scores_bc,
        _n_targets=n_targets,
        default_executor="process",
        input_binding=DataBinding(mnf_data_ref_name),
        input_plan_meta=DatasetPlanMeta(
            shape=(height, width, num_features),
            dtype=np.dtype(np.float32),
        ),
        broadcast_input={
            _lambda_vals_bc: DataBinding(mnf_eigenvalues_ref_name),
            _target_mnf_bc: DataBinding(target_mnf_ref_name),
            _mf_scores_bc: DataBinding(matched_filter_ref_name),
        },
    )

    return AlgorithmPipeline(
        mnf_pipeline.stages
        + [
            transform_matrix_stage,
            lambda_stage,
            transform_targets_stage,
            inv_lambda_stage,
            matched_filter_stage,
            infeasibility_stage,
        ]
    )


# ---------------------------------------------------------------------------
# Matched Filter Stage
# ---------------------------------------------------------------------------
# Implements:
#   T(x) = [(t - μ)^T Γ⁻¹ (x - μ)] / [(t - μ)^T Γ⁻¹ (t - μ)]
#
# where:
#   x      – observed pixel spectrum  (B,)
#   t      – target reference spectrum (B,)
#   μ      – noise mean               (B,)
#   Γ⁻¹   – inverse noise covariance  (B, B)
#
# For N_targets reference spectra the output per spatial chunk is
# shaped (chunk_h, chunk_w, N_targets); each band-slice is the
# matched-filter score against one target.
# ---------------------------------------------------------------------------


def _run_matched_filter_tile(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    inv_noise_cov_ref: DataRef,
    noise_mean_ref: DataRef,
    target_spectra_ref: DataRef,
) -> None:
    """Worker: compute the matched-filter score for one spatial tile."""

    client = get_process_storage_client()

    data_tile, region_meta = client.read_region(input_ref, input_region, filter_data=False)
    data_arr = np.asarray(np.ma.getdata(data_tile), dtype=np.float64)  # (h, w, B_total)

    if data_arr.ndim != 3:
        raise ValueError(f"Expected dataset tile shape [y][x][b], got {data_arr.shape}")

    chunk_h, chunk_w, b_total = data_arr.shape

    # ------------------------------------------------------------------
    # Strip bad bands from the data cube
    # ------------------------------------------------------------------
    if region_meta.bad_bands is None:
        good_band_runs = [(0, b_total)]
    else:
        good_band_runs = get_good_band_runs(np.asarray(region_meta.bad_bands))

    if len(good_band_runs) == 0:
        raise ValueError("Matched filter requires at least one valid band; all bands are flagged as bad.")

    good_chunks = split_dataset_tile_by_good_band_runs(data_arr, good_band_runs)
    data_good = np.concatenate(good_chunks, axis=2)  # (h, w, b_good)
    b_good = data_good.shape[2]

    # ------------------------------------------------------------------
    # Read broadcast inputs and restrict them to good bands
    # ------------------------------------------------------------------
    noise_mean_raw, _ = client.read_data(noise_mean_ref)
    noise_mean_full = np.asarray(np.ma.getdata(noise_mean_raw), dtype=np.float64).ravel()  # (B_total,)
    noise_mean = np.concatenate([noise_mean_full[start:end] for start, end in good_band_runs])  # (b_good,)

    inv_cov_raw, _ = client.read_data(inv_noise_cov_ref)
    inv_cov_full = np.asarray(np.ma.getdata(inv_cov_raw), dtype=np.float64)  # (B_total, B_total)
    if inv_cov_full.ndim == 3:
        inv_cov_full = np.squeeze(inv_cov_full, axis=2)
    if inv_cov_full.ndim != 2:
        raise ValueError(f"Inverse noise covariance must be 2-D, got shape {inv_cov_full.shape}")
    good_indices = np.concatenate([np.arange(s, e) for s, e in good_band_runs])
    inv_cov = inv_cov_full[np.ix_(good_indices, good_indices)]  # (b_good, b_good)

    targets_raw, _ = client.read_data(target_spectra_ref)
    targets_full = np.asarray(np.ma.getdata(targets_raw), dtype=np.float64)  # (N_targets, B_total)
    if targets_full.ndim == 1:
        targets_full = targets_full[np.newaxis, :]
    if targets_full.ndim != 2:
        raise ValueError(f"Target spectra must be 1-D or 2-D, got shape {targets_full.shape}")
    targets = np.concatenate(
        [targets_full[:, start:end] for start, end in good_band_runs], axis=1
    )  # (N_targets, b_good)

    # ------------------------------------------------------------------
    # Flatten to pixels and remove nodata rows
    # ------------------------------------------------------------------
    flat = data_good.reshape(chunk_h * chunk_w, b_good)  # (n_pixels, b_good)

    nodata = region_meta.nodata
    if nodata is not None:
        if np.isnan(nodata):
            nodata_mask = np.any(np.isnan(flat), axis=1)
        else:
            nodata_mask = np.any(flat == nodata, axis=1)
        valid_indices = np.where(~nodata_mask)[0]
    else:
        valid_indices = np.arange(chunk_h * chunk_w)

    flat_valid = flat[valid_indices]  # (n_valid, b_good)

    # ------------------------------------------------------------------
    # Matched-filter computation on valid pixels only
    #   T(x) = [(t-μ)ᵀ Γ⁻¹ (x-μ)] / [(t-μ)ᵀ Γ⁻¹ (t-μ)]
    # ------------------------------------------------------------------
    x_centered = flat_valid - noise_mean  # (n_valid, b_good)
    t_centered = targets - noise_mean  # (N_targets, b_good)

    # t_Sigma_inv[i] = t_centered[i] @ Γ⁻¹  →  (N_targets, b_good)
    t_Sigma_inv = t_centered @ inv_cov

    # Numerator: x_centered @ t_Sigma_inv.T  →  (n_valid, N_targets)
    numerators = x_centered @ t_Sigma_inv.T

    # Denominator: scalar per target
    denominators = (t_centered * t_Sigma_inv).sum(axis=1)  # (N_targets,)
    denominators = np.where(denominators == 0.0, np.finfo(np.float64).eps, denominators)

    scores_valid = numerators / denominators  # (n_valid, N_targets)

    # ------------------------------------------------------------------
    # Scatter scores back to full pixel grid; nodata pixels get NaN
    # ------------------------------------------------------------------
    n_targets = targets.shape[0]
    scores_flat = np.full((chunk_h * chunk_w, n_targets), fill_value=np.nan, dtype=np.float32)
    scores_flat[valid_indices] = scores_valid.astype(np.float32)
    scores = scores_flat.reshape(chunk_h, chunk_w, n_targets)

    client.write_spec(output_write, scores)


def _validate_matched_filter_inputs(
    input_ref: DataRef,
    inv_noise_cov_ref: DataRef,
    noise_mean_ref: DataRef,
    target_spectra_ref: DataRef,
) -> None:
    """Pre-task validation: confirm that all inputs are dimensionally consistent.

    Runs once before any spatial tile work units execute.

    Args:
        input_ref: DataRef for the input hyperspectral data cube (H, W, B_total).
        inv_noise_cov_ref: DataRef for the inverse noise covariance matrix.
        noise_mean_ref: DataRef for the noise mean vector.
        target_spectra_ref: DataRef for the target spectra array.

    Raises:
        ValueError: If any dimensional mismatch is detected.
    """
    client = get_process_storage_client()

    dataset_meta = client.get_meta(input_ref)
    b_total = dataset_meta.shape[2]

    # Determine the number of good bands after bad-band removal
    if dataset_meta.bad_bands is None:
        good_band_runs = [(0, b_total)]
    else:
        good_band_runs = get_good_band_runs(np.asarray(dataset_meta.bad_bands))
    b_good = sum(end - start for start, end in good_band_runs)

    if b_good == 0:
        raise ValueError("Matched filter requires at least one valid band; all bands are flagged as bad.")

    # Validate noise mean: must have length == b_good
    mean_meta = client.get_meta(noise_mean_ref)
    mean_len = mean_meta.shape[0] if len(mean_meta.shape) == 1 else int(np.prod(mean_meta.shape))
    if mean_len != b_good:
        raise ValueError(
            f"Noise mean length ({mean_len}) does not match the number of good bands "
            f"({b_good}) after bad-band removal. "
            f"Supply a noise mean computed from the same good-band-filtered data."
        )

    # Validate inverse covariance: must be square with side == b_good
    cov_meta = client.get_meta(inv_noise_cov_ref)
    cov_shape = cov_meta.shape
    if len(cov_shape) < 2 or cov_shape[0] != b_good or cov_shape[1] != b_good:
        raise ValueError(
            f"Inverse noise covariance shape {cov_shape} is not ({b_good}, {b_good}). "
            f"It must be square with side equal to the number of good bands ({b_good}) "
            f"after bad-band removal."
        )

    # Validate target spectra: band dimension must match b_total (full, pre-removal)
    target_meta = client.get_meta(target_spectra_ref)
    target_shape = target_meta.shape
    if len(target_shape) == 1:
        target_bands = target_shape[0]
    elif len(target_shape) == 2:
        target_bands = target_shape[1]
    else:
        raise ValueError(f"Target spectra must be 1-D (B,) or 2-D (N_targets, B), got shape {target_shape}.")
    if target_bands != b_total:
        raise ValueError(
            f"Target spectra band count ({target_bands}) does not match the total number of "
            f"bands in the input data cube ({b_total}). "
            f"Supply full-band target spectra (including bad bands); bad bands are removed "
            f"internally before the matched-filter is applied."
        )


@dataclass
class MatchedFilterStage(MapStage):
    """Task stage that computes matched-filter detection scores for a hyperspectral cube.

    Implements the matched filter:

    .. code-block:: text

        T(x) = [(t - μ)ᵀ Γ⁻¹ (x - μ)] / [(t - μ)ᵀ Γ⁻¹ (t - μ)]

    where ``x`` is an observed pixel spectrum, ``t`` is a target reference
    spectrum, ``μ`` is the noise mean, and ``Γ⁻¹`` is the inverse noise
    covariance matrix.

    The stage uses ``SpatialTileScheme`` to split the cube into spatial tiles
    (all bands, sub-region of rows/columns), processes each tile independently
    in a worker process, and writes results to a single pre-allocated output
    dataset.

    **Band handling:**
        Before any arithmetic, bad bands are stripped from the data cube using
        the ``bad_bands`` metadata attached to the input dataset.  The noise
        mean and inverse covariance matrix must therefore be provided in
        *good-band space* (i.e. already computed on the band-filtered data).
        Target spectra must be provided in *full-band space* (same total band
        count as the raw cube); bad bands are removed from them internally.

    **Nodata handling:**
        Pixels whose value in any good band equals the dataset's ``nodata``
        value (or is NaN when ``nodata`` is NaN) are excluded from the
        matched-filter computation and written as ``NaN`` in the output.

    **Output:**
        A single float32 dataset of shape ``(H, W, N_targets)`` where band
        slice ``i`` is the matched-filter score map for target ``i``.

    Attributes:
        _output_ref_name: Allocation name for the output dataset.
            Defaults to ``"matched_filter_output"``.
        _inv_noise_cov_ref_name: Broadcast-input key for the inverse noise
            covariance matrix (good-band space, shape ``(b_good, b_good)``).
            Defaults to ``"inv_noise_cov"``.
        _noise_mean_ref_name: Broadcast-input key for the noise mean vector
            (good-band space, shape ``(b_good,)``).
            Defaults to ``"noise_mean"``.
        _target_spectra_ref_name: Broadcast-input key for the target spectra
            array (full-band space, shape ``(N_targets, B_total)`` or
            ``(B_total,)`` for a single target).
            Defaults to ``"target_spectra"``.
        _num_targets: Number of target spectra (rows in the target array).
            Resolved automatically by ``get_matched_filter_stage``.

    Note:
        All three broadcast inputs (``inv_noise_cov_ref``, ``noise_mean_ref``,
        ``target_spectra_ref``) must be present in ``broadcast_input`` before
        the stage is constructed; ``__post_init__`` raises ``ValueError`` if
        any are missing.  Use ``get_matched_filter_stage`` or
        ``get_matched_filter_pipeline`` to construct the stage safely.
    """

    _output_ref_name: str = "matched_filter_output"
    _inv_noise_cov_ref_name: str = "inv_noise_cov"
    _noise_mean_ref_name: str = "noise_mean"
    _target_spectra_ref_name: str = "target_spectra"
    _num_targets: int = 1
    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=3,
            bytes_per_scalar_out=0,
            scratch_bytes_per_scalar_in=0,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

    def __post_init__(self) -> None:
        for name in (
            self._inv_noise_cov_ref_name,
            self._noise_mean_ref_name,
            self._target_spectra_ref_name,
        ):
            if name not in self.broadcast_input:
                raise ValueError(f"MatchedFilterStage requires '{name}' in broadcast_input")
        self.output_bindings = list(self.output_bindings) + [DataBinding(self._output_ref_name)]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        if not isinstance(input_region, DatasetRegionRef):
            raise TypeError("MatchedFilterStage expects a DatasetRegionRef input region")
        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1,
            x0=input_region.x0,
            x1=input_region.x1,
            b0=0,
            b1=self._num_targets,
        )

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
    ) -> List[AllocationRequest]:
        _ = chosen_scheme
        assert isinstance(
            input_meta, DatasetPlanMeta
        ), "MatchedFilterStage requires DatasetPlanMeta input_meta"
        size_est = input_meta.height * input_meta.width * self._num_targets * np.dtype(np.float32).itemsize
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="dataset",
                residency="ram_cacheable",
                size_est=size_est,
                shape=(input_meta.height, input_meta.width, self._num_targets),
                dtype=np.dtype(np.float32),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            )
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        output_write = output_writes[self._output_ref_name]
        inv_noise_cov_ref = broadcast_inputs[self._inv_noise_cov_ref_name]
        noise_mean_ref = broadcast_inputs[self._noise_mean_ref_name]
        target_spectra_ref = broadcast_inputs[self._target_spectra_ref_name]
        return partial(
            _run_matched_filter_tile,
            input_ref,
            input_region,
            output_write,
            inv_noise_cov_ref,
            noise_mean_ref,
            target_spectra_ref,
        )

    def pre_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        _ = (full_input_region, output_writes)
        inv_noise_cov_ref = broadcast_inputs[self._inv_noise_cov_ref_name]
        noise_mean_ref = broadcast_inputs[self._noise_mean_ref_name]
        target_spectra_ref = broadcast_inputs[self._target_spectra_ref_name]
        return partial(
            _validate_matched_filter_inputs,
            input_ref,
            inv_noise_cov_ref,
            noise_mean_ref,
            target_spectra_ref,
        )


def get_matched_filter_stage(
    dataset_ref: DataRef,
    inv_noise_cov_ref: DataRef,
    noise_mean_ref: DataRef,
    target_spectra_ref: DataRef,
    output_ref_name: str = "matched_filter_output",
    inv_noise_cov_ref_name: str = "inv_noise_cov",
    noise_mean_ref_name: str = "noise_mean",
    target_spectra_ref_name: str = "target_spectra",
) -> "MatchedFilterStage":
    """
    Build a :class:`MatchedFilterStage` by inspecting the target spectra ref
    to determine the number of targets at planning time.

    Args:
        dataset_ref:           DataRef for the input hyperspectral data cube (H, W, B).
        inv_noise_cov_ref:     DataRef for the (B, B) inverse noise covariance matrix.
        noise_mean_ref:        DataRef for the (B,) noise mean vector.
        target_spectra_ref:    DataRef for the (N_targets, B) target spectra array.
        output_ref_name:       Name for the output allocation (default "matched_filter_output").
        inv_noise_cov_ref_name: Broadcast-input key for inv_noise_cov (default "inv_noise_cov").
        noise_mean_ref_name:   Broadcast-input key for noise_mean (default "noise_mean").
        target_spectra_ref_name: Broadcast-input key for target_spectra (default "target_spectra").

    Returns:
        A fully configured :class:`MatchedFilterStage`.
    """
    client = get_process_storage_client()

    dataset_meta = client.get_meta(dataset_ref)
    if len(dataset_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [H][W][B], got {dataset_meta.shape}")

    targets_meta = client.get_meta(target_spectra_ref)
    if len(targets_meta.shape) == 1:
        num_targets = 1
    elif len(targets_meta.shape) == 2:
        num_targets = targets_meta.shape[0]
    else:
        raise ValueError(f"Target spectra must be 1-D (B,) or 2-D (N_targets, B), got {targets_meta.shape}")

    input_meta = DatasetPlanMeta(shape=dataset_meta.shape, dtype=np.dtype(dataset_meta.elem_type))

    return MatchedFilterStage(
        _output_ref_name=output_ref_name,
        _inv_noise_cov_ref_name=inv_noise_cov_ref_name,
        _noise_mean_ref_name=noise_mean_ref_name,
        _target_spectra_ref_name=target_spectra_ref_name,
        _num_targets=num_targets,
        default_executor="process",
        input_plan_meta=input_meta,
        chunking_scheme_type=SpatialTileScheme,
        broadcast_input={
            inv_noise_cov_ref_name: inv_noise_cov_ref,
            noise_mean_ref_name: noise_mean_ref,
            target_spectra_ref_name: target_spectra_ref,
        },
    )


def get_matched_filter_pipeline(
    dataset_ref: DataRef,
    inv_noise_cov_ref: DataRef,
    noise_mean_ref: DataRef,
    target_spectra_ref: DataRef,
    output_ref_name: str = "matched_filter_output",
) -> AlgorithmPipeline:
    """
    Return a single-stage :class:`AlgorithmPipeline` that runs the matched filter.

    The output dataset has shape (H, W, N_targets) and dtype float32, where
    output[:, :, i] is the matched-filter score map for the i-th target spectrum.
    """
    return AlgorithmPipeline(
        [
            get_matched_filter_stage(
                dataset_ref=dataset_ref,
                inv_noise_cov_ref=inv_noise_cov_ref,
                noise_mean_ref=noise_mean_ref,
                target_spectra_ref=target_spectra_ref,
                output_ref_name=output_ref_name,
            )
        ]
    )


# ---------------------------------------------------------------------------
# Full MTMF pipeline: noise stats → inverse covariance → matched filter
# ---------------------------------------------------------------------------

_MTMF_NOISE_MEAN_NAME = "mtmf_noise_mean"
_MTMF_NOISE_COV_NAME = "mtmf_noise_covariance"
_MTMF_INV_COV_NAME = "mtmf_inv_noise_covariance"
_MTMF_NOISE_PLAN_BINDING = "mtmf_noise_ref"

# ---------------------------------------------------------------------------
# MTMFSemanticTask
# ---------------------------------------------------------------------------


class MTMFSemanticTask(QObject, SemanticTask):
    """Semantic task that runs the full Mixture Tuned Matched Filter pipeline.

    Registers the source and noise datasets, stacks the target spectra into a
    single array, builds the four-stage MTMF pipeline (noise mean → noise
    covariance → pseudoinverse → matched filter), and on completion slices the
    3-D output ``(H, W, N_targets)`` into individual score-map datasets — one
    per target — and loads them into the WISER application state.

    Args:
        app_state: The running :class:`ApplicationState` used to register
            results and obtain unique dataset names.
        app_services: Application services used for dataset registration and
            task allocation.
        source_dataset: The hyperspectral image cube to score.
        noise_dataset: A dataset used to estimate background noise statistics
            (mean and covariance).  Must have the same spectral bands as
            ``source_dataset``.
        target_spectra: One or more target reference spectra.  Each spectrum
            must span the *full* band count of ``source_dataset`` (bad-band
            removal is performed internally).
        output_ref_name: Storage allocation name for the matched-filter output
            cube.  Defaults to ``"matched_filter_output"``.

    Signals:
        result_ready: Emitted in the completion callback with the score array
            ``(H, W, N_targets)`` and the list of target names.  Connected to
            :meth:`_load_result_into_wiser` which runs on the Qt main thread.

    Note:
        The task's ``input_ref`` is the **source** image cube.  The noise
        dataset is supplied through ``extra_plan_bindings`` under
        ``mtmf_noise_ref`` so the first two pipeline stages can read it while
        the matched-filter stage uses ``__task_input__`` for the source.
    """

    result_ready = Signal(object)

    def __init__(
        self,
        app_state: ApplicationState,
        app_services: AppServices,
        source_dataset: RasterDataSet,
        target_spectra: List[Spectrum],
        output_ref_name: str = "matched_filter_output",
        noise_dataset: Optional[RasterDataSet] = None,
        shift_diff_noise_direction: Optional[ShiftDiffNoiseDirection] = None,
    ) -> None:
        QObject.__init__(self)

        if not target_spectra:
            raise ValueError("MTMFSemanticTask requires at least one target spectrum.")

        if noise_dataset is None and shift_diff_noise_direction is None:
            raise ValueError(
                "MTMFSemanticTask requires either noise_dataset (Dark Image Noise) "
                "or shift_diff_noise_direction (Image Cube Noise)."
            )
        if noise_dataset is not None and shift_diff_noise_direction is not None:
            raise ValueError(
                "MTMFSemanticTask: supply noise_dataset or shift_diff_noise_direction, not both."
            )

        source_ref = app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=source_dataset)
        )

        # Stack target spectra into a single (N_targets, B) float32 array and
        # allocate it as a named array DataRef.
        target_names: List[str] = []
        spectra_arrays: List[np.ndarray] = []
        for i, spectrum in enumerate(target_spectra):
            name = getattr(spectrum, "get_name", lambda: None)()
            target_names.append(name if name else f"Target {i + 1}")
            spectra_arrays.append(np.asarray(spectrum.get_spectrum(), dtype=np.float32))

        target_array = np.stack(spectra_arrays, axis=0)  # (N_targets, B)

        process_client = get_process_storage_client()
        target_spectra_ref = app_services.storage_service.allocate_data(
            AllocationRequest(
                name="mtmf_target_spectra",
                kind="array",
                residency="ram_cacheable",
                size_est=int(target_array.size * target_array.dtype.itemsize),
                shape=target_array.shape,
                dtype=target_array.dtype,
            )
        )
        process_client.write_data(target_spectra_ref, target_array)

        if shift_diff_noise_direction is not None:
            # Image Cube Noise: shift-difference computed internally from the source cube.
            pipeline = get_mnf_mtmf_pipeline(
                dataset_ref=source_ref,
                target_spectra_ref=target_spectra_ref,
                output_ref_name=output_ref_name,
                shift_diff_noise_direction=shift_diff_noise_direction,
            )
            noise_label = f"Image Cube ({shift_diff_noise_direction.name.capitalize()})"
            extra_bindings: Dict[str, DataRef] = {}
        else:
            # Dark Image Noise: external dataset.
            assert noise_dataset is not None
            noise_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=noise_dataset)
            )
            pipeline = get_mnf_mtmf_pipeline(
                dataset_ref=source_ref,
                target_spectra_ref=target_spectra_ref,
                output_ref_name=output_ref_name,
            )
            noise_label = noise_dataset.get_name() or "Noise Dataset"
            extra_bindings = {_MTMF_NOISE_PLAN_BINDING: noise_ref}

        SemanticTask.__init__(
            self,
            priority_class=PriorityClass.BACKGROUND,
            input_ref=source_ref,
            algorithm_pipeline=pipeline,
            task_title="Matched Filter (MTMF)",
            task_variables={
                "Source": source_dataset.get_name() or "Dataset",
                "Noise": noise_label,
                "Targets": len(target_spectra),
            },
            extra_plan_bindings=extra_bindings,
        )
        self.id = app_state.take_next_id()
        self._app_state = app_state
        self._source_dataset = source_dataset
        self._output_ref_name = output_ref_name
        self._target_names: List[str] = target_names
        self.result_ready.connect(self._load_result_into_wiser)

    def completion_callback(self, bindings: Dict[str, DataRef]) -> None:
        """Read the full score cube and emit it for loading on the main thread.

        Args:
            bindings: Mapping of allocation names to resolved :class:`DataRef`
                objects produced by the task pipeline.

        Raises:
            KeyError: If the expected output binding is not present.
        """
        output_ref = bindings.get(self._output_ref_name)
        if output_ref is None:
            raise KeyError(f"Missing MTMF output binding: '{self._output_ref_name}'")

        storage_client = get_process_storage_client()
        meta = storage_client.get_meta(output_ref)
        height, width, n_targets = meta.shape
        region = DatasetRegionRef(y0=0, y1=height, x0=0, x1=width, b0=0, b1=n_targets)
        scores_data, _ = storage_client.read_region(output_ref, region, filter_data=False)
        self.result_ready.emit(np.asarray(scores_data, dtype=np.float32))

    @Slot(object)
    def _load_result_into_wiser(self, scores_data: object) -> None:
        """Slice the score cube into individual datasets and add them to WISER.

        Each band slice ``scores[:, :, i]`` becomes a separate float32 dataset
        named after its target spectrum.  Pixels that were excluded as nodata
        during the matched-filter computation are stored as ``NaN``.

        Args:
            scores_data: The ``(H, W, N_targets)`` score array emitted by
                :meth:`completion_callback`.
        """
        target_names = list(self._target_names)
        scores_array = np.asarray(scores_data, dtype=np.float32)  # (H, W, N_targets)
        names: List[str] = list(target_names)

        loader = self._app_state.get_loader()
        cache = self._app_state.get_cache()
        source_name = self._source_dataset.get_name() or "Dataset"
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        for i, target_name in enumerate(names):
            score_band = scores_array[:, :, i : i + 1]  # (H, W, 1)
            score_by_band = score_band.transpose(2, 0, 1)  # (1, H, W)

            score_dataset = loader.dataset_from_numpy_array(score_by_band, cache)
            score_dataset.set_name(self._app_state.unique_dataset_name(f"MF [{target_name}]: {source_name}"))
            score_dataset.set_description(
                f"Matched filter score for '{target_name}' against '{source_name}' ({timestamp})"
            )
            score_dataset.set_data_ignore_value(float("nan"))
            score_dataset.copy_spatial_metadata(self._source_dataset.get_spatial_metadata())
            self._app_state.add_dataset(score_dataset, view_dataset=False)


# ---------------------------------------------------------------------------
# MTMF dialog
# ---------------------------------------------------------------------------

_INPUT_TYPE_SPECTRUM = 0
_INPUT_TYPE_IMAGE_CUBE = 1


class _NoiseMethod(IntEnum):
    SHIFT_DOWN = 1
    SHIFT_UP = 2
    SHIFT_LEFT = 3
    SHIFT_RIGHT = 4


_NOISE_METHOD_TO_DIRECTION: Dict[_NoiseMethod, ShiftDiffNoiseDirection] = {
    _NoiseMethod.SHIFT_DOWN: ShiftDiffNoiseDirection.DOWN,
    _NoiseMethod.SHIFT_UP: ShiftDiffNoiseDirection.UP,
    _NoiseMethod.SHIFT_LEFT: ShiftDiffNoiseDirection.LEFT,
    _NoiseMethod.SHIFT_RIGHT: ShiftDiffNoiseDirection.RIGHT,
}


def _spectrum_to_single_pixel_dataset(spectrum: Spectrum, app_state: ApplicationState) -> RasterDataSet:
    """Wrap a spectrum as a 1x1xB raster for cube-based pipelines."""
    loader = app_state.get_loader()
    cache = app_state.get_cache()
    arr = np.asarray(spectrum.get_spectrum(), dtype=np.float32)
    arr_by_band = arr[:, np.newaxis, np.newaxis]
    ds = loader.dataset_from_numpy_array(arr_by_band, cache)
    ds.copy_spectral_metadata(spectrum.get_spectral_metadata())
    bb = spectrum.get_bad_bands()
    if bb is not None:
        ds.set_bad_bands(np.asarray(bb).astype(int).tolist())
    nodata = getattr(spectrum, "get_data_ignore_value", lambda: None)()
    if nodata is not None:
        ds.set_data_ignore_value(nodata)
    return ds


class MTMFDialog(QDialog):
    """Dialog to choose source, noise, and targets for :class:`MTMFSemanticTask`."""

    def __init__(
        self,
        app_state: ApplicationState,
        app_services: AppServices,
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.setModal(False)
        self._app_state = app_state
        self._app_services = app_services

        self._ui = Ui_MTMF_Dialog()
        self._ui.setupUi(self)

        self._ui.cbox_input_type.addItem(self.tr("Spectrum"), _INPUT_TYPE_SPECTRUM)
        self._ui.cbox_input_type.addItem(self.tr("Image Cube"), _INPUT_TYPE_IMAGE_CUBE)
        self._ui.cbox_input_type.currentIndexChanged.connect(lambda _i: self._populate_input_combo())

        self._ui.cbox_noise_method.addItem(self.tr("Shift Difference Down"), _NoiseMethod.SHIFT_DOWN)
        self._ui.cbox_noise_method.addItem(self.tr("Shift Difference Up"), _NoiseMethod.SHIFT_UP)
        self._ui.cbox_noise_method.addItem(self.tr("Shift Difference Left"), _NoiseMethod.SHIFT_LEFT)
        self._ui.cbox_noise_method.addItem(self.tr("Shift Difference Right"), _NoiseMethod.SHIFT_RIGHT)
        self._populate_noise_combo()
        self._populate_target_combo()
        self._ui.cbox_input_type.setCurrentIndex(1)
        self._populate_input_combo()

    def showEvent(self, event) -> None:
        # Refresh combos and (re)connect signals each time the dialog is shown.
        self._populate_noise_combo()
        self._populate_target_combo()
        self._populate_input_combo()
        self._app_state.dataset_added.connect(self._on_datasets_changed)
        self._app_state.dataset_removed.connect(self._on_datasets_changed)
        self._app_state.active_spectrum_changed.connect(self._on_spectra_changed)
        self._app_state.collected_spectra_changed.connect(self._on_spectra_changed)
        self._app_state.roi_added.connect(self._on_rois_changed)
        self._app_state.roi_removed.connect(self._on_rois_changed)
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        self._app_state.dataset_added.disconnect(self._on_datasets_changed)
        self._app_state.dataset_removed.disconnect(self._on_datasets_changed)
        self._app_state.active_spectrum_changed.disconnect(self._on_spectra_changed)
        self._app_state.collected_spectra_changed.disconnect(self._on_spectra_changed)
        self._app_state.roi_added.disconnect(self._on_rois_changed)
        self._app_state.roi_removed.disconnect(self._on_rois_changed)
        super().closeEvent(event)

    def _on_datasets_changed(self, *_args) -> None:
        """Refresh dataset-backed combos, preserving current selections."""
        prev_input = self._ui.cbox_input.currentData()
        prev_noise = self._ui.cbox_noise.currentData()
        self._populate_noise_combo()
        self._populate_input_combo()
        if prev_input is not None:
            idx = self._ui.cbox_input.findData(prev_input)
            if idx >= 0:
                self._ui.cbox_input.setCurrentIndex(idx)
        if prev_noise is not None:
            idx = self._ui.cbox_noise.findData(prev_noise)
            if idx >= 0:
                self._ui.cbox_noise.setCurrentIndex(idx)

    def _on_spectra_changed(self, *_args) -> None:
        """Refresh spectrum-backed combos, preserving current selections."""
        prev_input = self._ui.cbox_input.currentData()
        prev_target = self._ui.cbox_target.currentData()
        self._populate_target_combo()
        if self._ui.cbox_input_type.currentData() == _INPUT_TYPE_SPECTRUM:
            self._populate_input_combo()
            if prev_input is not None:
                idx = self._ui.cbox_input.findData(prev_input)
                if idx >= 0:
                    self._ui.cbox_input.setCurrentIndex(idx)
        if prev_target is not None:
            idx = self._ui.cbox_target.findData(prev_target)
            if idx >= 0:
                self._ui.cbox_target.setCurrentIndex(idx)

    def _on_rois_changed(self, *_args) -> None:
        """Refresh noise combo when ROIs are added or removed."""
        prev_noise = self._ui.cbox_noise.currentData()
        self._populate_noise_combo()
        if prev_noise is not None:
            idx = self._ui.cbox_noise.findData(prev_noise)
            if idx >= 0:
                self._ui.cbox_noise.setCurrentIndex(idx)

    def select_image_cube_dataset(self, dataset_id: Optional[int]) -> None:
        """Prefer Image Cube mode and select ``dataset_id`` when present."""
        idx = self._ui.cbox_input_type.findData(_INPUT_TYPE_IMAGE_CUBE)
        if idx >= 0:
            self._ui.cbox_input_type.setCurrentIndex(idx)
        self._populate_input_combo()
        if dataset_id is not None:
            in_idx = self._ui.cbox_input.findData(dataset_id)
            if in_idx >= 0:
                self._ui.cbox_input.setCurrentIndex(in_idx)

    def _spectra_for_plot_input(self) -> List[Spectrum]:
        """Active spectrum first, then collected spectra (unique by id)."""
        seen: set = set()
        out: List[Spectrum] = []
        active = self._app_state.get_active_spectrum()
        if active is not None and active.get_id() is not None:
            seen.add(active.get_id())
            out.append(active)
        for s in self._app_state.get_collected_spectra():
            sid = s.get_id()
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            out.append(s)
        return out

    def _populate_combo_with_separator(self, combo, entries: List[tuple]) -> None:
        """Fill combo with (label, userData) pairs, then separator and (no data)."""
        combo.clear()
        for label, data in entries:
            combo.addItem(label, data)
        combo.insertSeparator(combo.count())
        combo.addItem(self.tr("(no data)"), None)

    def _populate_input_combo(self) -> None:
        mode = self._ui.cbox_input_type.currentData()
        if mode == _INPUT_TYPE_SPECTRUM:
            pairs = []
            for sp in self._spectra_for_plot_input():
                sid = sp.get_id()
                if sid is None:
                    continue
                name = sp.get_name() or self.tr("<unnamed>")
                pairs.append((name, sid))
            self._populate_combo_with_separator(self._ui.cbox_input, pairs)
        else:
            pairs = []
            for ds in self._app_state.get_datasets():
                did = ds.get_id()
                if did is None:
                    continue
                pairs.append((ds.get_name() or self.tr("<unnamed>"), did))
            self._populate_combo_with_separator(self._ui.cbox_input, pairs)

    def _add_section_header(self, combo, label: str) -> None:
        """Insert a bold, non-selectable section-header item into *combo*."""
        combo.addItem(label)
        idx = combo.count() - 1
        item = combo.model().item(idx)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setFlags(item.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsEnabled)

    def _populate_noise_combo(self) -> None:
        """Populate cbox_noise with datasets and ROIs (always, independent of noise method)."""
        combo = self._ui.cbox_noise
        combo.clear()

        self._add_section_header(combo, self.tr("Datasets"))
        combo.insertSeparator(combo.count())
        for ds in self._app_state.get_datasets():
            did = ds.get_id()
            if did is None:
                continue
            combo.addItem(ds.get_name() or self.tr("<unnamed>"), did)

        combo.insertSeparator(combo.count())
        self._add_section_header(combo, self.tr("Regions of Interest"))
        combo.insertSeparator(combo.count())
        for roi in self._app_state.get_rois():
            rid = roi.get_id()
            if rid is None:
                continue
            combo.addItem(roi.get_name() or self.tr("<unnamed>"), ("roi", rid))

        combo.insertSeparator(combo.count())
        combo.addItem(self.tr("(no data)"), None)

        # Auto-select: first dataset → first ROI → (no data)
        first_dataset_idx = -1
        first_roi_idx = -1
        for i in range(combo.count()):
            data = combo.itemData(i)
            if first_dataset_idx < 0 and isinstance(data, int):
                first_dataset_idx = i
            elif first_roi_idx < 0 and isinstance(data, tuple) and data[0] == "roi":
                first_roi_idx = i
        if first_dataset_idx >= 0:
            combo.setCurrentIndex(first_dataset_idx)
        elif first_roi_idx >= 0:
            combo.setCurrentIndex(first_roi_idx)
        else:
            combo.setCurrentIndex(combo.count() - 1)

    def _populate_target_combo(self) -> None:
        pairs = []
        all_spec = self._app_state.get_all_spectra()
        for sid in sorted(all_spec.keys()):
            sp = all_spec[sid]
            label = sp.get_name() or (self.tr("Spectrum") + f" ({sid})")
            pairs.append((label, sid))
        self._populate_combo_with_separator(self._ui.cbox_target, pairs)

    def _resolve_source_dataset(self) -> RasterDataSet:
        mode = self._ui.cbox_input_type.currentData()
        data = self._ui.cbox_input.currentData()
        if data is None:
            raise ValueError(self.tr('Select an input source (not "(no data)").'))
        if mode == _INPUT_TYPE_IMAGE_CUBE:
            ds = self._app_state.get_dataset(int(data))
            if ds is None:
                raise ValueError(self.tr("Selected input dataset is no longer available."))
            return ds
        raise ValueError(self.tr("We currently don't support selecting an input spectra."))

    def _resolve_shift_diff_direction(self) -> ShiftDiffNoiseDirection:
        method = self._ui.cbox_noise_method.currentData()
        direction = _NOISE_METHOD_TO_DIRECTION.get(method)
        if direction is None:
            raise ValueError(self.tr("Select a shift difference direction as the noise method."))
        return direction

    def _resolve_noise_dataset(self) -> RasterDataSet:
        data = self._ui.cbox_noise.currentData()

        if data is None:
            raise ValueError(self.tr('Select a noise source (not "(no data)").'))

        if isinstance(data, tuple) and data[0] == "roi":
            raise NotImplementedError(self.tr("Region of Interest noise is not yet implemented."))

        ds = self._app_state.get_dataset(int(data))
        if ds is None:
            raise ValueError(self.tr("Selected noise dataset is no longer available."))
        return ds

    def _resolve_target_spectra(self) -> List[Spectrum]:
        data = self._ui.cbox_target.currentData()
        if data is None:
            raise ValueError(self.tr('Select a target spectrum (not "(no data)").'))
        sp = self._app_state.get_all_spectra().get(int(data))
        if sp is None:
            raise ValueError(self.tr("Selected target spectrum is no longer available."))
        return [sp]

    def _perform_mtmf(self) -> None:
        source_ds = self._resolve_source_dataset()
        target_list = self._resolve_target_spectra()
        direction = self._resolve_shift_diff_direction()
        task = MTMFSemanticTask(
            app_state=self._app_state,
            app_services=self._app_services,
            source_dataset=source_ds,
            target_spectra=target_list,
            shift_diff_noise_direction=direction,
        )
        task_plan = self._app_services.task_planner.plan_semantic_task(task)
        self._app_services.task_manager.register_and_submit_task_plan(
            self._app_services.scheduler,
            task_plan,
        )

    def accept(self) -> None:
        try:
            self._perform_mtmf()
        except ValueError as exc:
            QMessageBox.warning(self, self.tr("Mixture Tuned Matched Filter"), str(exc))
            return
        QMessageBox.information(
            self,
            self.tr("Mixture Tuned Matched Filter"),
            self.tr("MTMF is running in the background."),
        )
