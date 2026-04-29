"""Matched target / matched filter (MTMF) task stage for hyperspectral cubes."""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from wiser.utils.primitives import (
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    SpatialTileScheme,
)
from wiser.utils.task_system import (
    AlgorithmPipeline,
    BasePlanMeta,
    DatasetPlanMeta,
    MapStage,
    ResourceModel,
    WriteSpec,
)
from wiser.utils.task_stage_utils import get_good_band_runs, split_dataset_tile_by_good_band_runs
from wiser.utils.worker_runtime import get_process_storage_client


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


@dataclass
class MatchedFilterStage(MapStage):
    """
    Compute the matched-filter detection score between every pixel in a
    hyperspectral data cube and one or more reference target spectra.

    Inputs (all supplied via broadcast_input as DataRef objects):
        inv_noise_cov_ref  - (B, B) inverse noise covariance matrix
        noise_mean_ref     - (B,)   noise mean vector
        target_spectra_ref - (N_targets, B) array; each row is one target

    Output:
        A single dataset of shape (H, W, N_targets) where
        output[:, :, i] is the matched-filter score map for target i.
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
