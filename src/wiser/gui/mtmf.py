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
