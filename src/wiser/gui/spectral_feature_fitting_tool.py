# Child class for Spectral Feature Fitting using the generic parent
from __future__ import annotations

import os
from typing import Dict, Any, Tuple, List
import numpy as np
from numba import types, prange
from scipy.interpolate import interp1d

from astropy import units as u
from wiser.raster.loader import RasterDataLoader
from wiser.raster.spectrum import NumPyArraySpectrum, Spectrum
from wiser.gui.app_state import ApplicationState
from .generic_spectral_tool import GenericSpectralComputationTool
from wiser.utils.numba_wrapper import numba_njit_wrapper
from .util import (
    interp1d_monotonic,
    interp1d_monotonic_numba,
    slice_to_bounds_3D,
    slice_to_bounds_3D_numba,
    slice_to_bounds_1D,
    slice_to_bounds_1D_numba,
    dot3d,
    dot3d_numba,
    compute_rmse,
    compute_rmse_numba,
)
from wiser.gui.permanent_plugins.continuum_removal_plugin import (
    continuum_removal_image,
    continuum_removal_image_numba,
    continuum_removal,
    continuum_removal_numba,
)

SFF_NEUTRAL_FILL_VALUE = 1.0


def compute_sff_image(
    target_image_arr: np.ndarray,
    target_wavelengths: np.ndarray,
    target_bad_bands: np.ndarray,
    min_wvl: np.float32,
    max_wvl: np.float32,
    reference_spectra: np.ndarray,
    reference_spectra_wvls: np.ndarray,
    reference_spectra_bad_bands: np.ndarray,
    reference_spectra_indices: np.ndarray,
    thresholds: np.float32,
):
    if (
        reference_spectra_wvls.shape[0] != reference_spectra_bad_bands.shape[0]
        or reference_spectra.shape[0] != reference_spectra_wvls.shape[0]
    ):
        raise ValueError("Shape mismatch in reference spectra and wavelengths/bad bands.")

    # Slice image and metadata to wavelength bounds.
    target_image_arr_sliced, target_wvls_sliced, target_bad_bands_sliced = slice_to_bounds_3D(
        target_image_arr,
        target_wavelengths,
        target_bad_bands,
        min_wvl,
        max_wvl,
    )
    target_image_arr_sliced = target_image_arr_sliced[target_bad_bands_sliced, :, :]

    # Meta-data needed to do continuum removal
    good_bands = np.array([0] * target_image_arr_sliced.shape[0], dtype=np.bool_)
    target_good_wvls = target_wvls_sliced[target_bad_bands_sliced].copy()
    rows = target_image_arr_sliced.shape[1]
    cols = target_image_arr_sliced.shape[2]
    bands = target_image_arr_sliced.shape[0]

    # [b][y][x] -> [y][x][b], continuum removal takes the array in this way
    target_image_arr_sliced = target_image_arr_sliced.transpose((1, 2, 0))
    target_image_cr = np.float32(1.0) - continuum_removal_image(
        target_image_arr_sliced,
        good_bands,
        target_good_wvls,
        rows,
        cols,
        bands,
    )
    # Now, we must set the non-finite values to 0, so they don't show up in
    # the scale or RMSE
    target_non_finite_mask = ~np.isfinite(target_image_cr)
    target_image_cr[target_non_finite_mask] = 0.0
    # [b][y][x] -> [y][x][b]
    target_image_cr = target_image_cr.transpose((1, 2, 0))

    # Validate numerical integrity
    if not np.isfinite(target_image_cr).all():
        raise ValueError("Target image array is not finite after cleaning")

    num_spectra = reference_spectra_indices.shape[0] - 1

    # Output buffers: one layer per reference spectrum.
    out_classification = np.empty(
        (
            num_spectra,
            target_image_arr_sliced.shape[0],
            target_image_arr_sliced.shape[1],
        ),
        dtype=np.bool_,
    )
    out_rmse = np.empty(
        (
            num_spectra,
            target_image_arr_sliced.shape[0],
            target_image_arr_sliced.shape[1],
        ),
        dtype=np.float32,
    )
    out_scale = np.empty(
        (
            num_spectra,
            target_image_arr_sliced.shape[0],
            target_image_arr_sliced.shape[1],
        ),
        dtype=np.float32,
    )

    for i in prange(reference_spectra_indices.shape[0] - 1):
        # Get reference and clean
        start = reference_spectra_indices[i]
        end = reference_spectra_indices[i + 1]
        ref_spectrum = reference_spectra[start:end]
        ref_wvls = reference_spectra_wvls[start:end]
        ref_bad_bands = reference_spectra_bad_bands[start:end]

        # For reference spectrum:
        # interpolate (regrid to target) --> slice to target bounds -->
        # remove ref's bad bands (we regrid this so we can do this on target's
        # bounds) --> continuum remove --> remove target's bad bands

        # Regrid to target wvls
        if ref_wvls.shape == target_wavelengths.shape and np.allclose(
            ref_wvls, target_wavelengths, rtol=0.0, atol=1e-9
        ):
            ref_spectrum_interp = ref_spectrum
            ref_bad_bands_interp = ref_bad_bands
        else:
            ref_spectrum_interp = interp1d_monotonic(
                ref_wvls,
                ref_spectrum,
                target_wavelengths,
            )

            ref_bad_bands_float = ref_bad_bands.astype(np.float32)
            ref_bad_bands_interp = interp1d_monotonic(
                ref_wvls,
                ref_bad_bands_float,
                target_wavelengths,
            )
            ref_bad_bands_interp[ref_bad_bands_interp < 1.0] = 0.0
            ref_bad_bands_interp[~np.isfinite(ref_bad_bands_interp)] = 0.0
            ref_bad_bands_interp = ref_bad_bands_interp.astype(np.bool_)

        # Slice to target bounds
        ref_spectrum_sliced, _, ref_bad_bands_sliced = slice_to_bounds_1D(
            ref_spectrum_interp,
            target_wavelengths,
            ref_bad_bands_interp,
            min_wvl,
            max_wvl,
        )

        # Remove bad bands
        ref_spec_bb_removed = ref_spectrum_sliced[ref_bad_bands_sliced]
        ref_wvl_bb_removed = target_wvls_sliced[ref_bad_bands_sliced]
        # Calculate Continuum Removal
        ref_spec_cr, _ = continuum_removal(
            ref_spec_bb_removed,
            ref_wvl_bb_removed,
        )
        # Put back into the target wvl grid
        ref_spec_cr_target_grid = np.full_like(
            ref_spectrum_sliced,
            SFF_NEUTRAL_FILL_VALUE,
            dtype=np.float32,
        )
        ref_spec_cr_target_grid[ref_bad_bands_sliced] = ref_spec_cr

        # Remove the target wvl bad bands
        ref_spec_cr_target_grid_target_bb = ref_spec_cr_target_grid[target_bad_bands_sliced]
        ref_bad_bands = ref_bad_bands_sliced[target_bad_bands_sliced]

        # We technically shouldn't need to do this because the user's mask should work
        # but we want to be defensive against bad inputs
        finite_mask = np.isfinite(ref_spec_cr_target_grid_target_bb).astype(np.bool_)
        ref_bad_bands = ref_bad_bands & finite_mask
        ref_spectrum_cr = np.float32(1.0) - ref_spec_cr_target_grid_target_bb
        ref_spectrum_cr[~ref_bad_bands] = 0.0

        # Now they are all the same elgnth and 1's where we kee, 0's where
        # we remove/delete
        num = dot3d(target_image_cr, ref_spectrum_cr, ref_bad_bands)
        denom = np.float32((ref_spectrum_cr * ref_spectrum_cr).sum())
        scale = num / denom

        rmse = compute_rmse(target_image_cr, scale, ref_spectrum_cr, ref_bad_bands)
        thr = thresholds[i]
        out_classification[i, :, :] = rmse < thr
        out_rmse[i, :, :] = rmse
        out_scale[i, :, :] = scale

    return out_classification, out_rmse, out_scale


compute_sff_image_sig = types.Tuple(
    (
        types.boolean[:, :, :],
        types.float32[:, :, :],
        types.float32[:, :, :],
    )
)(
    types.float32[:, :, :],
    types.float32[:],
    types.boolean[:],
    types.float32,
    types.float32,
    types.float32[:],
    types.float32[:],
    types.boolean[:],
    types.uint32[:],
    types.float32[:],
)


@numba_njit_wrapper(
    non_njit_func=compute_sff_image,
    signature=compute_sff_image_sig,
    parallel=True,
    cache=True,
)
def compute_sff_image_numba(
    target_image_arr: np.ndarray,
    target_wavelengths: np.ndarray,
    target_bad_bands: np.ndarray,
    min_wvl: np.float32,
    max_wvl: np.float32,
    reference_spectra: np.ndarray,
    reference_spectra_wvls: np.ndarray,
    reference_spectra_bad_bands: np.ndarray,
    reference_spectra_indices: np.ndarray,
    thresholds: np.float32,
):
    """Compute Spectral Feature Fitting (SFF) classification for an image.

    This function applies the Spectral Feature Fitting (SFF) algorithm to each
    pixel spectrum of a hyperspectral cube using a bank of reference spectra.
    The workflow is:

    1. Slice the image cube and reference spectra to the requested wavelength range.
    2. Remove bands marked as bad by ``target_bad_bands``.
    3. Continuum-remove and invert both the image spectra and reference spectra.
    4. Compute the scale factor and RMSE (residual error) between each pixel
       spectrum and each reference spectrum.
    5. Apply per-spectrum thresholds to generate boolean classification layers.

    The computation is optimized for Numba and parallelized over the reference
    spectra.

    Args:
        target_image_arr (np.ndarray):
            Float32 array of shape ``(bands, rows, cols)`` representing the
            hyperspectral image cube.
        target_wavelengths (np.ndarray):
            Float32 1D array of wavelength positions for the target image.
        target_bad_bands (np.ndarray):
            Boolean 1D mask of length ``bands`` indicating which bands to keep.
        min_wvl (np.float32):
            Minimum wavelength for slicing (inclusive).
        max_wvl (np.float32):
            Maximum wavelength for slicing (inclusive).
        reference_spectra (np.ndarray):
            Float32 1D array of concatenated reference spectra.
        reference_spectra_wvls (np.ndarray):
            Float32 1D array of wavelengths aligned with ``reference_spectra``.
        reference_spectra_bad_bands (np.ndarray):
            Boolean 1D mask describing bad samples in the reference spectra. 1's are
            bands to keep, 0's are bands to remove.
        reference_spectra_indices (np.ndarray):
            UInt32 1D array of segment boundaries; each pair ``[i, i+1]``
            defines a single reference spectrum slice inside the packed array.
        thresholds (np.ndarray):
            Float32 1D array of RMSE thresholds, one per reference spectrum.

    Returns:
        Tuple(np.ndarray, np.ndarray, np.ndarray):
            A tuple ``(classification, rmse, scale)`` where each element is a
            3D array of shape ``(num_refs, rows, cols)``

            * ``classification`` — boolean mask where ``True`` indicates a
              match for the corresponding reference spectrum.
            * ``rmse`` — float32 RMSE image for each reference spectrum.
            * ``scale`` — float32 scale factor image for each reference spectrum.

    Raises:
        ValueError: If wavelength arrays or bad-band masks have inconsistent
            shapes or contain non-finite values.

    Notes:
        This function is the Numba-accelerated implementation of SFF. The pure
        Python version is ``compute_sff_image`` and shares the same behavior.

    """
    if (
        reference_spectra_wvls.shape[0] != reference_spectra_bad_bands.shape[0]
        or reference_spectra.shape[0] != reference_spectra_wvls.shape[0]
    ):
        raise ValueError("Shape mismatch in reference spectra and wavelengths/bad bands.")

    # Slice image and metadata to wavelength bounds.
    target_image_arr_sliced, target_wvls_sliced, target_bad_bands_sliced = slice_to_bounds_3D_numba(
        target_image_arr,
        target_wavelengths,
        target_bad_bands,
        min_wvl,
        max_wvl,
    )
    target_image_arr_sliced = target_image_arr_sliced[target_bad_bands_sliced, :, :]

    # Meta-data needed to do continuum removal
    good_bands = np.array([0] * target_image_arr_sliced.shape[0], dtype=np.bool_)
    good_wvls = target_wvls_sliced[target_bad_bands_sliced].copy()
    rows = target_image_arr_sliced.shape[1]
    cols = target_image_arr_sliced.shape[2]
    bands = target_image_arr_sliced.shape[0]

    # [b][y][x] -> [y][x][b], continuum removal takes the array in this way
    target_image_arr_sliced = target_image_arr_sliced.transpose((1, 2, 0))
    target_image_cr = np.float32(1.0) - continuum_removal_image_numba(
        target_image_arr_sliced,
        good_bands,
        good_wvls,
        rows,
        cols,
        bands,
    )
    # Now, we must set the non-finite values to 0, so they don't show up in
    # the scale or RMSE
    target_non_finite_mask = ~np.isfinite(target_image_cr)
    B, Y, X = target_image_cr.shape
    for b in prange(B):
        for y in range(Y):
            for x in range(X):
                if target_non_finite_mask[b, y, x]:
                    target_image_cr[b, y, x] = 0.0
    # [b][y][x] -> [y][x][b]
    target_image_cr = target_image_cr.transpose((1, 2, 0))

    # Validate numerical integrity
    if not np.isfinite(target_image_cr).all():
        raise ValueError("Target image array is not finite after cleaning")

    num_spectra = reference_spectra_indices.shape[0] - 1

    # Output buffers: one layer per reference spectrum.
    out_classification = np.empty(
        (
            num_spectra,
            target_image_arr_sliced.shape[0],
            target_image_arr_sliced.shape[1],
        ),
        dtype=np.bool_,
    )
    out_rmse = np.empty(
        (
            num_spectra,
            target_image_arr_sliced.shape[0],
            target_image_arr_sliced.shape[1],
        ),
        dtype=np.float32,
    )
    out_scale = np.empty(
        (
            num_spectra,
            target_image_arr_sliced.shape[0],
            target_image_arr_sliced.shape[1],
        ),
        dtype=np.float32,
    )

    for i in prange(reference_spectra_indices.shape[0] - 1):
        # Get reference and clean
        start = reference_spectra_indices[i]
        end = reference_spectra_indices[i + 1]
        ref_spectrum = reference_spectra[start:end]
        ref_wvls = reference_spectra_wvls[start:end]
        ref_bad_bands = reference_spectra_bad_bands[start:end]

        # For reference spectrum:
        # interpolate (regrid to target wvls) --> slice to target bounds -->
        # remove ref's bad bands --> continuum remove --> regrid to target wvls sliced
        # --> remove target's bad bands

        # Regrid to target wvls
        if ref_wvls.shape == target_wavelengths.shape and np.allclose(
            ref_wvls, target_wavelengths, rtol=0.0, atol=1e-9
        ):
            ref_spectrum_interp = ref_spectrum
            ref_bad_bands_interp = ref_bad_bands
        else:
            ref_spectrum_interp = interp1d_monotonic_numba(
                ref_wvls,
                ref_spectrum,
                target_wavelengths,
            )

            ref_bad_bands_float = ref_bad_bands.astype(np.float32)
            ref_bad_bands_interp = interp1d_monotonic_numba(
                ref_wvls,
                ref_bad_bands_float,
                target_wavelengths,
            )
            ref_bad_bands_interp[ref_bad_bands_interp < 1.0] = 0.0
            ref_bad_bands_interp[~np.isfinite(ref_bad_bands_interp)] = 0.0
            ref_bad_bands_interp = ref_bad_bands_interp.astype(np.bool_)

        ref_spectrum_sliced, ref_wvls_sliced, ref_bad_bands_sliced = slice_to_bounds_1D_numba(
            ref_spectrum_interp,
            target_wavelengths,
            ref_bad_bands_interp,
            min_wvl,
            max_wvl,
        )

        # Remove bad bands
        ref_spec_bb_removed = ref_spectrum_sliced[ref_bad_bands_sliced]
        ref_wvl_bb_removed = ref_wvls_sliced[ref_bad_bands_sliced]
        # Calculate Continuum Removal
        ref_spec_cr, _ = continuum_removal_numba(
            ref_spec_bb_removed,
            ref_wvl_bb_removed,
        )
        # Put back into the target wvl grid
        ref_spec_cr_target_grid = np.full_like(
            ref_spectrum_sliced,
            SFF_NEUTRAL_FILL_VALUE,
            dtype=np.float32,
        )
        ref_spec_cr_target_grid[ref_bad_bands_sliced] = ref_spec_cr

        # Remove the target wvl bad bands
        ref_spec_cr_target_grid_target_bb = ref_spec_cr_target_grid[target_bad_bands_sliced]
        ref_bad_bands = ref_bad_bands_sliced[target_bad_bands_sliced]

        # We technically shouldn't need to do this because the user's mask should work
        # but we want to be defensive against bad inputs
        finite_mask = np.isfinite(ref_spec_cr_target_grid_target_bb).astype(np.bool_)
        ref_bad_bands = ref_bad_bands & finite_mask
        ref_spectrum_cr = np.float32(1.0) - ref_spec_cr_target_grid_target_bb
        ref_spectrum_cr[~ref_bad_bands] = 0.0

        num = dot3d_numba(target_image_cr, ref_spectrum_cr, ref_bad_bands)
        denom = np.float32((ref_spectrum_cr * ref_spectrum_cr).sum())
        scale = num / denom

        rmse = compute_rmse_numba(target_image_cr, scale, ref_spectrum_cr, ref_bad_bands)
        thr = thresholds[i]
        out_classification[i, :, :] = rmse < thr
        out_rmse[i, :, :] = rmse
        out_scale[i, :, :] = scale

    return out_classification, out_rmse, out_scale


class SFFTool(GenericSpectralComputationTool):
    SETTINGS_NAMESPACE = "Wiser/SFFPlugin"
    RUN_BUTTON_TEXT = "Run SFF"
    SCORE_HEADER = "Fit RMS"
    THRESHOLD_HEADER = "Initial RMSE"
    THRESHOLD_SPIN_CONFIG = dict(min=0.0, max=1.0, decimals=4, step=0.005)

    def __init__(self, app_state: ApplicationState, parent=None):
        self._max_rms: float = 0.03  # metric-specific name as requested
        super().__init__("Spectral Feature Fitting", app_state, parent)
        self._ui.method_threshold.setValue(self._max_rms)
        self._maybe_add_default_library()

    # keep metric-specific name but wire into parent's UI
    def set_method_threshold(self, value: float | None) -> None:
        self._max_rms = float(value) if value is not None else self._max_rms

    def details_columns(self) -> List[tuple]:
        return [
            (self.SCORE_HEADER, "score"),
            ("Max RMS", "threshold"),
            ("Scale", "scale"),
        ]

    def filename_stub(self) -> str:
        return "SFF"

    def default_library_path(self):
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(
            gui_dir,
            "../data/" "usgs_default_ref_lib",
            "USGS_Mineral_Spectral_Library.hdr",
        )
        return default_path

    # ---------- SFF helpers ----------
    @staticmethod
    def _resample_to(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
        if x_src.size < 2 or np.all(~np.isfinite(y_src)):
            return np.full_like(x_dst, np.nan, dtype=float)
        # Slice to bounds should have already taken care of the case that something is outside
        # of x_src's range, so this shold only happen if a value is really close to the range
        # which will only happen from a floating point error. We want to keep these values.
        f = interp1d(x_src, y_src, kind="linear", bounds_error=False, fill_value="extrapolate")
        return f(x_dst)

    def compute_score(
        self,
        target: Spectrum,
        ref: Spectrum,
        min_wvl: u.Quantity,
        max_wvl: u.Quantity,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Spectral Feature Fitting (SFF) score between a target spectrum and a reference
        spectrum, implemented to match the logic of `compute_sff_image` but for 1D spectra.

        Returns
        -------
        rms : float
            RMSE of the continuum-removed, inverted target vs. scaled reference.
        info : dict
            Currently contains {"scale": scale}.
        """
        # Convert wavelengths to a common unit and build numpy arrays
        target_unit = target.get_wavelength_units()
        if target_unit is None:
            raise ValueError("Target spectrum has no wavelength units.")

        target_wvls_arr = np.array([w.to(target_unit).value for w in target.get_wavelengths()])
        target_arr = np.asarray(target.get_spectrum(), dtype=np.float32)

        # For now, treat all bands as "good" (no bad-band mask available at this layer).
        # This matches the SAM `compute_score` approach.
        target_bad_bands = target.get_bad_bands()

        min_wvl_value = float(min_wvl.to(target_unit).value)
        max_wvl_value = float(max_wvl.to(target_unit).value)

        # Slice target to wavelength bounds (analogous to slice_to_bounds_3D in image version)
        target_arr_sliced, target_wvls_sliced, target_bad_bands_sliced = slice_to_bounds_1D(
            spectrum_arr=target_arr,
            wvls=target_wvls_arr,
            bad_bands=target_bad_bands,
            min_wvl=min_wvl_value,
            max_wvl=max_wvl_value,
        )

        target_arr = target_arr_sliced[target_bad_bands_sliced]
        target_arr_cr, _ = continuum_removal(target_arr, target_wvls_sliced[target_bad_bands_sliced])
        t_cr = np.float32(1.0) - target_arr_cr
        target_non_finite_mask = ~np.isfinite(t_cr)
        t_cr[target_non_finite_mask] = 0.0

        # Reference: resample to target wavelengths, then slice similarly
        ref_arr = np.asarray(ref.get_spectrum(), dtype=np.float32)
        ref_wvls_arr = np.array([w.to(target_unit).value for w in ref.get_wavelengths()])
        ref_bad_bands = ref.get_bad_bands()

        # Regrid reference to match target wavelength sampling if needed
        if ref_wvls_arr.shape == target_wvls_arr.shape and np.allclose(
            ref_wvls_arr, target_wvls_arr, rtol=0.0, atol=1e-9
        ):
            ref_resampled = ref_arr
            ref_bad_bands_resampled = ref_bad_bands
        else:
            # Same conceptual step as interp1d_monotonic(...) in compute_sff_image
            ref_resampled = self._resample_to(ref_wvls_arr, ref_arr, target_wvls_arr)
            ref_bad_bands_resampled = self._resample_to(
                ref_wvls_arr,
                ref_bad_bands.astype(np.float32),
                target_wvls_arr,
            )
            ref_bad_bands_resampled[ref_bad_bands_resampled < 1.0] = 0.0
            ref_bad_bands_resampled[~np.isfinite(ref_bad_bands_resampled)] = 0.0
            ref_bad_bands_resampled = ref_bad_bands_resampled.astype(np.bool_)

        # Slice the resampled reference to the same wavelength bounds
        ref_arr_sliced, _, ref_bad_bands_sliced = slice_to_bounds_1D(
            spectrum_arr=ref_resampled,
            wvls=target_wvls_arr,
            bad_bands=ref_bad_bands_resampled,
            min_wvl=min_wvl_value,
            max_wvl=max_wvl_value,
        )

        ref_arr_bb_removed = ref_arr_sliced[ref_bad_bands_sliced]
        ref_wvl_bb_removed = target_wvls_sliced[ref_bad_bands_sliced]
        ref_cr_bb_removed, _ = continuum_removal(ref_arr_bb_removed, ref_wvl_bb_removed)
        ref_cr = np.full_like(
            ref_arr_sliced,
            SFF_NEUTRAL_FILL_VALUE,
            dtype=np.float32,
        )
        ref_cr[ref_bad_bands_sliced] = ref_cr_bb_removed

        ref_cr = ref_cr[target_bad_bands_sliced]
        ref_bad_bands = ref_bad_bands_sliced[target_bad_bands_sliced]
        ref_bad_bands = ref_bad_bands & np.isfinite(ref_cr).astype(np.bool_)

        r_cr = np.float32(1.0) - ref_cr
        r_cr[~ref_bad_bands] = 0.0

        # Validate numerical integrity similar to compute_sff_image checks
        if not np.isfinite(t_cr).all():
            return (np.nan, {})
        if not np.isfinite(r_cr).all():
            return (np.nan, {})

        # SFF core: scale, residual, RMSE
        num = float(np.dot(r_cr * ref_bad_bands, t_cr))
        denom = float(np.dot(r_cr * ref_bad_bands, r_cr))

        if denom == 0.0 or not np.isfinite(denom):
            return (np.nan, {})

        scale = num / denom

        resid = t_cr - scale * r_cr
        rms = float(np.sqrt(np.sum(resid**2) / ref_bad_bands.sum()))

        return rms, {"scale": float(scale)}

    def compute_score_image(
        self,
        target_image_name: str,
        target_image_arr: np.ndarray,
        target_wavelengths: np.ndarray,
        target_bad_bands: np.ndarray,
        min_wvl: np.float32,
        max_wvl: np.float32,
        reference_spectra: List[NumPyArraySpectrum],
        reference_spectra_arr: np.ndarray,
        reference_spectra_wvls: np.ndarray,
        reference_spectra_bad_bands: np.ndarray,
        reference_spectra_indices: np.ndarray,
        thresholds: np.ndarray,
        python_mode: bool = False,
    ) -> List[int]:
        if not python_mode:
            out_cls, out_rmse, out_scale = compute_sff_image_numba(
                target_image_arr,
                target_wavelengths,
                target_bad_bands,
                min_wvl,
                max_wvl,
                reference_spectra_arr,
                reference_spectra_wvls,
                reference_spectra_bad_bands,
                reference_spectra_indices,
                thresholds,
            )
        else:
            out_cls, out_rmse, out_scale = compute_sff_image(
                target_image_arr,
                target_wavelengths,
                target_bad_bands,
                min_wvl,
                max_wvl,
                reference_spectra_arr,
                reference_spectra_wvls,
                reference_spectra_bad_bands,
                reference_spectra_indices,
                thresholds,
            )
        loader = RasterDataLoader()

        # Load in out_cls dataset
        out_cls_dataset = loader.dataset_from_numpy_array(out_cls)
        out_cls_dataset.set_name(
            self._app_state.unique_dataset_name(f"SFF CLS, Img: {target_image_name}"),
        )
        band_descriptions = []

        for i in range(0, len(reference_spectra)):
            spectrum_name = reference_spectra[i].get_name()
            band_descriptions.append(f"Spec: {spectrum_name}")

        out_cls_dataset.set_band_descriptions(band_descriptions)
        self._app_state.add_dataset(out_cls_dataset)

        # Load in out_rmse dataset
        out_rmse_dataset = loader.dataset_from_numpy_array(out_rmse)
        out_rmse_dataset.set_name(
            self._app_state.unique_dataset_name(f"SFF RMSE, Img: {target_image_name}"),
        )
        band_descriptions = []

        for i in range(0, len(reference_spectra)):
            spectrum_name = reference_spectra[i].get_name()
            band_descriptions.append(f"Spec: {spectrum_name}")

        out_rmse_dataset.set_band_descriptions(band_descriptions)
        self._app_state.add_dataset(out_rmse_dataset)

        # Load in out_scale dataset
        out_scale_dataset = loader.dataset_from_numpy_array(out_scale)
        out_scale_dataset.set_name(
            self._app_state.unique_dataset_name(f"SFF SCALE, Img: {target_image_name}"),
        )
        band_descriptions = []

        for i in range(0, len(reference_spectra)):
            spectrum_name = reference_spectra[i].get_name()
            band_descriptions.append(f"Spec: {spectrum_name}")

        out_scale_dataset.set_band_descriptions(band_descriptions)
        self._app_state.add_dataset(out_scale_dataset)

        return [
            out_cls_dataset.get_id(),
            out_rmse_dataset.get_id(),
            out_scale_dataset.get_id(),
        ]
