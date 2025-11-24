# Child class for Spectral Feature Fitting using the generic parent
from __future__ import annotations

import os
from typing import Dict, Any, Tuple, List
import numpy as np
from numba import types, prange
from scipy.interpolate import interp1d

from astropy import units as u
from wiser.raster.spectrum import NumPyArraySpectrum
from wiser.gui.app_state import ApplicationState
from .generic_spectral_tool import GenericSpectralComputationTool
from wiser.utils.numba_wrapper import numba_njit_wrapper
from .util import (
    interp1d_monotonic_numba,
    slice_to_bounds_3D_numba,
    slice_to_bounds_1D_numba,
    dot3d_numba,
)
from wiser.gui.permanent_plugins.continuum_removal_plugin import (
    continuum_removal_image_numba,
    continuum_removal_numba,
)


def compute_sff_image(
    target_image_arr: np.ndarray,  # float32[:, :, :]
    target_wavelengths: np.ndarray,  # float32[:]
    target_bad_bands: np.ndarray,  # bool[:]
    min_wvl: np.float32,  # float32
    max_wvl: np.float32,  # float32
    reference_spectra: np.ndarray,  # float32 [:]
    reference_spectra_wvls: np.ndarray,  # float32 [:], in target_image_arr units
    reference_spectra_bad_bands: np.ndarray,  # bool[:]
    reference_spectra_indices: np.ndarray,  # uint32[:]
):
    pass


compute_sff_image_sig = types.float32[:, :, :](
    types.float32[:, :, :],  # target_image_arr
    types.float32[:],  # target_wavelengths
    types.boolean[:],  # target_bad_bands
    types.float32,  # min_wvl
    types.float32,  # max_wvl
    types.float32[:],  # reference_spectra
    types.float32[:],  # reference_spectra_wvls
    types.boolean[:],  # reference_spectra_bad_bands
    types.uint32[:],  # reference_spectra_indices
)


# @numba_njit_wrapper(
#     non_njit_func=compute_sff_image,
#     signature=compute_sff_image_sig,
#     parallel=True,
# )
def compute_sff_image_numba(
    target_image_arr: np.ndarray,  # float32[:, :, :]
    target_wavelengths: np.ndarray,  # float32[:]
    target_bad_bands: np.ndarray,  # bool[:]
    min_wvl: np.float32,  # float32
    max_wvl: np.float32,  # float32
    reference_spectra: np.ndarray,  # float32 [:]
    reference_spectra_wvls: np.ndarray,  # float32[:], in target_image_arr units
    reference_spectra_bad_bands: np.ndarray,  # bool[:]
    reference_spectra_indices: np.ndarray,  # uint32[:]
):
    """
    1. Slice image to range
    2. Create output array
    3. For each spectra. Slice it to range. Continuum remove, compute the scale and rms
    """
    if (
        reference_spectra_wvls.shape[0] != reference_spectra_bad_bands.shape[0]
        or reference_spectra.shape[0] != reference_spectra_wvls.shape[0]
    ):
        raise ValueError("Shape mismatch in reference spectra and wavelengths/bad bands.")

    target_image_arr_sliced, target_wvls_sliced, target_bad_bands_sliced = slice_to_bounds_3D_numba(
        target_image_arr,
        target_wavelengths,
        target_bad_bands,
        min_wvl,
        max_wvl,
    )
    target_image_arr_sliced = target_image_arr_sliced[target_bad_bands_sliced, :, :]

    target_image_cr = np.float32(1.0) - continuum_removal_image_numba(
        target_image_arr_sliced,
        np.array([0] * target_image_arr_sliced.shape[0], dtype=np.bool_),
        target_wvls_sliced[target_bad_bands],
        np.int32(target_image_arr_sliced.shape[1]),
        np.int32(target_image_arr_sliced.shape[2]),
        np.int32(target_image_arr_sliced.shape[0]),
    )
    if not np.isfinite(target_image_arr_sliced).all():
        raise ValueError("Target image array is not finite after cleaning")
    if not np.isfinite(reference_spectra[reference_spectra_bad_bands]).all():
        raise ValueError("Reference spectra array is not finite")
    num_spectra = reference_spectra_indices.shape[0] - 1
    out = np.empty(
        (
            num_spectra,
            target_image_arr_sliced.shape[1],
            target_image_arr_sliced.shape[2],
        ),
        dtype=np.float32,
    )

    target_image_arr_sliced = target_image_arr_sliced.transpose((1, 2, 0))
    for i in prange(reference_spectra_indices.shape[0] - 1):
        start = reference_spectra_indices[i]
        end = reference_spectra_indices[i + 1]
        ref_spectrum = reference_spectra[start:end]
        ref_wvls = reference_spectra_wvls[start:end]
        ref_bad_bands = reference_spectra_bad_bands[start:end]

        ref_spectrum_sliced, ref_wvls_sliced, ref_bad_bands_sliced = slice_to_bounds_1D_numba(
            ref_spectrum,
            ref_wvls,
            ref_bad_bands,
            min_wvl,
            max_wvl,
        )

        # Interpolate to target wvl's spacing
        if ref_wvls_sliced.shape == target_wvls_sliced.shape and np.allclose(
            ref_wvls_sliced, target_wvls_sliced, rtol=0.0, atol=1e-9
        ):
            ref_spectrum_interp = ref_spectrum_sliced
        else:
            ref_spectrum_interp = interp1d_monotonic_numba(
                ref_wvls_sliced,
                ref_spectrum_sliced,
                target_wvls_sliced,
            )

        ref_spectrum_good_bands = ref_spectrum_interp[target_bad_bands_sliced]
        wvls_sliced_good_bands = target_wvls_sliced[target_bad_bands_sliced]

        # Continuum remove and invert
        ref_spectrum_cr, _ = continuum_removal_numba(
            ref_spectrum_good_bands,
            wvls_sliced_good_bands,
        )
        ref_spectrum_cr = np.float32(1.0) - ref_spectrum_cr

        num = dot3d_numba(target_image_cr, ref_spectrum_cr)
        denom = np.float32((ref_spectrum_cr * ref_spectrum_cr).sum())

        scale = num / denom
        resid = target_image_cr - scale * ref_spectrum_cr
        rms = np.sqrt(np.nanmean(resid**2))  # noqa: F841

    return out


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
        print(f"New max rms: {self._max_rms}")

    def details_columns(self) -> List[tuple]:
        # include scale column that is specific to SFF
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

    @staticmethod
    def _continuum_curve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = y.size
        if n < 2:
            return np.maximum(y, 1e-12)
        idx = [0]
        for i in range(1, n - 1):
            if (y[i] >= y[i - 1]) and (y[i] >= y[i + 1]):
                idx.append(i)
        idx.append(n - 1)
        idx = np.unique(idx)
        changed = True
        while changed and len(idx) > 2:
            changed = False
            keep = [idx[0]]
            for k in range(1, len(idx) - 1):
                i_prev, i, i_next = keep[-1], idx[k], idx[k + 1]
                x0, y0 = x[i_prev], y[i_prev]
                x1, y1 = x[i_next], y[i_next]
                if x1 == x0:
                    if y[i] < max(y0, y1):
                        changed = True
                        continue
                else:
                    t = (x[i] - x0) / (x1 - x0)
                    y_line = y0 + t * (y1 - y0)
                    if y[i] < y_line - 1e-12:
                        changed = True
                        continue
                keep.append(i)
            keep.append(idx[-1])
            idx = np.array(keep, dtype=int)
        f = interp1d(x[idx], y[idx], kind="linear", bounds_error=False, fill_value="extrapolate")
        cont = f(x)
        return np.maximum(cont, 1e-12)

    @classmethod
    def _continuum_remove_and_invert(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cont = cls._continuum_curve(x, y)
        cr = y / cont
        return 1.0 - cr

    # compute SFF RMS + scale between target and ref
    def compute_score(
        self,
        target: NumPyArraySpectrum,
        ref: NumPyArraySpectrum,
        min_wvl: u.Quantity,
        max_wvl: u.Quantity,
    ) -> Tuple[float, Dict[str, Any]]:
        MIN_SAMPLES = 3
        t_reflect, t_wls = self._slice_to_bounds(
            spectrum=target,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
        r_reflect, r_wls = self._slice_to_bounds(
            spectrum=ref,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

        t_x = t_wls.value
        r_x = r_wls.value

        if np.array_equal(r_x, t_x):
            r_reflect_rs = r_reflect
        else:
            r_reflect_rs = self._resample_to(r_x, r_reflect, t_x)

        valid = np.isfinite(t_reflect) & np.isfinite(r_reflect_rs)
        if valid.sum() < MIN_SAMPLES:
            return (np.nan, {})

        t_xv = t_x[valid]
        t_reflect_v = t_reflect[valid]
        r_reflect_v = r_reflect_rs[valid]

        a_t = self._continuum_remove_and_invert(t_xv, t_reflect_v)
        a_r = self._continuum_remove_and_invert(t_xv, r_reflect_v)
        if not np.any(np.isfinite(a_r)) or np.allclose(a_r, 0.0, atol=1e-12):
            return (np.nan, {})

        num = float(np.dot(a_r, a_t))
        den = float(np.dot(a_r, a_r))
        if den <= 0:
            return (np.nan, {})
        scale = max(0.0, num / den)

        resid = a_t - scale * a_r
        rms = float(np.sqrt(np.nanmean(resid**2)))
        return rms, {"scale": float(scale)}
