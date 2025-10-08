# Child class for Spectral Feature Fitting using the generic parent
from __future__ import annotations

import os
from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.interpolate import interp1d

from astropy import units as u
from wiser.raster.spectrum import NumPyArraySpectrum
from wiser.gui.app_state import ApplicationState
from .generic_spectral_tool import GenericSpectralComputationTool


class SFFTool(GenericSpectralComputationTool):
    SETTINGS_NAMESPACE = "Wiser/SFFPlugin"
    RUN_BUTTON_TEXT = "Run SFF"
    SCORE_HEADER = "Fit RMS"
    THRESHOLD_HEADER = "Initial RMSE"
    THRESHOLD_SPIN_CONFIG = dict(min=0.0, max=1.0, decimals=4, step=0.005)
    SPEC_THRESHOLD_ATTR = "_sff_max_rms"

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

    # compute SFF RMS + scale between self._target and ref
    def compute_score(self, ref: NumPyArraySpectrum) -> Tuple[float, Dict[str, Any]]:
        if self._target is None:
            raise RuntimeError("compute_score called without a target set")

        MIN_SAMPLES = 3
        t_reflect, t_wls = self._slice_to_bounds(self._target)
        r_reflect, r_wls = self._slice_to_bounds(ref)

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
