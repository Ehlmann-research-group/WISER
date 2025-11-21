# Child class for Spectral Angle Mapper using the generic parent
from __future__ import annotations

import os
from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u

from wiser.raster.spectrum import NumPyArraySpectrum
from wiser.gui.app_state import ApplicationState
from .generic_spectral_tool import GenericSpectralComputationTool


class SAMTool(GenericSpectralComputationTool):
    SETTINGS_NAMESPACE = "Wiser/SAMPlugin"
    RUN_BUTTON_TEXT = "Run SAM"
    SCORE_HEADER = "Angle (°)"
    THRESHOLD_HEADER = "Initial Angle (°)"
    THRESHOLD_SPIN_CONFIG = dict(min=0.0, max=180.0, decimals=2, step=1.0)

    def __init__(self, app_state: ApplicationState, parent=None):
        self._threshold: float = 5.0  # metric-specific name as requested
        # initialize UI threshold spin to default
        super().__init__("Spectral Angle Mapper", app_state, parent)
        self._ui.method_threshold.setValue(self._threshold)
        self._maybe_add_default_library()

    # maintain metric-specific name while using parent's storage
    def set_method_threshold(self, value: float | None) -> None:
        self._threshold = float(value) if value is not None else self._threshold

    # child-specific column layout
    def details_columns(self) -> List[tuple]:
        # score already labeled as Angle (°) via SCORE_HEADER
        return [(self.SCORE_HEADER, "score"), ("Threshold (°)", "threshold")]

    def filename_stub(self) -> str:
        return "SAM"

    def default_library_path(self):
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(
            gui_dir,
            "../data/" "usgs_default_ref_lib",
            "USGS_Mineral_Spectral_Library.hdr",
        )
        return default_path

    # ---------- SAM helpers ----------
    @staticmethod
    def _resample_to(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
        """
        Resample y_src defined on x_src onto x_dst using linear interpolation.
        Returns NaNs when insufficient data or invalid inputs are detected.
        """
        if x_src.size < 2 or np.all(~np.isfinite(y_src)):
            return np.full_like(x_dst, np.nan, dtype=float)  # TODO (Joshua G-K) Raise error instead
        interp_fn = interp1d(x_src, y_src, bounds_error=False, fill_value="extrapolate")
        return interp_fn(x_dst)

    # compute spectral angle between target and ref in degrees
    def compute_score(
        self,
        target: NumPyArraySpectrum,
        ref: NumPyArraySpectrum,
        min_wvl: u.Quantity,
        max_wvl: u.Quantity,
    ) -> Tuple[float, Dict[str, Any]]:
        MIN_SAMPLES = 3
        t_arr, t_wls = self._slice_to_bounds(
            spectrum=target,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
        r_arr, r_wls = self._slice_to_bounds(
            spectrum=ref,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

        t_x = t_wls.value
        r_x = r_wls.value

        if r_x.shape == t_x.shape and np.allclose(r_x, t_x, rtol=0, atol=1e-9):
            r_resampled = r_arr
        else:
            r_resampled = self._resample_to(r_x, r_arr, t_x)

        valid = np.isfinite(t_arr) & np.isfinite(r_resampled)
        t_vec = t_arr[valid]
        r_vec = r_resampled[valid]
        if t_vec.size < MIN_SAMPLES:
            return (np.nan, {})

        denom = np.linalg.norm(t_vec) * np.linalg.norm(r_vec)
        if denom == 0:
            angle_deg = 90.0
        else:
            cosang = np.clip(np.dot(t_vec, r_vec) / denom, -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cosang)))

        return angle_deg, {}
