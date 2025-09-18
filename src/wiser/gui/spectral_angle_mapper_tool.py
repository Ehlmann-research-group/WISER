# Child class for Spectral Angle Mapper using the generic parent
from __future__ import annotations

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
    THRESHOLD_SPIN_CONFIG = dict(min=0.0, max=180.0, decimals=2, step=1.0)
    SPEC_THRESHOLD_ATTR = "_sam_threshold"

    def __init__(self, app_state: ApplicationState, parent=None):
        self._threshold: float = 5.0  # metric-specific name as requested
        super().__init__(app_state, parent)
        # initialize UI threshold spin to default
        self._ui.method_threshold.setValue(self._threshold)

    # maintain metric-specific name while using parent's storage
    def set_method_threshold(self, value: float | None) -> None:
        self._threshold = float(value) if value is not None else self._threshold

    # child-specific column layout
    def details_columns(self) -> List[tuple]:
        # score already labeled as Angle (°) via SCORE_HEADER
        return [(self.SCORE_HEADER, "score"), ("Threshold (°)", "threshold")]

    def filename_stub(self) -> str:
        return "SAM"

    # compute spectral angle between target and ref in degrees
    def compute_score(self, ref: NumPyArraySpectrum) -> Tuple[float, Dict[str, Any]]:
        if self._target is None:
            raise RuntimeError("compute_score called without a target set")

        MIN_SAMPLES = 3
        t_arr, t_wls = self._slice_to_bounds(self._target)
        r_arr, r_wls = self._slice_to_bounds(ref)

        t_x = t_wls.value
        r_x = r_wls.value

        if np.array_equal(r_x, t_x):
            r_resampled = r_arr
        else:
            interp_fn = interp1d(r_x, r_arr, bounds_error=False, fill_value=np.nan)
            r_resampled = interp_fn(t_x)

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
