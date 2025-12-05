# Child class for Spectral Angle Mapper using the generic parent
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
    compute_image_norm,
    compute_image_norm_numba,
)


def compute_sam_image(
    target_image_arr: np.ndarray,
    target_wavelengths: np.ndarray,
    target_bad_bands: np.ndarray,
    min_wvl: np.float32,
    max_wvl: np.float32,
    reference_spectra: np.ndarray,
    reference_spectra_wvls: np.ndarray,
    reference_spectra_bad_bands: np.ndarray,
    reference_spectra_indices: np.ndarray,
    thresholds: np.ndarray,
):
    """
    Python/NumPy method of the function compute_sam_image_numba
    """
    if (
        reference_spectra_wvls.shape[0] != reference_spectra_bad_bands.shape[0]
        or reference_spectra.shape[0] != reference_spectra_wvls.shape[0]
    ):
        raise ValueError("Shape mismatch in reference spectra and wavelengths/bad bands.")

    # Slice the target image array, wvls, and bad bands in the bands dimension to get
    # into the user specified range
    target_image_arr_sliced, _, target_bad_bands_sliced = slice_to_bounds_3D(
        target_image_arr,
        target_wavelengths,
        target_bad_bands,
        min_wvl,
        max_wvl,
    )
    # Now slice the bad bands out of the array
    target_image_arr_sliced = target_image_arr_sliced[target_bad_bands_sliced, :, :]
    if not np.isfinite(target_image_arr_sliced).all():
        raise ValueError("Target image array is not finite after cleaning")
    if not np.isfinite(reference_spectra[reference_spectra_bad_bands]).all():
        raise ValueError("Reference spectra array is not finite")

    num_spectra = reference_spectra_indices.shape[0] - 1
    out_classification = np.empty(
        (
            num_spectra,
            target_image_arr_sliced.shape[1],
            target_image_arr_sliced.shape[2],
        ),
        dtype=np.bool_,
    )
    out_angle = np.empty(
        (
            num_spectra,
            target_image_arr_sliced.shape[1],
            target_image_arr_sliced.shape[2],
        ),
        dtype=np.float32,
    )

    # If we change dot3d_numba we could not transpose the array here
    target_image_arr_sliced = target_image_arr_sliced.transpose((1, 2, 0))  # [b][y][x] -> [y][x][b]

    for i in range(reference_spectra_indices.shape[0] - 1):
        # Extract desired spectra and slice to user specified wvl bounds
        start = reference_spectra_indices[i]
        end = reference_spectra_indices[i + 1]
        ref_spectrum = reference_spectra[start:end]
        ref_wvls = reference_spectra_wvls[start:end]
        ref_bad_bands = reference_spectra_bad_bands[start:end]

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
            ref_bad_bands_interp = ref_bad_bands_interp.astype(np.bool_)

        ref_spectrum_sliced, _, ref_bad_bands_sliced = slice_to_bounds_1D(
            ref_spectrum_interp,
            target_wavelengths,
            ref_bad_bands_interp,
            min_wvl,
            max_wvl,
        )

        # Create mask: 1 = finite, 0 = non-finite
        finite_mask = np.isfinite(ref_spectrum_sliced).astype(np.bool_)

        # AND with existing mask
        ref_bad_bands = ref_bad_bands_sliced & finite_mask

        ref_spectrum_good_bands = ref_spectrum_sliced[target_bad_bands_sliced]
        ref_bad_bands = ref_bad_bands_sliced[target_bad_bands_sliced]

        # Compute the angle
        ref_spectrum_good_bands[~ref_bad_bands] = 0.0
        ref_spec_norm = np.sqrt(
            (np.dot(ref_spectrum_good_bands, ref_spectrum_good_bands)),
        )
        target_image_arr_norm = compute_image_norm(
            target_image_arr=target_image_arr_sliced,
            ref_bad_bands=ref_bad_bands,
        )
        denom = target_image_arr_norm * ref_spec_norm
        dot_prod_out = dot3d(
            target_image_arr_sliced,
            ref_spectrum_good_bands,
            ref_bad_bands,
        )
        cosang = np.clip(
            dot_prod_out / denom,
            -1.0,
            1.0,
        )
        angle = np.degrees(np.arccos(cosang))
        out_angle[i, :, :] = angle

        thr = thresholds[i]
        out_classification[i, :, :] = angle < thr

    return out_classification, out_angle


compute_sam_image_sig = types.Tuple(
    (
        types.boolean[:, :, :],
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
    non_njit_func=compute_sam_image,
    signature=compute_sam_image_sig,
    parallel=True,
    cache=True,
)
def compute_sam_image_numba(
    target_image_arr: np.ndarray,
    target_wavelengths: np.ndarray,
    target_bad_bands: np.ndarray,
    min_wvl: np.float32,
    max_wvl: np.float32,
    reference_spectra: np.ndarray,
    reference_spectra_wvls: np.ndarray,
    reference_spectra_bad_bands: np.ndarray,
    reference_spectra_indices: np.ndarray,
    thresholds: np.ndarray,
):
    """Compute a Spectral Angle Mapper (SAM) classification and angle image using Numba.

    This function computes the Spectral Angle Mapper (SAM) between each pixel in a
    hyperspectral image and a set of reference spectra, interpolating the
    reference spectra to match the image wavelengths. It returns both a boolean
    classification mask (per reference spectrum) and the corresponding SAM angles
    (in degrees).

    The workflow is:

    1. Slice the target image, wavelengths, and bad-band mask to the user-specified
       wavelength range ``[min_wvl, max_wvl]``.
    2. Remove bands marked as bad in ``target_bad_bands`` from the image.
    3. For each reference spectrum:
       * Slice to the same wavelength range.
       * Optionally interpolate to match ``target_wavelengths`` if grids differ.
       * Apply the target image's bad-band mask to the interpolated spectrum.
       * Compute SAM:
         - Normalize the target spectra and reference spectrum.
         - Compute the dot product and angle (in degrees).
       * Threshold the angle to produce a boolean classification map.

    The function is intended to be JIT-compiled with Numba for efficient per-pixel
    computation over large hyperspectral images.

    Args:
        target_image_arr (np.ndarray):
            3D float32 array of shape ``(bands, rows, cols)`` containing the target
            hyperspectral image data. Bands correspond to ``target_wavelengths``.
        target_wavelengths (np.ndarray):
            1D float32 array of shape ``(bands,)`` containing the wavelengths
            associated with ``target_image_arr``. Must be strictly increasing.
        target_bad_bands (np.ndarray):
            1D boolean array of shape ``(bands,)`` indicating which bands in the
            target image are considered bad (``True`` for bad bands). These bands
            are removed before SAM computation.
        min_wvl (np.float32):
            Minimum wavelength (inclusive) of the spectral range to use, in the
            same units as ``target_wavelengths``.
        max_wvl (np.float32):
            Maximum wavelength (inclusive) of the spectral range to use, in the
            same units as ``target_wavelengths``.
        reference_spectra (np.ndarray):
            1D float32 array containing one or more concatenated reference spectra.
            Individual spectra are segmented using ``reference_spectra_indices``.
        reference_spectra_wvls (np.ndarray):
            1D float32 array of the same length as ``reference_spectra`` giving
            the wavelength associated with each reference spectrum sample, in the
            same units as ``target_wavelengths``.
        reference_spectra_bad_bands (np.ndarray):
            1D boolean array of the same length as ``reference_spectra`` indicating
            bad samples in the reference spectra (``True`` for bad samples).
        reference_spectra_indices (np.ndarray):
            1D uint32 array of shape ``(num_spectra + 1,)`` giving start and end
            indices into ``reference_spectra`` (and companion arrays) for each
            reference spectrum. For spectrum ``i``, the slice is
            ``reference_spectra[reference_spectra_indices[i]:reference_spectra_indices[i+1]]``.
        thresholds (np.ndarray):
            1D float32 array of shape ``(num_spectra,)`` specifying SAM angle
            thresholds (in degrees) for each reference spectrum. Angles strictly
            less than the threshold are marked as matches in the output
            classification mask.

    Returns:
        Tuple(np.ndarray, np.ndarray):
            A tuple ``(out_classification, out_angle)`` where

            - ``out_classification`` is a 3D boolean array of shape
              ``(num_spectra, rows, cols)``. Element
              ``out_classification[i, y, x]`` is ``True`` if the SAM angle between
              pixel ``(y, x)`` and reference spectrum ``i`` is less than
              ``thresholds[i]``.
            - ``out_angle`` is a 3D float32 array of shape
              ``(num_spectra, rows, cols)`` containing the SAM angle (in degrees)
              between each pixel and each reference spectrum. Element
              ``out_angle[i, y, x]`` is the SAM angle for pixel ``(y, x)`` and
              reference spectrum ``i``.

    Raises:
        ValueError:
            If the shapes of ``reference_spectra``, ``reference_spectra_wvls``, and
            ``reference_spectra_bad_bands`` are inconsistent.
        ValueError:
            If ``target_image_arr_sliced`` contains non-finite values after slicing
            and bad-band removal.
        ValueError:
            If any of the reference spectra entries marked as good
            (``reference_spectra_bad_bands == False``) contain non-finite values.

    Notes:
        This function is wrapped by ``numba_njit_wrapper`` and compiled with
        Numba's ``nopython`` mode and parallel execution enabled. When Numba is
        not available, the non-JIT fallback function ``compute_sam_image`` is
        used instead.

    """
    if (
        reference_spectra_wvls.shape[0] != reference_spectra_bad_bands.shape[0]
        or reference_spectra.shape[0] != reference_spectra_wvls.shape[0]
    ):
        raise ValueError("Shape mismatch in reference spectra and wavelengths/bad bands.")

    # Slice the target image array, wvls, and bad bands in the bands dimension to get
    # into the user specified range
    target_image_arr_sliced, _, target_bad_bands_sliced = slice_to_bounds_3D_numba(
        target_image_arr,
        target_wavelengths,
        target_bad_bands,
        min_wvl,
        max_wvl,
    )
    # Now slice the bad bands out of the array
    target_image_arr_sliced = target_image_arr_sliced[target_bad_bands_sliced, :, :]
    if not np.isfinite(target_image_arr_sliced).all():
        raise ValueError("Target image array is not finite after cleaning")
    if not np.isfinite(reference_spectra[reference_spectra_bad_bands]).all():
        raise ValueError("Reference spectra array is not finite")

    num_spectra = reference_spectra_indices.shape[0] - 1
    out_classification = np.empty(
        (
            num_spectra,
            target_image_arr_sliced.shape[1],
            target_image_arr_sliced.shape[2],
        ),
        dtype=np.bool_,
    )
    out_angle = np.empty(
        (
            num_spectra,
            target_image_arr_sliced.shape[1],
            target_image_arr_sliced.shape[2],
        ),
        dtype=np.float32,
    )

    # If we change dot3d_numba we could not transpose the array here
    target_image_arr_sliced = target_image_arr_sliced.transpose((1, 2, 0))  # [b][y][x] -> [y][x][b]

    for i in prange(reference_spectra_indices.shape[0] - 1):
        # Extract desired spectra and slice to user specified wvl bounds
        start = reference_spectra_indices[i]
        end = reference_spectra_indices[i + 1]
        ref_spectrum = reference_spectra[start:end]
        ref_wvls = reference_spectra_wvls[start:end]
        ref_bad_bands = reference_spectra_bad_bands[start:end]

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
            ref_bad_bands_interp = ref_bad_bands_interp.astype(np.bool_)

        ref_spectrum_sliced, _, ref_bad_bands_sliced = slice_to_bounds_1D_numba(
            ref_spectrum_interp,
            target_wavelengths,
            ref_bad_bands_interp,
            min_wvl,
            max_wvl,
        )

        # Create mask: 1 = finite, 0 = non-finite
        finite_mask = np.isfinite(ref_spectrum_sliced).astype(np.bool_)

        # AND with existing mask.
        # Technically we shouldn't have to do this because the datasets mask
        # should properly mask out the nan's. We only mask the finite values
        # on the reference bands because it is computationally inexpensive
        # and is defensive
        ref_bad_bands = ref_bad_bands_sliced & finite_mask

        ref_spectrum_good_bands = ref_spectrum_sliced[target_bad_bands_sliced]
        ref_bad_bands = ref_bad_bands_sliced[target_bad_bands_sliced]

        # Compute the angle
        ref_spectrum_good_bands[~ref_bad_bands] = 0.0
        ref_spec_norm = np.sqrt(
            (np.dot(ref_spectrum_good_bands, ref_spectrum_good_bands)),
        )
        target_image_arr_norm = compute_image_norm_numba(
            target_image_arr=target_image_arr_sliced,
            ref_bad_bands=ref_bad_bands,
        )

        denom = target_image_arr_norm * ref_spec_norm

        dot_prod_out = dot3d_numba(
            target_image_arr_sliced,
            ref_spectrum_good_bands,
            ref_bad_bands,
        )
        cosang = np.clip(
            dot_prod_out / denom,
            -1.0,
            1.0,
        )
        angle = np.degrees(np.arccos(cosang))
        out_angle[i, :, :] = angle

        thr = thresholds[i]
        out_classification[i, :, :] = angle < thr

    return out_classification, out_angle


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

    # SAM helpers
    @staticmethod
    def _resample_to(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
        """
        Resample y_src defined on x_src onto x_dst using linear interpolation.
        Returns NaNs when insufficient data or invalid inputs are detected.
        """
        if x_src.size < 2 or np.all(~np.isfinite(y_src)):
            raise ValueError("Insufficient data or non finite data values! Can't perform interpolation.")
        interp_fn = interp1d(x_src, y_src, bounds_error=False, fill_value=np.nan)
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
        target_unit = target.get_wavelength_units()
        target_wvls_arr = np.array([wvl.to(target_unit).value for wvl in target.get_wavelengths()])
        target_bad_bands = target.get_bad_bands()

        target_arr = target.get_spectrum()
        min_wvl_value = min_wvl.to(target_unit).value
        max_wvl_value = max_wvl.to(target_unit).value
        target_arr_sliced, _, target_bad_bands_sliced = slice_to_bounds_1D(
            spectrum_arr=target_arr,
            wvls=target_wvls_arr,
            bad_bands=target_bad_bands,
            min_wvl=min_wvl_value,
            max_wvl=max_wvl_value,
        )

        ref_arr = ref.get_spectrum()
        ref_wvls_arr = np.array([wvl.to(target_unit).value for wvl in ref.get_wavelengths()])
        ref_bad_bands = ref.get_bad_bands()

        if ref_wvls_arr.shape == target_wvls_arr.shape and np.allclose(
            ref_wvls_arr, target_wvls_arr, rtol=0, atol=1e-9
        ):
            r_resampled = ref_arr
            ref_bad_bands_resampled = ref_bad_bands
        else:
            r_resampled = self._resample_to(ref_wvls_arr, ref_arr, target_wvls_arr)
            ref_bad_bands_resampled = self._resample_to(
                ref_wvls_arr, ref_bad_bands.astype(np.float32), target_wvls_arr
            )
            ref_bad_bands_resampled[ref_bad_bands_resampled < 1.0] = 0.0
            ref_bad_bands_resampled = ref_bad_bands_resampled.astype(np.bool_)

        ref_arr_sliced, _, ref_bad_bands_sliced = slice_to_bounds_1D(
            spectrum_arr=r_resampled,
            wvls=target_wvls_arr,
            bad_bands=ref_bad_bands_resampled,
            min_wvl=min_wvl_value,
            max_wvl=max_wvl_value,
        )

        valid = np.isfinite(target_arr_sliced) & np.isfinite(ref_arr_sliced)
        valid &= ref_bad_bands_sliced
        valid &= target_bad_bands_sliced
        target_arr_valid = target_arr_sliced[valid]
        ref_arr_valid = ref_arr_sliced[valid]
        if target_arr_valid.size < MIN_SAMPLES:
            return (np.nan, {})

        denom = np.linalg.norm(target_arr_valid) * np.linalg.norm(ref_arr_valid)
        if denom == 0:
            angle_deg = 90.0
        else:
            cosang = np.clip(np.dot(target_arr_valid, ref_arr_valid) / denom, -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cosang)))

        return angle_deg, {}

    def compute_score_image(
        self,
        target_image_name: str,
        target_image_arr: np.ndarray,
        target_wavelengths: np.ndarray,
        target_bad_bands: np.ndarray,
        min_wvl: np.float32,
        max_wvl: np.float32,
        reference_spectra: List[Spectrum],
        reference_spectra_arr: np.ndarray,
        reference_spectra_wvls: np.ndarray,
        reference_spectra_bad_bands: np.ndarray,
        reference_spectra_indices: np.ndarray,
        thresholds: np.ndarray,
        python_mode: bool = False,
    ) -> List[int]:
        if not python_mode:
            out_classification, out_angle = compute_sam_image_numba(
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
            out_classification, out_angle = compute_sam_image(
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

        # Load in out_angle dataset
        out_angle_dataset = loader.dataset_from_numpy_array(out_angle)
        out_angle_dataset.set_name(
            self._app_state.unique_dataset_name(f"SAM Angle, Img: {target_image_name}"),
        )
        band_descriptions = []

        for i in range(0, len(reference_spectra)):
            spectrum_name = reference_spectra[i].get_name()
            band_descriptions.append(f"Spec: {spectrum_name}")

        out_angle_dataset.set_band_descriptions(band_descriptions)
        self._app_state.add_dataset(out_angle_dataset)

        # Load in out_classification_dataset
        out_cls_dataset = loader.dataset_from_numpy_array(out_classification)
        out_cls_dataset.set_name(self._app_state.unique_dataset_name(f"SAM CLS, Img: {target_image_name}"))

        band_descriptions = []
        for i in range(0, len(reference_spectra)):
            spectrum_name = reference_spectra[i].get_name()
            band_descriptions.append(f"Spec: {spectrum_name}")

        out_cls_dataset.set_band_descriptions(band_descriptions)
        self._app_state.add_dataset(out_cls_dataset)

        return [out_cls_dataset.get_id(), out_angle_dataset.get_id()]
