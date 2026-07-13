"""
Non-GUI scene-ingestion pipeline for the Seamless Mosaic feature (issue #634).

When a user adds a scene to the mosaic, its materialized GDAL source must pass a
few gates before it can be displayed or composited:

  1. :func:`validate_scene` — reject scenes that are ungeoreferenced (identity
     geotransform or missing SRS) or whose band count disagrees with the existing
     scenes. Missing nodata and dtype mismatches are *deliberately not* rejected
     (see the function docstring).
  2. :func:`build_overviews` — build internal pyramid overviews on the materialized
     temp GeoTIFF so preview rendering (#637) is fast without a first-paint stutter.
  3. :func:`compute_stretch_bounds` — sample a stable, extent-independent contrast
     stretch per display band from an internal overview, so the pixel compositor's
     contrast (#637) does not drift as the user zooms (#675).
  4. :func:`compute_footprint_wkt` — derive the valid-pixel outline via
     ``gdal.Footprint`` so geometry rendering (#636) has the real shape.

This module is **Qt-free** (GDAL + stdlib only) so it is unit-testable without a
running application. The GUI orchestration (materialize -> build_overviews ->
compute_stretch_bounds -> compute_footprint on a background thread) lives in the
mosaic pane (#638-adjacent).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from osgeo import gdal

from wiser.raster.mosaic_compositor import STRETCH_HI_PCT, STRETCH_LO_PCT, select_display_bands
from wiser.utils.progress import ProgressReporter, gdal_progress_callback

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.mosaic_controller import MosaicScene

# GDAL's default when a source has no geotransform (mirrors RasterDataSet's identity
# sentinel in dataset_impl.read_geo_transform). An ungeoreferenced scene reports
# exactly this, so it is what the georef gate rejects.
_IDENTITY_GEO_TRANSFORM: Tuple[float, float, float, float, float, float] = (
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
)

# Pyramid decimation levels for preview overviews. Powers of two keep GDAL's
# averaging cheap and cover a wide zoom range for typical scene sizes.
_OVERVIEW_LEVELS: List[int] = [2, 4, 8, 16]

# Overview decimation level compute_stretch_bounds samples from: coarser than the
# finest overview (cheap -- no full-resolution read), but finer than the coarsest, so
# the sample is large enough that a small bright/dark feature isn't missed by the
# NEAREST-resampled overview's sparser pixels (#675).
_STRETCH_SAMPLE_OVERVIEW_LEVEL = 4


class SceneValidationError(ValueError):
    """
    Raised when a candidate scene cannot join the mosaic.

    Carries a human-readable :attr:`reason` so the GUI can surface it directly in a
    warning dialog without re-deriving a message.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _is_identity_geo_transform(geo_transform: Tuple[float, ...]) -> bool:
    """True if ``geo_transform`` is GDAL's identity sentinel (i.e. ungeoreferenced)."""
    if geo_transform is None or len(geo_transform) != len(_IDENTITY_GEO_TRANSFORM):
        # A malformed/absent transform is treated as "not georeferenced".
        return True
    return all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(geo_transform, _IDENTITY_GEO_TRANSFORM)
    )


def is_dataset_georeferenced(dataset: "RasterDataSet") -> bool:
    """
    True if ``dataset`` is placeable on a mosaic grid: it has a real (non-identity)
    geotransform **and** a non-empty spatial reference.

    This is the single definition of "georeferenced" shared by the strict
    :func:`validate_scene` gate and the mosaic's live/pending classification (the
    "pending scene" feature). A dataset that fails this cannot be materialized onto the
    common grid and is carried as a pending placeholder until it is georeferenced.
    Note ``get_geo_transform()`` never returns ``None`` (it returns the identity
    transform for ungeoreferenced data), so the geotransform check is identity-based.
    """
    return not _is_identity_geo_transform(dataset.get_geo_transform()) and bool(
        dataset.get_wkt_spatial_reference()
    )


def validate_scene_addable(dataset: "RasterDataSet", existing_scenes: List["MosaicScene"]) -> None:
    """
    Enforce only the gates that make a candidate **fundamentally un-addable**, regardless
    of whether it is georeferenced yet: a duplicate scene, or a band-count mismatch.

    Unlike :func:`validate_scene`, this deliberately does *not* reject an
    ungeoreferenced or CRS-incompatible dataset — those are added as disabled "pending"
    scenes (the mosaic pane branches on :func:`is_dataset_georeferenced`) and fixed later
    via the in-place georeferencer. Raises :class:`SceneValidationError` with a specific
    reason for each rejected case; returns ``None`` when the scene may be added.
    """
    if any(scene.dataset.get_id() == dataset.get_id() for scene in existing_scenes):
        raise SceneValidationError(f'"{dataset.get_name() or dataset.get_id()}" is already in the mosaic.')

    if existing_scenes:
        expected_bands = existing_scenes[0].dataset.num_bands()
        candidate_bands = dataset.num_bands()
        if candidate_bands != expected_bands:
            # TODO(#640): replace this hard rejection with a warn-but-allow path so
            # scenes with differing band counts can still be added intentionally.
            raise SceneValidationError(
                f"Scene has {candidate_bands} band(s) but the mosaic's scenes have "
                f"{expected_bands}. All scenes must share a band count."
            )


def validate_scene(dataset: "RasterDataSet", existing_scenes: List["MosaicScene"]) -> None:
    """
    Validate a candidate ``dataset`` against the mosaic's existing scenes (strict gate).

    Raises :class:`SceneValidationError` with a specific reason for each rejected
    case; returns ``None`` when the scene is acceptable.

    Rejected:
      * **Ungeoreferenced** — the geotransform is GDAL's identity sentinel. Note
        ``get_geo_transform()`` never returns ``None`` (it returns the identity
        transform for ungeoreferenced data), so the check is identity-based.
      * **No SRS** — ``get_wkt_spatial_reference()`` is empty/None.
      * **Duplicate / band-count mismatch** — see :func:`validate_scene_addable`.

    Deliberately NOT rejected:
      * **No nodata value** — many valid datasets lack one. nodata only sharpens the
        footprint; ``gdal.Footprint`` without it returns the full raster rectangle,
        which is a valid (coarser) footprint.
      * **Data-type mismatch across scenes** — no dtype consistency is required at
        ingestion. The compositor (#635/#637) promotes to the widest common type at
        warp time, so enforcing equal dtypes here would only cause false rejections.

    This remains the strict "must be immediately placeable" gate. The mosaic pane no
    longer calls it directly (it splits the georef check off so ungeoreferenced scenes
    can be added as pending), but it is kept as the canonical full check.
    """
    if _is_identity_geo_transform(dataset.get_geo_transform()):
        raise SceneValidationError(
            "Scene is not georeferenced (no geotransform); it cannot be placed on the " "mosaic grid."
        )

    if not dataset.get_wkt_spatial_reference():
        raise SceneValidationError(
            "Scene has no spatial reference system (SRS); it cannot be placed on the " "mosaic grid."
        )

    validate_scene_addable(dataset, existing_scenes)


def build_overviews(gdal_path: str, progress: Optional[ProgressReporter] = None) -> None:
    """
    Build internal pyramid overviews on the materialized GeoTIFF at ``gdal_path``.

    ``gdal_path`` must be the WISER-owned, writable temp copy produced by
    :class:`~wiser.raster.mosaic_materialize.SceneMaterializer` — never the user's
    original dataset (which is read-only and must stay untouched). Opening the temp
    copy ``GA_Update`` and calling ``BuildOverviews`` embeds the overviews *inside*
    that same ``.tif`` (internal overviews, no sidecar).

    ``progress`` is driven by GDAL's own ``BuildOverviews`` progress callback and
    defaults to a no-op reporter.
    """
    # If not ProgressReporter is supplied, this essentially becomes a no-op
    progress = progress or ProgressReporter()
    gdal.UseExceptions()
    ds = gdal.Open(gdal_path, gdal.GA_Update)
    if ds is None:
        raise RuntimeError(f"Cannot open {gdal_path} for overview building")
    try:
        ds.BuildOverviews("NEAREST", _OVERVIEW_LEVELS, callback=gdal_progress_callback(progress))
    finally:
        ds = None  # flush/close


def _stretch_sample_source(band: gdal.Band) -> gdal.Band:
    """
    Pick what :func:`compute_stretch_bounds` reads for one band: the
    ``_STRETCH_SAMPLE_OVERVIEW_LEVEL`` overview when it was built, else the coarsest
    overview available, else the full-resolution band (no overviews at all).
    """
    count = band.GetOverviewCount()
    if count == 0:
        return band
    if _STRETCH_SAMPLE_OVERVIEW_LEVEL in _OVERVIEW_LEVELS:
        target_index = _OVERVIEW_LEVELS.index(_STRETCH_SAMPLE_OVERVIEW_LEVEL)
        if target_index < count:
            return band.GetOverview(target_index)
    return band.GetOverview(count - 1)


def compute_stretch_bounds(
    gdal_path: str,
    dataset: "RasterDataSet",
    progress: Optional[ProgressReporter] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute stable, extent-independent 2-98 percentile stretch bounds per display
    band for the materialized scene at ``gdal_path``.

    Cached once at ingest (issue #675) so the pixel compositor's on-screen contrast
    no longer drifts as the user zooms: :func:`wiser.raster.mosaic_compositor.render_scene_argb`
    stretches every render against these fixed bounds instead of recomputing
    percentiles from whatever pixels happen to be in the current viewport. Samples
    from an internal overview (see :func:`_stretch_sample_source`) rather than a
    full-resolution read, since ``build_overviews`` -- which always runs immediately
    before this in the ingestion pipeline -- makes that cheap regardless of the
    source's full-resolution size.

    Returns a dict keyed by **source band index** (0-based, matching
    :func:`~wiser.raster.mosaic_compositor.select_display_bands`), so it stays correct
    regardless of channel ordering. A band with no valid (non-nodata) pixels in the
    sample is simply omitted; :func:`~wiser.raster.mosaic_compositor._stretch_band`
    falls back to an on-the-fly stretch for it.

    ``progress`` reports coarse per-band fractions and defaults to a no-op reporter.
    """
    progress = progress or ProgressReporter()
    gdal.UseExceptions()
    ds = gdal.Open(gdal_path)
    if ds is None:
        raise RuntimeError(f"Cannot open {gdal_path} for stretch-bounds computation")

    num_data_bands = ds.RasterCount
    display_bands = select_display_bands(dataset, num_data_bands)

    bounds: Dict[int, Tuple[float, float]] = {}
    for i, band_idx in enumerate(display_bands):
        band = ds.GetRasterBand(band_idx + 1)
        arr = _stretch_sample_source(band).ReadAsArray().astype(np.float64)
        nodata = band.GetNoDataValue()
        valid = arr != nodata if nodata is not None else np.ones(arr.shape, dtype=bool)
        values = arr[valid]
        if values.size:
            bounds[band_idx] = (
                float(np.percentile(values, STRETCH_LO_PCT)),
                float(np.percentile(values, STRETCH_HI_PCT)),
            )
        progress.report_fraction((i + 1) / len(display_bands), "Computing stretch bounds")
    return bounds


def compute_footprint_wkt(gdal_path: str, progress: Optional[ProgressReporter] = None) -> str:
    """
    Return the valid-pixel footprint of ``gdal_path`` as a WKT polygon.

    The polygon is in the dataset's own CRS (no reprojection here; that is
    #635/#636). When the source has a nodata value, the footprint traces the
    valid-pixel boundary; without one, ``gdal.Footprint`` returns the full raster
    rectangle — a valid, coarser footprint.

    ``progress`` is driven by GDAL's own ``Footprint`` progress callback and defaults
    to a no-op reporter.
    """
    progress = progress or ProgressReporter()
    gdal.UseExceptions()
    wkt = gdal.Footprint(None, gdal_path, format="WKT", callback=gdal_progress_callback(progress))
    if not wkt:
        raise RuntimeError(f"gdal.Footprint returned an empty result for {gdal_path}")
    return wkt
