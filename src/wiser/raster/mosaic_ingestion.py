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
  3. :func:`compute_footprint_wkt` — derive the valid-pixel outline via
     ``gdal.Footprint`` so geometry rendering (#636) has the real shape.

This module is **Qt-free** (GDAL + stdlib only) so it is unit-testable without a
running application. The GUI orchestration (materialize -> build_overviews ->
compute_footprint on a background thread) lives in the mosaic pane (#638-adjacent).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, TYPE_CHECKING

from osgeo import gdal

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


def validate_scene(dataset: "RasterDataSet", existing_scenes: List["MosaicScene"]) -> None:
    """
    Validate a candidate ``dataset`` against the mosaic's existing scenes.

    Raises :class:`SceneValidationError` with a specific reason for each rejected
    case; returns ``None`` when the scene is acceptable.

    Rejected:
      * **Ungeoreferenced** — the geotransform is GDAL's identity sentinel. This is
        a defensive gate against bad inputs. Note ``get_geo_transform()`` never
        returns ``None`` (it returns the identity transform for ungeoreferenced
        data), so the check is identity-based, not ``is None``.
      * **No SRS** — ``get_wkt_spatial_reference()`` is empty/None.
      * **Band-count mismatch** — the candidate's band count differs from the first
        existing scene's. See ``# TODO(#640)`` for the future warn-but-allow path.

    Deliberately NOT rejected:
      * **No nodata value** — many valid datasets lack one. nodata only sharpens the
        footprint; ``gdal.Footprint`` without it returns the full raster rectangle,
        which is a valid (coarser) footprint.
      * **Data-type mismatch across scenes** — no dtype consistency is required at
        ingestion. The compositor (#635/#637) promotes to the widest common type at
        warp time, so enforcing equal dtypes here would only cause false rejections.
    """
    if _is_identity_geo_transform(dataset.get_geo_transform()):
        raise SceneValidationError(
            "Scene is not georeferenced (no geotransform); it cannot be placed on the " "mosaic grid."
        )

    if not dataset.get_wkt_spatial_reference():
        raise SceneValidationError(
            "Scene has no spatial reference system (SRS); it cannot be placed on the " "mosaic grid."
        )

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
