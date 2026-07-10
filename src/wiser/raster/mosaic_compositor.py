"""
Qt-free per-scene pixel renderer for the Seamless Mosaic feature (EPIC #629, issue
#637).

This is the read half of the mosaic's **pixel layer**: given one materialized scene
and the current viewport (a world rectangle in the target CRS plus an output pixel
size), it warps the scene onto that window at screen resolution and returns an
``(H, W, 4)`` uint8 **RGBA** array. The alpha channel is the scene's *validity mask*
-- ``0`` on nodata / outside-footprint pixels, ``255`` where the scene has real data
-- so the view can stack scenes with native alpha compositing (lower scenes show
through upper scenes' holes).

Kept Qt-free (GDAL/NumPy only, like :mod:`wiser.raster.mosaic_controller`) so it is
unit-testable without a running application and can run on a background thread: it
produces a plain NumPy array, and the GUI side (:mod:`wiser.gui.mosaic_view`) wraps
that into a ``QImage`` on the main thread.

Why warp rather than a plain ``ReadAsArray``: a mosaic composes scenes onto one shared
target CRS, so each scene generally has to be reprojected. ``gdal.Warp`` handles the
reprojection, the downsampling (it reads from the internal overviews built in #634
because the output is far coarser than the source), the alignment to the requested
window, and -- via ``dstAlpha=True`` -- the validity mask, all in one call. Compare the
existing warp seam :func:`wiser.raster.mosaic_controller._warped_resolution`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
from osgeo import gdal

if TYPE_CHECKING:
    from wiser.raster.mosaic_controller import MosaicScene

# Percentile bounds for the per-scene linear contrast stretch. A 2-98% stretch is a
# common, robust default that ignores a few extreme outliers without needing the
# main view's live stretch state (this preview is intentionally self-contained).
# Public (not module-private) because wiser.raster.mosaic_ingestion.compute_stretch_bounds
# uses the same bounds when precomputing a scene's stable stretch (issue #675).
STRETCH_LO_PCT = 2.0
STRETCH_HI_PCT = 98.0


def render_scene_argb(
    scene: "MosaicScene",
    target_wkt: str,
    world_extent: Tuple[float, float, float, float],
    out_width: int,
    out_height: int,
    resample_alg: int = gdal.GRA_NearestNeighbour,
) -> np.ndarray:
    """
    Warp ``scene`` onto ``world_extent`` at ``out_width`` x ``out_height`` and return
    an ``(out_height, out_width, 4)`` uint8 RGBA array.

    ``world_extent`` is ``(min_x, min_y, max_x, max_y)`` in the target CRS (the
    view's visible world rectangle). RGB comes from the dataset's default display
    bands (1 band -> grayscale, 3 bands -> RGB), contrast-stretched per band against
    ``scene.stretch_bounds`` -- precomputed, extent-independent 2-98 percentile bounds
    cached at ingest (issue #675), so contrast does not drift with zoom -- falling back
    to a viewport percentile stretch when a scene has no cached bounds. Alpha is the
    warp's validity mask (0 on nodata / outside coverage).

    A scene whose data does not cover ``world_extent`` comes back fully transparent
    (alpha all 0), so callers can render it unconditionally.
    """
    if out_width <= 0 or out_height <= 0:
        return np.zeros((max(out_height, 0), max(out_width, 0), 4), dtype=np.uint8)

    min_x, min_y, max_x, max_y = world_extent
    warped = gdal.Warp(
        "",
        scene.gdal_path,
        format="MEM",
        dstSRS=target_wkt,
        outputBounds=(min_x, min_y, max_x, max_y),
        width=out_width,
        height=out_height,
        dstAlpha=True,
        resampleAlg=resample_alg,
        multithread=True,
    )
    if warped is None:
        # Warp can decline (e.g. an output window fully disjoint from the source);
        # treat that as "this scene contributes nothing here".
        return np.zeros((out_height, out_width, 4), dtype=np.uint8)

    # The alpha band dstAlpha appended is always the last band; the data bands are the
    # ones before it (the scene's own bands, in order).
    alpha_index = warped.RasterCount
    num_data_bands = alpha_index - 1
    alpha = warped.GetRasterBand(alpha_index).ReadAsArray().astype(np.uint8)

    rgba = np.zeros((out_height, out_width, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha

    valid = alpha > 0
    if num_data_bands <= 0 or not valid.any():
        # No data bands, or nothing valid in view: leave RGB black, alpha as computed.
        return rgba

    display_bands = select_display_bands(scene.dataset, num_data_bands)
    channels = [
        _stretch_band(
            warped.GetRasterBand(band_idx + 1).ReadAsArray().astype(np.float32),
            valid,
            bounds=scene.stretch_bounds.get(band_idx) if scene.stretch_bounds else None,
        )
        for band_idx in display_bands
    ]

    if len(channels) == 1:  # grayscale: replicate into R=G=B
        rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = channels[0]
    else:
        rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2] = channels[0], channels[1], channels[2]
    # Force invalid pixels fully clear (0, 0, 0, 0) so RGB never bleeds under a
    # transparent alpha (the flat-band stretch otherwise whitens them).
    rgba[~valid, :3] = 0
    return rgba


def select_display_bands(dataset, num_data_bands: int) -> Sequence[int]:
    """
    Pick the 0-based source band indices to render as RGB (or 1 for grayscale).

    Prefers the dataset's default display bands; falls back to the first three bands
    (or the single band) and clamps any out-of-range index defensively.

    Takes ``dataset`` directly (rather than a :class:`MosaicScene`) so
    :func:`wiser.raster.mosaic_ingestion.compute_stretch_bounds` can pick the same
    display bands at ingest time, before a ``MosaicScene`` exists (issue #675).
    """
    bands = None
    getter = getattr(dataset, "get_default_display_bands", None)
    if getter is not None:
        try:
            bands = getter()
        except Exception:  # noqa: BLE001 - metadata access must never break a paint
            bands = None

    if not bands:
        bands = (0,) if num_data_bands < 3 else (0, 1, 2)
    # Grayscale if a single band was chosen or the scene only has one band.
    if len(bands) == 1 or num_data_bands < 3:
        return (min(int(bands[0]), num_data_bands - 1),)
    return tuple(min(int(b), num_data_bands - 1) for b in bands[:3])


def _stretch_band(
    band: np.ndarray, valid: np.ndarray, bounds: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Linearly stretch ``band`` to uint8 [0, 255] using ``bounds``, or the 2-98
    percentile of its valid pixels when ``bounds`` is ``None``. A flat/degenerate
    range (``hi <= lo``) maps to full white so it stays visible.

    ``bounds`` is the scene's precomputed, extent-independent stretch (see
    :func:`wiser.raster.mosaic_ingestion.compute_stretch_bounds`, issue #675) — passing
    it decouples the on-screen contrast from whatever pixels happen to be in the
    current viewport. ``None`` is the fallback for a scene with no cached bounds (e.g.
    a :class:`MosaicScene` built directly in a test, bypassing ingestion), which
    reproduces the pre-#675 viewport-percentile behavior.
    """
    if bounds is not None:
        lo, hi = bounds
    else:
        values = band[valid]
        lo = float(np.percentile(values, STRETCH_LO_PCT))
        hi = float(np.percentile(values, STRETCH_HI_PCT))
    if hi <= lo:
        out = np.full(band.shape, 255.0, dtype=np.float32)
    else:
        out = (band - lo) * (255.0 / (hi - lo))
    np.clip(out, 0.0, 255.0, out=out)
    return out.astype(np.uint8)
