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

from typing import TYPE_CHECKING, Tuple

import numpy as np
from osgeo import gdal

if TYPE_CHECKING:
    from wiser.raster.mosaic_controller import MosaicScene

# Percentile bounds for the per-scene linear contrast stretch. A 2-98% stretch is a
# common, robust default that ignores a few extreme outliers without needing the
# main view's live stretch state (this preview is intentionally self-contained).
_STRETCH_LO_PCT = 2.0
_STRETCH_HI_PCT = 98.0


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
    view's visible world rectangle). ``scene.gdal_path`` is the **display-only**
    artifact (#677) whose bands already *are* the resolved display bands, so RGB is
    read straight from bands 1..N in order (1 band -> grayscale, 3 bands -> RGB) --
    the "which source bands" decision was made upstream at materialize time, not here.
    Each band is contrast-stretched over the valid pixels; alpha is the warp's
    validity mask (0 on nodata / outside coverage).

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

    # The display-only artifact holds exactly the display bands (1 or 3), so read them
    # all, in order -- band 1/2/3 are R/G/B. Cap at 3 defensively; a well-formed
    # display-only file never has more.
    channels = [
        _stretch_band(
            warped.GetRasterBand(band_idx + 1).ReadAsArray().astype(np.float32),
            valid,
        )
        for band_idx in range(min(num_data_bands, 3))
    ]

    if len(channels) == 1:  # grayscale (single display band): replicate into R=G=B
        rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = channels[0]
    else:  # bands are already R, G, B
        rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2] = channels[0], channels[1], channels[2]
    # Force invalid pixels fully clear (0, 0, 0, 0) so RGB never bleeds under a
    # transparent alpha (the flat-band stretch otherwise whitens them).
    rgba[~valid, :3] = 0
    return rgba


def _stretch_band(band: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Linearly stretch ``band`` to uint8 [0, 255] using the 2-98 percentile of its
    valid pixels. A flat band (no spread) maps to full white so it stays visible.
    """
    values = band[valid]
    lo = float(np.percentile(values, _STRETCH_LO_PCT))
    hi = float(np.percentile(values, _STRETCH_HI_PCT))
    if hi <= lo:
        out = np.full(band.shape, 255.0, dtype=np.float32)
    else:
        out = (band - lo) * (255.0 / (hi - lo))
    np.clip(out, 0.0, 255.0, out=out)
    return out.astype(np.uint8)
