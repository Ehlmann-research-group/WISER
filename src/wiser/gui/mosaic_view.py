"""
Custom view widget for the Seamless Mosaic feature (EPIC #629).

Scaffolding only (issue #633): this defines the widget shell and the two-layer
paint structure, but draws nothing beyond an empty background. It is a **sibling**
of :class:`~wiser.gui.rasterview.RasterView`, not a subclass: ``RasterView`` is
one-dataset -> one-QPixmap -> zoom-the-pixmap, whereas a mosaic is N scenes composited
onto one shared world grid with a vector overlay on top (see EPIC "alternatives
considered").

Later issues fill in the two layers, both drawn through the seams stubbed here:

  * #637 — the **pixel layer**: per-scene ARGB caches (alpha = validity) composited
    bottom-to-top by :meth:`composite`, fed from overviews at screen resolution.
  * #636 — the **vector overlay**: footprints, bounding box, and overlap highlight,
    drawn in world->screen coordinates on top of the pixel layer.
"""

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np
from osgeo import ogr

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .app_state import ApplicationState
from .util import get_painter

from wiser.raster.mosaic_compositor import render_scene_argb
from wiser.raster.mosaic_controller import (
    MosaicController,
    MosaicScene,
    compute_union_overlaps,
)
from wiser.utils.primitives import PriorityClass

if TYPE_CHECKING:
    from .app_services import AppServices
    from .mosaic_pane import MosaicPane

logger = logging.getLogger(__name__)

# Debounce window for pan/zoom pixel re-reads: coalesce a gesture's burst of paints
# into one background read once the camera settles for this long.
_PIXEL_READ_DEBOUNCE_MS = 80

# Tile-based pixel cache (#674). World space is quantized into a fixed grid of ARGB
# tiles per discrete zoom bucket; a pan/zoom reuses already-cached tiles and only warps
# the newly-exposed ones, so the loaded margin travels with the user without re-warping
# the whole viewport each time.
#
#  * _TILE_PX        — the output size (px, square) each tile is warped at.
#  * _TILE_CACHE_MAX — LRU bound on the number of cached tiles (~256 tiles of 256x256
#                      ARGB ~= 64 MB). Tune alongside the debounce interval.
#  * _PREFETCH_RING  — how many extra tile rings around the viewport to read, so the
#                      loaded margin extends past the visible edge in every direction.
_TILE_PX = 256
_TILE_CACHE_MAX = 256
_PREFETCH_RING = 1

# Nudge the exclusive upper edge off an exact tile boundary so a rectangle whose max
# lands precisely on a cell border does not pull in the next (empty) cell.
_TILE_EPS = 1e-9


def _zoom_bucket(world_units_per_pixel: float) -> int:
    """
    Snap a continuous camera scale to a discrete power-of-two zoom bucket.

    The camera's ``world_units_per_pixel`` (wupp) is continuous, but caching tiles at
    every exact wupp would make the tiniest zoom invalidate the whole cache. Instead we
    render tiles at the nearest power-of-two resolution ``bucket_wupp = 2 ** bucket`` and
    let the world->screen affine scale them to the actual zoom (the standard slippy-map
    look), so a whole range of zooms shares one set of cached tiles.

    ``round`` keeps the draw scale ``bucket_wupp / wupp`` within ``[1/sqrt2, sqrt2]``
    (~[0.71, 1.41]); it is defined purely in terms of wupp, so it is CRS-agnostic
    (degrees or metres, no per-CRS constant). Non-finite/non-positive scales fall back
    to bucket 0.
    """
    if not math.isfinite(world_units_per_pixel) or world_units_per_pixel <= 0.0:
        return 0
    return round(math.log2(world_units_per_pixel))


def _bucket_wupp(bucket: int) -> float:
    """The quantized render resolution (world units per pixel) for a zoom ``bucket``."""
    return 2.0**bucket


def _tile_world_extent(bucket: int, col: int, row: int) -> Tuple[float, float, float, float]:
    """
    The world rectangle ``(min_x, min_y, max_x, max_y)`` a ``(col, row)`` cell covers.

    Tiles are ``_TILE_PX`` pixels square at the bucket's resolution, so a cell spans
    ``tws = _TILE_PX * bucket_wupp`` world units, anchored at the CRS's real ``(0, 0)``
    origin. The anchoring makes the grid stable across pans at a fixed bucket, so the
    same world area always maps to the same cell (hence a cache hit after a pan).
    """
    tws = _TILE_PX * _bucket_wupp(bucket)
    return (col * tws, row * tws, (col + 1) * tws, (row + 1) * tws)


def _tiles_covering_extent(
    bucket: int, world_extent: Tuple[float, float, float, float], ring: int = 0
) -> List[Tuple[int, int]]:
    """
    The ``(col, row)`` cells whose tiles cover ``world_extent`` at ``bucket``, plus an
    optional ``ring`` of extra cells on every side (the prefetch margin).

    ``world_extent`` is ``(min_x, min_y, max_x, max_y)`` in the target CRS. Returns the
    cells in row-major order; an empty/degenerate extent yields no cells.
    """
    min_x, min_y, max_x, max_y = world_extent
    if max_x <= min_x or max_y <= min_y:
        return []
    tws = _TILE_PX * _bucket_wupp(bucket)
    col_lo = math.floor(min_x / tws) - ring
    col_hi = math.floor((max_x - _TILE_EPS) / tws) + ring
    row_lo = math.floor(min_y / tws) - ring
    row_hi = math.floor((max_y - _TILE_EPS) / tws) + ring
    return [(c, r) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]


# Identity of one cached tile: (id(scene), zoom bucket, tile col, tile row). Used as the
# opaque key threaded through the warp worker and back so the GUI thread knows which cell
# a returned RGBA array belongs to.
_TileKey = Tuple[int, int, int, int]


def _render_scene_layers(
    jobs: List[Tuple[_TileKey, MosaicScene, str, Tuple[float, float, float, float], int, int, int]],
) -> List[Tuple[_TileKey, np.ndarray]]:
    """
    Warp each job's tile into an RGBA array. Runs on a **scheduler thread** (no Qt).

    ``jobs`` is ``(tile_key, scene, target_wkt, world_extent, width, height,
    resample_alg)``; the result pairs each key with its ``(H, W, 4)`` RGBA array for
    the GUI thread to wrap into a ``QImage``. A single tile that fails to render is
    skipped rather than failing the whole batch (mirrors the per-scene guard in the
    synchronous path).
    """
    out: List[Tuple[_TileKey, np.ndarray]] = []
    for key, scene, target_wkt, world_extent, width, height, resample_alg in jobs:
        try:
            out.append((key, render_scene_argb(scene, target_wkt, world_extent, width, height, resample_alg)))
        except Exception:  # noqa: BLE001 — a bad scene must not break the batch
            pass
    return out


# Vector-overlay styling (#636), matching the ENVI reference: green footprint
# outlines, a dashed bounding box, and a magenta/purple overlap highlight. Pens are
# cosmetic so their width stays constant in screen pixels under the world->screen
# zoom (which would otherwise scale line width along with the geometry).
_FOOTPRINT_PEN = QPen(QColor(0, 200, 0), 2)
_FOOTPRINT_PEN.setCosmetic(True)
_BBOX_PEN = QPen(QColor(230, 230, 230), 1, Qt.DashLine)
_BBOX_PEN.setCosmetic(True)
_OVERLAP_PEN = QPen(QColor(200, 0, 200), 2)
_OVERLAP_PEN.setCosmetic(True)
_OVERLAP_BRUSH = QBrush(QColor(200, 0, 200, 80))  # translucent purple fill

# Multiplicative zoom per wheel notch (120 eighths-of-a-degree = one notch).
_WHEEL_ZOOM_STEP = 1.25


def _geometry_to_qpainterpath(geom: ogr.Geometry) -> QPainterPath:
    """
    Convert an OGR polygonal geometry to a ``QPainterPath`` in **world** coordinates
    (no view transform applied — that happens per-paint).

    Handles ``POLYGON`` (with interior rings as holes via the odd-even fill rule) and
    ``MULTIPOLYGON`` / ``GEOMETRYCOLLECTION``; non-polygonal members (a point/line
    from an edge-touching :meth:`Intersection`) are skipped so nothing degenerate is
    drawn.
    """
    path = QPainterPath()
    path.setFillRule(Qt.OddEvenFill)  # interior rings subtract as holes

    def add_polygon(poly: ogr.Geometry) -> None:
        for r in range(poly.GetGeometryCount()):
            ring = poly.GetGeometryRef(r)
            pts = [QPointF(x, y) for x, y, *_ in ring.GetPoints()]
            if pts:
                path.addPolygon(QPolygonF(pts))

    gtype = geom.GetGeometryType()
    if gtype in (ogr.wkbPolygon, ogr.wkbPolygon25D):
        add_polygon(geom)
    else:  # MultiPolygon / GeometryCollection: take only the polygonal members
        for g in range(geom.GetGeometryCount()):
            sub = geom.GetGeometryRef(g)
            if sub.GetGeometryType() in (ogr.wkbPolygon, ogr.wkbPolygon25D):
                add_polygon(sub)
    return path


@dataclass
class MosaicViewTransform:
    """
    Camera for the QGIS-style unbounded mosaic canvas (#636).

    The affine *itself* is just ``QTransform`` (6 floats), rebuilt fresh every paint.
    The only state that must persist between paints is this camera: a world-space
    center point plus a scale. There is no invented ``(0, 0)`` canvas origin — world
    coordinates come from the common/target CRS (e.g. UTM metres, degrees) and screen
    coordinates are Qt's usual widget space. The visible world rectangle is *derived*
    from ``center + scale + viewport size`` at paint time and never stored (storing it
    would distort the aspect ratio on resize).

    Nothing clamps ``center_x`` / ``center_y``: pan the camera arbitrarily far from
    every footprint and the canvas simply shows blank space, which is what makes it
    "unbounded" (as opposed to :class:`~wiser.gui.rasterview.RasterView`'s
    scroll-area-over-a-fixed-``QPixmap``, which is bounded by the pixmap's size).

    ``world_to_screen`` / ``screen_to_world`` take the viewport size as an argument
    rather than caching it, since ``QWidget`` already tracks its own size — this keeps
    a single source of truth and dodges a stale second copy. Shared with the pixel
    layer (#637), which is why #636 builds it.
    """

    center_x: float = 0.0  # world coordinates
    center_y: float = 0.0
    world_units_per_pixel: float = 1.0

    def fit_to_extent(self, extent: Tuple[float, float, float, float], viewport_size: QSize) -> None:
        """Center on and frame a world ``(min_x, min_y, max_x, max_y)`` extent."""
        min_x, min_y, max_x, max_y = extent
        self.center_x = (min_x + max_x) / 2.0
        self.center_y = (min_y + max_y) / 2.0
        self.world_units_per_pixel = max(
            (max_x - min_x) / max(viewport_size.width(), 1),
            (max_y - min_y) / max(viewport_size.height(), 1),
            1e-12,  # never zero, even for a degenerate single-point extent
        )

    def pan(self, dx_pixels: float, dy_pixels: float) -> None:
        """Shift the camera by a screen-pixel delta (screen y-down, world y-up)."""
        self.center_x -= dx_pixels * self.world_units_per_pixel
        self.center_y += dy_pixels * self.world_units_per_pixel

    def zoom(self, factor: float, anchor_pixel: QPointF, viewport_size: QSize) -> None:
        """Zoom by ``factor`` while keeping the world point under ``anchor_pixel`` fixed."""
        before = self.screen_to_world(anchor_pixel, viewport_size)
        self.world_units_per_pixel /= factor
        after = self.screen_to_world(anchor_pixel, viewport_size)
        self.center_x += before.x() - after.x()
        self.center_y += before.y() - after.y()

    def world_to_screen(self, viewport_size: QSize) -> QTransform:
        """Build the world->screen affine for the current camera and viewport."""
        s = 1.0 / self.world_units_per_pixel  # screen pixels per world unit
        vw, vh = viewport_size.width(), viewport_size.height()
        ox = self.center_x - (vw / 2.0) / s  # world x at the widget's left edge
        oy = self.center_y + (vh / 2.0) / s  # world y at the widget's top edge
        # m11=s, m22=-s => uniform scale with a y-flip (world y up, screen y down).
        return QTransform(s, 0.0, 0.0, -s, -s * ox, s * oy)

    def screen_to_world(self, pt: QPointF, viewport_size: QSize) -> QPointF:
        """Map a screen point back to world coordinates (inverse of the affine)."""
        inverted, ok = self.world_to_screen(viewport_size).inverted()
        if not ok:
            return QPointF(self.center_x, self.center_y)
        return inverted.map(pt)


class MosaicView(QWidget):
    """
    Renders a mosaic composition: a composited pixel image with a vector overlay.

    Behavior-free in this issue. Holds a reference to the non-GUI
    :class:`MosaicController` (the source of truth for scenes, z-order, and the
    common grid) and exposes the paint/compositing seams that #636 and #637 target.
    """

    # Delivers a completed background pixel read from the scheduler thread to the GUI
    # thread. Emitted from the future's done-callback (worker thread); the queued
    # connection runs :meth:`_apply_pixel_read` on the GUI thread. Payload is
    # ``(signature, world_extent, read_scene_ids, [(scene_key, rgba_ndarray), ...])``.
    _read_ready = Signal(object)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        app_state: Optional[ApplicationState] = None,
        controller: Optional[MosaicController] = None,
        mosaicpane: Optional["MosaicPane"] = None,
        app_services: Optional["AppServices"] = None,
    ) -> None:
        super().__init__(parent=parent)
        self._app_state = app_state
        self._controller = controller if controller is not None else MosaicController()
        self._mosaicpane = mosaicpane
        self._app_services = app_services

        # Pixel layer (#674): a tiled ARGB cache. World space is quantized into a fixed
        # grid of _TILE_PX-square tiles per discrete zoom bucket; each cached tile is one
        # scene's warp of one cell (keyed by _TileKey = (id(scene), bucket, col, row)).
        # A pan/zoom reuses already-cached tiles and only warps the newly-exposed ones,
        # so the loaded margin travels with the user instead of the whole viewport being
        # re-warped on every settle (fixes the black pan edge, #674).
        #
        # _tile_cache is an LRU (insertion-ordered, oldest first) bounded to
        # _TILE_CACHE_MAX. _inflight_tiles holds keys currently being warped so a tile is
        # never submitted twice. _read_epoch bumps whenever the cache is invalidated
        # (add/remove scene, CRS change) so a read submitted before the invalidation is
        # dropped instead of inserting now-stale pixels.
        self._tile_cache: "OrderedDict[_TileKey, QImage]" = OrderedDict()
        self._inflight_tiles: Set[_TileKey] = set()
        self._read_epoch = 0
        self._pixels_dirty = True
        # Camera identity at the last paint, so pan/zoom (which moves the camera) arms a
        # debounced read while a pure restack (reorder/visibility) does not.
        self._last_paint_signature: Optional[Tuple[float, float, float, int, int]] = None

        # Off-thread + debounced tile reads (#674). Pan/zoom paints coalesce into one
        # background read once the camera settles for _PIXEL_READ_DEBOUNCE_MS. Reads are
        # submitted to the scheduler (when available); cached tiles (including a nearest
        # other-bucket fallback) keep drawing in the interim so there is no black frame.
        self._pending_future = None
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(_PIXEL_READ_DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._start_pixel_read)
        self._read_ready.connect(self._apply_pixel_read)

        # Camera for the QGIS-style unbounded canvas (#636). The view frames the
        # mosaic extent once, the first time a grid is available (see paintEvent);
        # after that the user drives it via pan/zoom and it is never auto-refit.
        self._transform = MosaicViewTransform()
        self._has_fitted = False

        # Vector-overlay geometry cache (#636), all in world coordinates. Rebuilt
        # only when scenes / z-order / target CRS change (invalidate_overlay), never
        # on pan/zoom. _footprint_paths and _hidden_paths are parallel, bottom-to-top.
        self._geometry_dirty = True
        self._bbox_extent: Optional[Tuple[float, float, float, float]] = None
        self._footprint_paths: List[QPainterPath] = []
        self._hidden_paths: List[Optional[QPainterPath]] = []

        # Middle-button drag-to-pan state (anchor pixel of the last move, or None).
        self._pan_anchor: Optional[QPointF] = None

    def get_controller(self) -> MosaicController:
        return self._controller

    def set_controller(self, controller: MosaicController) -> None:
        self._controller = controller
        self.invalidate_overlay()

    # -- overlay geometry cache (#636) ----------------------------------------

    def invalidate_overlay(self) -> None:
        """
        Mark the vector-overlay geometry stale and schedule a repaint.

        Called by :class:`~wiser.gui.mosaic_pane.MosaicPane` after any controller
        mutation that changes what the overlay draws — a scene added/removed, a
        visibility toggle, a z-order move, or a target-CRS change. Pan/zoom does
        **not** call this: the cached world-space paths are unchanged, only the
        world->screen transform differs, so a plain repaint suffices there.
        """
        self._geometry_dirty = True
        self.update()

    def _rebuild_overlay_geometry(self) -> None:
        """
        Rebuild the world-space overlay paths from the controller.

        Runs on the GUI thread from :meth:`paintEvent` when the cache is dirty (so
        repeated invalidations before the next paint coalesce into one rebuild). The
        body is fully guarded: any OGR/OSR failure clears the cache and logs, rather
        than letting an exception escape a paint. Only fires on discrete edits, not
        on pan/zoom, so the synchronous GDAL/OSR work here is not on the hot path.
        """
        self._bbox_extent = None
        self._footprint_paths = []
        self._hidden_paths = []
        try:
            grid = self._controller.get_common_grid()
            self._bbox_extent = grid.extent  # may be None until the grid is built

            footprints = self._controller.visible_scene_footprints_in_common_crs()
            if not footprints:
                return
            hidden = compute_union_overlaps(footprints)
            for i, (_scene, geom) in enumerate(footprints):
                self._footprint_paths.append(_geometry_to_qpainterpath(geom))
                overlap = hidden.get(i)
                self._hidden_paths.append(_geometry_to_qpainterpath(overlap) if overlap is not None else None)
        except Exception:  # noqa: BLE001 — never let a repaint raise
            logger.exception("Failed to rebuild mosaic overlay geometry")
            self._bbox_extent = None
            self._footprint_paths = []
            self._hidden_paths = []

    # -- pixel layer cache (#637) ---------------------------------------------

    def invalidate_pixels(self) -> None:
        """
        Drop the whole tile cache (a GDAL re-read is required) and repaint.

        Called after a controller change that alters *which pixels* a tile maps to — a
        scene added/removed or a target-CRS change (world coordinates keep the same
        ``(col, row)`` key but now cover different data, so the cached pixels are no
        longer valid). Bumping ``_read_epoch`` makes any read already in flight drop its
        result instead of inserting stale tiles. The refill is scheduled lazily by the
        next :meth:`paintEvent`. Distinct from :meth:`recomposite_only`, which reuses the
        cache with no reads.
        """
        self._read_epoch += 1
        self._tile_cache.clear()
        self._inflight_tiles.clear()
        self._pixels_dirty = True
        self.update()

    def recomposite_only(self) -> None:
        """
        Repaint from the existing tile cache, reading **only** genuinely missing tiles.

        The fast path for a z-order reorder or a visibility toggle: the cached tiles are
        still valid (they are keyed per scene, independent of order/visibility), so a
        reorder or a hide is a pure repaint with no reads, and re-showing a scene reuses
        its still-cached tiles for free. The only case that reads is *revealing* a scene
        that was hidden at every prior read of this viewport (so its tiles were never
        warped); :meth:`_start_pixel_read` detects those missing tiles and warps just
        them — an all-cached restack submits zero warp jobs.
        """
        self._schedule_pixel_read()
        self.update()

    def zoom_to_extent(self, extent: Tuple[float, float, float, float], margin: float = 0.05) -> None:
        """
        Frame a world-coordinate ``(min_x, min_y, max_x, max_y)`` extent, padded by
        ``margin`` (a fraction of each side) so the target is not flush against the
        viewport edges, then repaint.

        This is the per-scene analogue of the one-time whole-mosaic auto-fit in
        :meth:`paintEvent`. It is a no-op until the widget has a real size (the camera
        math needs a non-degenerate viewport). Setting ``_has_fitted`` keeps the
        auto-fit branch from later snapping the camera back to the whole-grid extent,
        so a deliberate zoom is never yanked away. ``update()`` is enough: the moved
        camera makes ``paintEvent`` schedule the debounced, off-thread read for the
        newly-visible tiles (the same path as a pan/zoom gesture).
        """
        if self.width() <= 1 or self.height() <= 1:
            return
        min_x, min_y, max_x, max_y = extent
        pad_x = (max_x - min_x) * margin
        pad_y = (max_y - min_y) * margin
        self._transform.fit_to_extent(
            (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y), self.size()
        )
        self._has_fitted = True
        self.update()

    def _visible_world_extent(self) -> Tuple[float, float, float, float]:
        """The camera's visible rectangle as ``(min_x, min_y, max_x, max_y)`` in world."""
        size = self.size()
        tl = self._transform.screen_to_world(QPointF(0.0, 0.0), size)
        br = self._transform.screen_to_world(QPointF(self.width(), self.height()), size)
        # screen y-down => tl is the top (max world y), br the bottom (min world y).
        return (tl.x(), br.y(), br.x(), tl.y())

    @staticmethod
    def _envelope_intersects(geom: ogr.Geometry, extent: Tuple[float, float, float, float]) -> bool:
        """Cheap AABB test: does a footprint's envelope overlap the visible rect?"""
        gmin_x, gmax_x, gmin_y, gmax_y = geom.GetEnvelope()
        emin_x, emin_y, emax_x, emax_y = extent
        return not (gmax_x < emin_x or gmin_x > emax_x or gmax_y < emin_y or gmin_y > emax_y)

    def _numpy_to_argb_qimage(self, rgba: np.ndarray) -> QImage:
        """
        Wrap an ``(H, W, 4)`` uint8 RGBA array as a ``QImage.Format_ARGB32``.

        ``Format_ARGB32`` packs each pixel as a native-endian ``0xAARRGGBB`` uint32,
        so the channels are shifted into place (the alpha-aware analog of
        ``rasterview.make_rgb_image``, which forces opaque alpha). ``.copy()`` detaches
        the result from the transient NumPy buffer so it owns its own pixels.
        """
        h, w = rgba.shape[:2]
        a = rgba[:, :, 3].astype(np.uint32)
        r = rgba[:, :, 0].astype(np.uint32)
        g = rgba[:, :, 1].astype(np.uint32)
        b = rgba[:, :, 2].astype(np.uint32)
        argb = np.ascontiguousarray((a << 24) | (r << 16) | (g << 8) | b, dtype=np.uint32)
        return QImage(argb.data, w, h, QImage.Format_ARGB32).copy()

    def _current_signature(self) -> Tuple[float, float, float, int, int]:
        """The render signature (viewport identity) the pixel cache is keyed on."""
        return (
            self._transform.center_x,
            self._transform.center_y,
            self._transform.world_units_per_pixel,
            self.width(),
            self.height(),
        )

    def _render_bucket(self) -> int:
        """
        The zoom bucket tiles are rendered at — the camera's bucket, but never finer
        than the mosaic's own resolution.

        Rendering finer than the source data is pointless work: ``gdal.Warp`` would just
        upsample the common grid onto a denser grid, producing no new detail while
        (deeply below native) getting very slow. So the render bucket is floored at the
        common grid's resolution: once the user zooms past native, the same
        grid-resolution tiles are reused and the world->screen affine scales them up (one
        source pixel drawn as a block), which is both correct and free. The floor is a
        single view-wide value, so every scene still renders at one shared bucket and the
        per-cell :meth:`composite` stays pixel-aligned.
        """
        camera_bucket = _zoom_bucket(self._transform.world_units_per_pixel)
        gt = self._controller.get_common_grid().geotransform
        if gt is None:
            return camera_bucket
        grid_res = min(abs(gt[1]), abs(gt[5]))  # finest of the grid's x/y pixel size
        if grid_res <= 0.0:
            return camera_bucket
        return max(camera_bucket, _zoom_bucket(grid_res))  # never finer than the grid

    def _schedule_pixel_read(self) -> None:
        """
        Ask for a read of any tiles the current viewport needs but does not have.

        With a scheduler, this (re)starts the debounce timer so a pan/zoom gesture's
        burst of paints collapses into a single background read once the camera
        settles. Without one (e.g. a bare view in a unit test), it reads synchronously
        so behavior is still correct, just on the GUI thread.
        """
        if self._app_services is None:
            self._start_pixel_read()
        else:
            self._debounce_timer.start()

    def _start_pixel_read(self) -> None:
        """
        Snapshot the viewport on the GUI thread and warp its **missing** tiles.

        The cheap, Qt-adjacent work (resolving the target CRS, reprojecting footprints,
        the per-tile intersect test) happens here; only the heavy per-tile warps are
        handed to the worker. Tiles already cached or already in flight are skipped, so
        a pan warps only the newly-exposed cells and an all-cached restack warps
        nothing. With no scheduler the worker runs inline.
        """
        target_wkt = self._controller.get_target_crs()
        w, h = self.width(), self.height()
        if target_wkt is None or w <= 1 or h <= 1:
            return  # nothing to read yet (no grid, or the widget has no real size)

        world_extent = self._visible_world_extent()
        bucket = self._render_bucket()
        try:
            footprints = self._controller.visible_scene_footprints_in_common_crs()
        except Exception:  # noqa: BLE001 — never let a paint-driven read raise
            logger.exception("Failed to resolve mosaic scene footprints")
            footprints = []

        # Snapshot the resampling method on the GUI thread so the warp (which runs off
        # the GUI thread) uses the controller's current choice (#638).
        resample_alg = self._controller.get_resample_alg()
        cells = _tiles_covering_extent(bucket, world_extent, _PREFETCH_RING)
        jobs: List[Tuple[_TileKey, MosaicScene, str, Tuple[float, float, float, float], int, int, int]] = []
        for scene, geom in footprints:
            for col, row in cells:
                key: _TileKey = (id(scene), bucket, col, row)
                if key in self._tile_cache or key in self._inflight_tiles:
                    continue
                tile_extent = _tile_world_extent(bucket, col, row)
                if not self._envelope_intersects(geom, tile_extent):
                    continue  # this scene's data does not reach this cell — never warp it
                jobs.append((key, scene, target_wkt, tile_extent, _TILE_PX, _TILE_PX, resample_alg))

        if not jobs:
            return  # every needed tile is cached or already being read

        submitted = tuple(job[0] for job in jobs)
        self._inflight_tiles.update(submitted)
        epoch = self._read_epoch
        if self._app_services is None:
            self._apply_pixel_read((epoch, submitted, _render_scene_layers(jobs)))
            return

        def _done(future, epoch=epoch, submitted=submitted):
            # Runs on the scheduler thread; hop back to the GUI thread via the signal.
            try:
                results = future.result()
            except Exception:  # noqa: BLE001 — surface nothing rather than crash a worker
                logger.exception("Mosaic pixel read failed")
                results = []
            self._read_ready.emit((epoch, submitted, results))

        self._pending_future = self._app_services.scheduler.submit_thread(
            PriorityClass.BACKGROUND, _render_scene_layers, jobs
        )
        self._pending_future.add_done_callback(_done)

    def _apply_pixel_read(self, payload) -> None:
        """
        Insert completed tiles into the LRU cache on the GUI thread.

        ``payload`` is ``(epoch, submitted_keys, results)``. Every submitted key is
        cleared from ``_inflight_tiles`` first — including tiles the worker skipped on
        error — so a transient failure can be retried on the next settle. Results are
        dropped without inserting when ``epoch`` is stale (the cache was invalidated
        after this read was submitted), so a CRS change / add / remove never installs
        tiles that no longer match. Reads are otherwise **additive**: a late tile from a
        superseded pan is still a valid entry for its cell, kept or evicted by the LRU.
        """
        epoch, submitted_keys, results = payload
        self._inflight_tiles.difference_update(submitted_keys)
        if epoch != self._read_epoch:
            return  # cache was invalidated since this read was submitted; drop it
        for key, rgba in results:
            self._tile_cache[key] = self._numpy_to_argb_qimage(rgba)
            self._tile_cache.move_to_end(key)
        while len(self._tile_cache) > _TILE_CACHE_MAX:
            self._tile_cache.popitem(last=False)  # evict the least-recently-used tile
        self._pixels_dirty = False
        self.update()

    # -- pan / zoom -----------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 (Qt override)
        notches = event.angleDelta().y() / 120.0
        if notches == 0.0:
            return
        factor = _WHEEL_ZOOM_STEP**notches
        self._transform.zoom(factor, event.position(), self.size())
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MiddleButton:
            self._pan_anchor = event.position()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if self._pan_anchor is not None:
            pos = event.position()
            delta = pos - self._pan_anchor
            self._transform.pan(delta.x(), delta.y())
            self._pan_anchor = pos
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MiddleButton and self._pan_anchor is not None:
            self._pan_anchor = None
            event.accept()

    # -- tile drawing (#674) --------------------------------------------------

    def _draw_pixel_layer(self, painter: QPainter) -> None:
        """
        Draw the tiled pixel layer for the current viewport, in two passes.

        Pass 1 (fallback) draws the nearest *other* zoom bucket's cached tiles that
        intersect the viewport, scaled through the camera, so a zoom into a not-yet-read
        bucket is never black. Pass 2 (sharp) draws the current bucket: per cell it
        stacks each visible scene's tile via :meth:`composite` (the single stacking
        indirection point) and blits the result at the cell's world rect. Both passes
        map world rects through the world->screen affine, so off-screen tiles simply do
        not land on the widget.
        """
        w2s = self._transform.world_to_screen(self.size())
        world_extent = self._visible_world_extent()
        bucket = self._render_bucket()
        # bottom-to-top so later (upper) scenes composite over lower ones
        visible_scenes = [s for s in self._controller.get_scenes() if s.visible]
        scene_order = {id(s): i for i, s in enumerate(visible_scenes)}

        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self._draw_fallback_tiles(painter, w2s, scene_order, bucket, world_extent)

        for col, row in _tiles_covering_extent(bucket, world_extent):
            layers: List[Optional[QImage]] = []
            for scene in visible_scenes:
                key: _TileKey = (id(scene), bucket, col, row)
                img = self._tile_cache.get(key)
                if img is not None:
                    self._tile_cache.move_to_end(key)  # drawn => most-recently-used
                layers.append(img)
            cell = self.composite(layers)
            if cell is not None:
                self._draw_tile_image(painter, w2s, bucket, col, row, cell)

    def _draw_fallback_tiles(
        self,
        painter: QPainter,
        w2s: QTransform,
        scene_order: Dict[int, int],
        bucket: int,
        world_extent: Tuple[float, float, float, float],
    ) -> None:
        """
        Under-draw the nearest cached bucket other than ``bucket`` (a scaled preview).

        Picks, among cached tiles of currently-visible scenes that intersect the
        viewport, the bucket closest to the current one and draws its tiles bottom-to-top
        directly (no per-cell :meth:`composite` — cross-bucket tiles are not cell-aligned
        across scenes, and this is only a transient fill under the sharp pass). Bounded by
        the cache size, so it is cheap.
        """
        candidates = [
            key
            for key in self._tile_cache
            if key[1] != bucket
            and key[0] in scene_order
            and self._tile_intersects_viewport(key[1], key[2], key[3], world_extent)
        ]
        if not candidates:
            return
        best_bucket = min(candidates, key=lambda k: abs(k[1] - bucket))[1]
        tiles = [k for k in candidates if k[1] == best_bucket]
        tiles.sort(key=lambda k: scene_order[k[0]])  # bottom-to-top
        for _sid, b, col, row in tiles:
            self._draw_tile_image(painter, w2s, b, col, row, self._tile_cache[(_sid, b, col, row)])

    @staticmethod
    def _tile_intersects_viewport(
        bucket: int, col: int, row: int, world_extent: Tuple[float, float, float, float]
    ) -> bool:
        """Does a tile's world rect overlap the visible world rectangle?"""
        tx0, ty0, tx1, ty1 = _tile_world_extent(bucket, col, row)
        vx0, vy0, vx1, vy1 = world_extent
        return not (tx1 <= vx0 or tx0 >= vx1 or ty1 <= vy0 or ty0 >= vy1)

    def _draw_tile_image(
        self, painter: QPainter, w2s: QTransform, bucket: int, col: int, row: int, img: QImage
    ) -> None:
        """Blit one tile ``img`` at its ``(bucket, col, row)`` world rect through ``w2s``."""
        min_x, min_y, max_x, max_y = _tile_world_extent(bucket, col, row)
        target_rect = w2s.mapRect(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))
        painter.drawImage(target_rect, img)

    def composite(self, layers: List[Optional[QImage]]) -> Optional[QImage]:
        """
        Stack per-scene ARGB ``layers`` bottom-to-top into one image.

        ``layers`` is already in render order — index 0 is the bottom scene, the last
        index the top — with ``None`` for scenes that are hidden or have no cached
        layer, so the z-order is encoded in the list itself. Layers are painted in
        order with the default ``SourceOver`` compositing, so each layer's alpha lets
        lower scenes show through its nodata holes. Returns ``None`` when nothing is
        drawable.

        This is the single indirection point for the pixel layer: deferred work
        (seamlines, feathering) reimplements only this method's internals — stacking by
        territory mask instead of raw z-order — without touching callers.
        """
        images = [im for im in layers if im is not None]
        if not images:
            return None
        width = max(im.width() for im in images)
        height = max(im.height() for im in images)
        result = QImage(width, height, QImage.Format_ARGB32)
        result.fill(Qt.transparent)
        painter = QPainter(result)
        for im in images:  # bottom-to-top; later draws land on top
            painter.drawImage(0, 0, im)
        painter.end()
        return result

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt override)
        # Initial fit: frame the mosaic extent the first time a grid is available
        # *and* the widget has a real size (done here rather than on grid-build so we
        # never divide by a 0x0 not-yet-shown viewport). Later grid rebuilds do not
        # refit, so a user's pan/zoom is never yanked out from under them.
        #
        # When the mosaic goes empty (all scenes removed) the fit is armed again, so
        # the next scene re-frames the camera instead of leaving it parked on the
        # extent of the now-removed scene (which would render the new scene's
        # footprints off-screen).
        grid = self._controller.get_common_grid()
        if grid.extent is None:
            self._has_fitted = False
        elif not self._has_fitted and self.width() > 1 and self.height() > 1:
            self._transform.fit_to_extent(grid.extent, self.size())
            self._has_fitted = True

        if self._geometry_dirty:
            self._rebuild_overlay_geometry()
            self._geometry_dirty = False

        # Pixel layer: when the camera moved (pan/zoom) or the cache was explicitly
        # invalidated, schedule a debounced read of whatever tiles the new viewport is
        # missing. A pure restack (reorder/visibility) leaves the camera put, so it does
        # not arm here — it schedules its own (usually no-op) read via recomposite_only.
        signature = self._current_signature()
        if self._pixels_dirty or self._last_paint_signature != signature:
            self._pixels_dirty = False
            self._last_paint_signature = signature
            self._schedule_pixel_read()

        with get_painter(self) as painter:
            painter.fillRect(self.rect(), self.palette().window())

            # --- Layer 1: pixel layer (#674, tiled) ------------------------------
            # Cached tiles drawn in world coordinates through the affine; a nearest
            # other-bucket fallback under-draws so pan/zoom never shows a black frame.
            self._draw_pixel_layer(painter)

            # --- Layer 2: vector overlay (#636) ----------------------------------
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setWorldTransform(self._transform.world_to_screen(self.size()))

            # Bounding box: the union extent of all footprints in the common CRS.
            if self._bbox_extent is not None:
                min_x, min_y, max_x, max_y = self._bbox_extent
                painter.setPen(_BBOX_PEN)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))

            # Overlap highlight: for each scene, clip to its hidden region (the part
            # covered by anything above it in z-order) and fill + outline it in the
            # highlight color. Union approach => one clip/draw per scene, not per pair.
            for footprint_path, hidden_path in zip(self._footprint_paths, self._hidden_paths):
                if hidden_path is None:
                    continue
                painter.save()
                painter.setClipPath(hidden_path)
                painter.fillPath(hidden_path, _OVERLAP_BRUSH)
                painter.setPen(_OVERLAP_PEN)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(footprint_path)
                painter.restore()

            # All footprint outlines on top, normal color, so every boundary stays
            # visible even where it passes through an overlap region.
            painter.setPen(_FOOTPRINT_PEN)
            painter.setBrush(Qt.NoBrush)
            for footprint_path in self._footprint_paths:
                painter.drawPath(footprint_path)
