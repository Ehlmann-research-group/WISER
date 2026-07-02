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

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .app_state import ApplicationState
from .util import get_painter

from wiser.raster.mosaic_controller import MosaicController, MosaicScene

if TYPE_CHECKING:
    from .mosaic_pane import MosaicPane


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

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        app_state: Optional[ApplicationState] = None,
        controller: Optional[MosaicController] = None,
        mosaicpane: Optional["MosaicPane"] = None,
    ) -> None:
        super().__init__(parent=parent)
        self._app_state = app_state
        self._controller = controller if controller is not None else MosaicController()
        self._mosaicpane = mosaicpane

        # Pixel layer (#637): the composited, screen-resolution mosaic image.
        self._composite_pixmap: Optional[QPixmap] = None
        # Zoom factor for the world->screen affine (#636). 1.0 == no scaling yet.
        self._scale_factor: float = 1.0

    def get_controller(self) -> MosaicController:
        return self._controller

    def set_controller(self, controller: MosaicController) -> None:
        self._controller = controller
        self.update()

    def composite(self, layers: List[MosaicScene], order: List[int]) -> Optional[QImage]:
        """
        Composite per-scene layers into a single image, bottom-to-top by ``order``.

        This is the single indirection point for the pixel layer: today it is a stub;
        #637 stacks ARGB layers by alpha in z-order, and deferred work (seamlines,
        feathering) reimplements only this method's internals without touching callers.
        """
        # TODO(#637): build/stack per-scene ARGB caches (alpha = validity) in z-order.
        return None

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt override)
        with get_painter(self) as painter:
            # Empty background for now; a real theme background lands with the layers.
            painter.fillRect(self.rect(), self.palette().window())

            # --- Layer 1: pixel layer (#637) -------------------------------------
            # TODO(#637): draw self._composite_pixmap (the composited mosaic) here,
            # scaled by the world->screen affine.

            # --- Layer 2: vector overlay (#636) ----------------------------------
            # TODO(#636): draw footprints, bounding box, and overlap highlight on top,
            # using the world->screen affine derived from the controller's common grid.
