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

from typing import TYPE_CHECKING, List, Optional

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .app_state import ApplicationState
from .util import get_painter

from wiser.raster.mosaic_controller import MosaicController, MosaicScene

if TYPE_CHECKING:
    from .mosaic_pane import MosaicPane


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
