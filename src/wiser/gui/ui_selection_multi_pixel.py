from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .task_delegate import TaskDelegate
from wiser.raster.selection import MultiPixelSelection

from .util import scale_qpoint_by_float


# This is the offset for how individual pixels are drawn in a multi-pixel
# selection.
MULTIPIXEL_PIXEL_OFFSET = 1


def draw_multi_pixel_selection(rasterview, painter, mp_sel, color, active=False):
    """
    This helper function draws a multi-pixel selection in a rasterview, with
    various config options passed as arguments.
    """

    scale = rasterview.get_scale()

    pen = QPen(color)
    painter.setPen(pen)

    points_scaled = [scale_qpoint_by_float(p, scale) for p in mp_sel.get_pixels()]

    for p in points_scaled:
        if scale >= 6:
            painter.drawRect(
                p.x() + MULTIPIXEL_PIXEL_OFFSET,
                p.y() + MULTIPIXEL_PIXEL_OFFSET,
                scale - 2 * MULTIPIXEL_PIXEL_OFFSET,
                scale - 2 * MULTIPIXEL_PIXEL_OFFSET,
            )
        else:
            painter.drawRect(p.x(), p.y(), scale, scale)


class MultiPixelSelectionManipulator(TaskDelegate):
    """
    Since both creation and editing of multi-pixel selections is so similar,
    this base class implements the common functionality of both, and then
    subclasses customize it as needed.

    Left-click toggles a pixel.  Enter or right-click finishes the operation,
    and Esc cancels it; subclasses check self._cancelled in finish().
    """

    consumes_context_menu = True

    def __init__(self, point_set, rasterpane, rasterview=None):
        super().__init__(rasterpane, rasterview)
        self._points = point_set
        self._cancelled = False

    def on_mouse_release(self, mouse_event):
        if mouse_event.button() != Qt.LeftButton:
            return False

        point = self._rasterview.image_coord_to_raster_coord(mouse_event.localPos())

        # Toggle the point's inclusion in the set of pixels.
        if point in self._points:
            self._points.remove(point)
        else:
            self._points.add(point)

        return False

    def on_key_release(self, key_event):
        if key_event.key() == Qt.Key_Escape:
            self._cancelled = True
            return True

        return key_event.key() in (Qt.Key_Return, Qt.Key_Enter)

    def on_context_menu(self, context_menu_event):
        return True

    def draw_state(self, painter):
        if len(self._points) == 0:
            return

        scale = self._rasterview.get_scale()

        points_scaled = [scale_qpoint_by_float(p, scale) for p in self._points]

        color = self._app_state.get_config("raster.selection.edit_outline")
        painter.setPen(QPen(color))

        for p in points_scaled:
            if scale >= 6:
                painter.drawRect(
                    p.x() + MULTIPIXEL_PIXEL_OFFSET,
                    p.y() + MULTIPIXEL_PIXEL_OFFSET,
                    scale - 2 * MULTIPIXEL_PIXEL_OFFSET,
                    scale - 2 * MULTIPIXEL_PIXEL_OFFSET,
                )
            else:
                painter.drawRect(p.x(), p.y(), scale, scale)


class MultiPixelSelectionCreator(MultiPixelSelectionManipulator):
    def __init__(self, rasterpane, rasterview=None):
        super().__init__(set(), rasterpane, rasterview)
        self._app_state.show_status_text(
            "Left-click on pixels to add or remove them.  Press Enter or right-click"
            + " to finish, or Esc to cancel."
        )

    def finish(self):
        if self._cancelled:
            self._app_state.show_status_text("Multi-pixel selection cancelled.", 5)
            return

        if len(self._points) == 0:
            self._app_state.show_status_text("No pixels selected; not creating ROI.", 5)
            return

        sel = MultiPixelSelection(self._points)
        roi = self._rasterpane.get_current_roi()
        roi.add_selection(sel)

        # Signal that the ROI changed, so that everyone can be notified.
        self._rasterpane.roi_selection_changed.emit(roi, sel)

        self._app_state.clear_status_text()


class MultiPixelSelectionEditor(MultiPixelSelectionManipulator):
    def __init__(self, roi, mp_sel, rasterpane, rasterview=None):
        super().__init__(mp_sel.get_pixels(), rasterpane, rasterview)
        self._roi = roi
        self._mp_sel = mp_sel
        self._original_points = set(self._points)
        self._app_state.show_status_text(
            "Left-click on pixels to add or remove them.  Press Enter or right-click"
            + " to finish edits, or Esc to cancel."
        )

    def finish(self):
        if self._cancelled:
            self._points.clear()
            self._points.update(self._original_points)
            self._app_state.show_status_text("Multi-pixel edits cancelled.", 5)
            return

        if len(self._points) == 0:
            self._points.update(self._original_points)
            self._app_state.show_status_text("No pixels selected; selection left unchanged.", 5)
            return

        # Signal that the ROI changed, so that everyone can be notified.
        self._rasterpane.roi_selection_changed.emit(self._roi, self._mp_sel)

        self._app_state.clear_status_text()
