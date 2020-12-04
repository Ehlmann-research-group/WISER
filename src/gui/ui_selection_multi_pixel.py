from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .task_delegate import TaskDelegate
from raster.selection import MultiPixelSelection


# This is the offset for how the bounding rectangle is drawn around the pixels
# in a multi-pixel selection.
MULTIPIXEL_BOUNDS_OFFSET = 2

# This is the offset for how individual pixels are drawn in a multi-pixel
# selection.
MULTIPIXEL_PIXEL_OFFSET = 1


def draw_multi_pixel_selection(rasterview, painter, mp_sel, color, active=False):
    '''
    This helper function draws a multi-pixel selection in a rasterview, with
    various config options passed as arguments.
    '''

    scale = rasterview.get_scale()

    pen = QPen(color)
    painter.setPen(pen)

    points_scaled = [p * scale for p in mp_sel.get_pixels()]

    for p in points_scaled:
        if scale >= 6:
            painter.drawRect(p.x() + MULTIPIXEL_PIXEL_OFFSET,
                             p.y() + MULTIPIXEL_PIXEL_OFFSET,
                             scale - 2 * MULTIPIXEL_PIXEL_OFFSET,
                             scale - 2 * MULTIPIXEL_PIXEL_OFFSET)
        else:
            painter.drawRect(p.x(), p.y(), scale, scale)

    if len(points_scaled) > 1:
        # Compute the rectangle that bounds the points, and draw it.
        xs = [p.x() for p in points_scaled]
        ys = [p.y() for p in points_scaled]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        if not active:
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)

        rect_scaled = QRect(x_min - MULTIPIXEL_BOUNDS_OFFSET,
                            y_min - MULTIPIXEL_BOUNDS_OFFSET,
                            (x_max - x_min + scale) + 2 * MULTIPIXEL_BOUNDS_OFFSET,
                            (y_max - y_min + scale) + 2 * MULTIPIXEL_BOUNDS_OFFSET)

        painter.drawRect(rect_scaled)


class MultiPixelSelectionManipulator(TaskDelegate):
    '''
    Since both creation and editing of multi-pixel selections is so similar,
    this base class implements the common functionality of both, and then
    subclasses customize it as needed.
    '''

    def __init__(self, point_set, rasterpane, rasterview=None):
        super().__init__(rasterpane, rasterview)
        self._points = point_set

    def on_mouse_release(self, mouse_event):
        point = self._rasterview.image_coord_to_raster_coord(mouse_event.localPos())

        # Toggle the point's inclusion in the set of pixels.
        if point in self._points:
            self._points.remove(point)
        else:
            self._points.add(point)

        return False

    def on_key_release(self, key_event):
        '''
        In the multi-pixel selection manipulator, the Esc key ends the
        operation.
        '''
        return key_event.key() == Qt.Key_Escape


    def draw_state(self, painter):
        if len(self._points) == 0:
            return

        scale = self._rasterview.get_scale()

        points_scaled = [p * scale for p in self._points]

        color = self._app_state.get_config('raster.selection.edit_outline')
        painter.setPen(QPen(color))

        for p in points_scaled:
            if scale >= 6:
                painter.drawRect(p.x() + MULTIPIXEL_PIXEL_OFFSET,
                                 p.y() + MULTIPIXEL_PIXEL_OFFSET,
                                 scale - 2 * MULTIPIXEL_PIXEL_OFFSET,
                                 scale - 2 * MULTIPIXEL_PIXEL_OFFSET)
            else:
                painter.drawRect(p.x(), p.y(), scale, scale)


class MultiPixelSelectionCreator(MultiPixelSelectionManipulator):

    def __init__(self, rasterpane, rasterview=None):
        super().__init__(set(), rasterpane, rasterview)
        self._app_state.show_status_text(
            'Left-click on pixels to toggle their inclusion in the selection.' +
            '  Press Esc key to finish entry.')

    def finish(self):
        if len(self._points) == 0:
            self._app_state.show_status_text(
                'No pixels selected; not creating ROI.', 5)
            return

        sel = MultiPixelSelection(self._points)
        roi = self._rasterpane.get_current_roi()
        roi.add_selection(sel)

        # TODO(donnie):  Signal to the app-state that the ROI changed, so that
        #     everyone can be notified.

        self._app_state.clear_status_text()


class MultiPixelSelectionEditor(MultiPixelSelectionManipulator):

    def __init__(self, mp_sel, rasterpane, rasterview=None):
        super().__init__(mp_sel.get_pixels(), rasterpane, rasterview)
        self._mp_sel = mp_sel
        self._app_state.show_status_text(
            'Left-click on pixels to toggle their inclusion in the selection.' +
            '  Press Esc key to finish edits.')

    def finish(self):
        if len(self._points) == 0:
            # TODO(donnie):  In this case, we are ending up with an empty
            #     multi-pixel selection, which is no good!
            pass

        # TODO(donnie):  Signal to the app-state that the ROI changed, so that
        #     everyone can be notified.

        self._app_state.clear_status_text()
