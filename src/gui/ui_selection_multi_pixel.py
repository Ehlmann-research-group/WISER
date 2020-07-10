from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .task_delegate import TaskDelegate
from raster.selection import MultiPixelSelection


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
            painter.drawRect(p.x() + 1, p.y() + 1, scale - 2, scale - 2)
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

        rect_scaled = QRect(x_min - 2, y_min - 2,
            (x_max - x_min + scale) + 4, (y_max - y_min + scale) + 4)

        painter.drawRect(rect_scaled)


class MultiPixelSelectionCreator(TaskDelegate):

    def __init__(self, app_state, rasterview=None):
        super().__init__(rasterview)
        self._app_state = app_state
        self._points = set()

        self._app_state.show_status_text(
            'Left-click on pixels to toggle their inclusion in the selection.' +
            '  Press Esc key to finish entry.')

    def on_mouse_release(self, mouse_event):
        point = self._rasterview.image_coord_to_raster_coord(mouse_event.localPos())

        if point in self._points:
            self._points.remove(point)

        else:
            self._points.add(point)

        return False

    def on_key_release(self, key_event):
        '''
        In the multi-pixel selection creator, the Esc key ends the create
        operation.
        '''
        return key_event.key() == Qt.Key_Escape


    def draw_state(self, painter):
        if len(self._points) == 0:
            return

        scale = self._rasterview.get_scale()

        points_scaled = [p * scale for p in self._points]

        color = Qt.white # self._app_state.get_color_of('create-selection')
        painter.setPen(QPen(color))

        for p in points_scaled:
            if scale >= 6:
                painter.drawRect(p.x() + 1, p.y() + 1, scale - 2, scale - 2)
            else:
                painter.drawRect(p.x(), p.y(), scale, scale)

    def finish(self):
        if len(self._points) == 0:
            return

        sel = MultiPixelSelection(self._points)
        self._app_state.make_and_add_roi(sel)
        self._app_state.clear_status_text()
