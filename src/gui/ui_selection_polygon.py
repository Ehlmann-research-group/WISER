from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .task_delegate import TaskDelegate
from raster.selection import PolygonSelection

from .geom import distance, lines_cross


def draw_polygon_selection(rasterview, painter, poly_sel, color, active=False):
    '''
    Draw a polygon selection, with various options such as whether it is
    active (user-selected), etc.
    '''

    scale = rasterview.get_scale()

    pen = QPen(color)
    if not active:
        pen.setStyle(Qt.DashLine)
    painter.setPen(pen)

    points = poly_sel.get_points()
    points_scaled = [p * scale for p in points]
    for i in range(len(points)):
        painter.drawLine(points_scaled[i - 1], points_scaled[i])

    if active:
        # Draw boxes on all control-points.
        color = Qt.yellow
        painter.setPen(QPen(color))
        for cp in points_scaled:
            cp_scaled = cp * scale
            painter.fillRect(cp_scaled.x() - 2, cp_scaled.y() - 2, 4, 4, color)


class PolygonSelectionCreator(TaskDelegate):

    # This variable tunes the sensitivity of closing the polygon.  Lower values
    # require the start and end points to be closer together to close a polygon.
    SENSITIVITY = 3

    def __init__(self, app_state, rasterview=None):
        super().__init__(rasterview)

        self._app_state = app_state
        self._points = []
        self._cursor_position = None

        self._done = False

    def _close_enough(self, p1, p2):
        '''
        Returns True when two points in raster coordinates are "close enough" to
        consider the polygon to be closed.  The measure depends on the scale of
        the display, since it's easy to be accurate when zoomed in, but very
        difficult to be accurate when zoomed out.
        '''
        scale = self._rasterview.get_scale()
        d = distance(p1, p2)
        return (d < PolygonSelectionCreator.SENSITIVITY / scale)

    def on_mouse_press(self, mouse_event):
        return self._done

    def on_mouse_release(self, mouse_event):
        point = self._rasterview.image_coord_to_raster_coord(mouse_event.localPos())

        if len(self._points) > 0 and self._close_enough(point, self._points[0]):
            # User clicked on the first point in the polygon (or they clicked
            # close enough to it), closing it.  Therefore we are done.
            self._done = True

        else:
            # User is attempting to add another point to the polygon.
            # TODO(donnie):  Only accept if adding the point won't generate a
            #     weird mangled polygon.
            self._points.append(point)

        return self._done

    def on_mouse_move(self, mouse_event):
        # Store the current mouse location in raster coordinates,
        # for drawing the current state.

        point = self._rasterview.image_coord_to_raster_coord(mouse_event.localPos())
        self._cursor_position = point
        return False

    def on_key_press(self, key_event):
        return False

    def on_key_release(self, key_event):
        '''
        In the rectangle selection creator, the Esc key allows the user to
        cancel the points that have been entered, in the order they were
        entered.
        '''
        if key_event.key() != Qt.Key_Escape:
            return False

        if len(self.points) > 0:
            del self.points[-1]

        return False


    def draw_state(self, painter):
        if len(self._points) == 0:
            return

        scale = self._rasterview.get_scale()

        bad_point = False
        closed_loop = False

        points_scaled = [p * scale for p in self._points]
        if self._cursor_position is not None:
            cp_scaled = self._cursor_position * scale

            # Does the last line cross any earlier lines?
            last_line = QLine(points_scaled[-1], cp_scaled)
            for i in range(len(points_scaled) - 1):
                line = QLine(points_scaled[i], points_scaled[i + 1])
                if lines_cross(last_line, line):
                    bad_point = True

            if not bad_point and len(points_scaled) >= 3:
                closed_loop = self._close_enough(cp_scaled, points_scaled[0])

            points_scaled.append(cp_scaled)

        # print(f'PS:  bad_point = {bad_point}\tclosed_loop = {closed_loop}')

        color = Qt.white # self._app_state.get_color_of('create-selection')
        pen = QPen(color)

        if not closed_loop:
            pen.setStyle(Qt.DashLine)

        painter.setPen(pen)

        for i in range(len(points_scaled) - 1):
            # print(f'i = {i}\tlen(self.points) = {len(self.points)}')
            if bad_point and (i == len(self._points) - 1):
                pen = QPen(Qt.red)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)

            painter.drawLine(points_scaled[i], points_scaled[i + 1])

        # Draw boxes on the points themselves.

        color = Qt.yellow
        painter.setPen(QPen(color))

        for p_scaled in points_scaled:
            painter.fillRect(p_scaled.x() - 2, p_scaled.y() - 2, 4, 4, color)


    def finish(self):
        sel = PolygonSelection(self._points)
        self._app_state.make_and_add_roi(sel)
