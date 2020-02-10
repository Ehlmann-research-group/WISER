from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from enum import Enum

from .geom import *

from raster.selection import RectangleSelection, PolygonSelection, MultiPixelSelection


class SelectionCreator:
    def process_mouse_move(self):
        pass

    def process_mouse_click(self):
        pass

    def process_key_event(self):
        pass

    def draw_state(self, painter):
        pass


class RectangleSelectionCreator:
    class State(Enum):
        GET_POINT_1 = 1
        GET_POINT_2 = 2
        DONE = 3
        CANCEL = -1


    def __init__(self, raster_pane):
        self._raster_pane = raster_pane

        self.state = RectangleSelectionCreator.State.GET_POINT_1
        self.point1 = None
        self.point2 = None
        self.cursor_position = None

    def onMousePress(self, widget, mouse_event):
        return False

    def onMouseRelease(self, widget, mouse_event):
        rasterview = self._raster_pane.get_rasterview()
        point = rasterview.image_coord_to_raster_coord(mouse_event.localPos())

        if self.state == RectangleSelectionCreator.State.GET_POINT_1:
            self.point1 = point
            self.state = RectangleSelectionCreator.State.GET_POINT_2

        elif self.state == RectangleSelectionCreator.State.GET_POINT_2:
            self.point2 = point
            self.state = RectangleSelectionCreator.State.DONE

        return self.state == RectangleSelectionCreator.State.DONE

    def onMouseMove(self, widget, mouse_event):
        # Store the current mouse location in raster coordinates,
        # for drawing the current state.

        rasterview = self._raster_pane.get_rasterview()
        point = rasterview.image_coord_to_raster_coord(mouse_event.localPos())

        self.cursor_position = point

        return False

    def onKeyPress(self, widget, key_event):
        return False

    def onKeyRelease(self, widget, key_event):
        '''
        In the rectangle selection creator, the Esc key allows the user to
        cancel the points that have been entered, in the order they were
        entered.
        '''
        if key_event.key() != Qt.Key_Escape:
            return False

        if self.state == RectangleSelectionCreator.State.GET_POINT_2:
            self.point2 = None
            self.state = RectangleSelectionCreator.State.GET_POINT_1

        return False


    def draw_state(self, painter):

        # The only state that we really draw anything in, is when we have
        # point 1 and are waiting for point 2.  Before that, we have nothing to
        # draw, and after that, the selection-creation process is done and the
        # actual selection will be drawn by other code.
        if self.state == RectangleSelectionCreator.State.GET_POINT_2:

            rasterview = self._raster_pane.get_rasterview()
            scale = rasterview.get_scale()

            if self.point1 is not None and self.cursor_position is not None:
                # Draw the box between point 1 and the current mouse position, in a
                # dotted rectangle.

                color = Qt.white # self._app_state.get_color_of('create-selection')
                pen = QPen(color)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)

                p1_scaled = self.point1 * scale
                p2_scaled = self.cursor_position * scale

                rect = get_rectangle(p1_scaled, p2_scaled)
                painter.drawRect(rect)

            # Draw boxes on the two points themselves.

            color = Qt.yellow
            painter.setPen(QPen(color))

            if self.point1 is not None:
                p1_scaled = self.point1 * scale
                painter.fillRect(p1_scaled.x() - 2, p1_scaled.y() - 2, 4, 4, color)

            if self.cursor_position is not None:
                p2_scaled = self.cursor_position * scale
                painter.fillRect(p2_scaled.x() - 2, p2_scaled.y() - 2, 4, 4, color)


    def get_selection(self):
        return RectangleSelection(self.point1, self.point2)


class PolygonSelectionCreator:

    # This variable tunes the sensitivity of closing the polygon.  Lower values
    # require the start and end points to be closer together to close a polygon.
    SENSITIVITY = 3

    def __init__(self, raster_pane):
        self._raster_pane = raster_pane

        self.points = []
        self.cursor_position = None

        self.done = False

    def _close_enough(self, p1, p2):
        '''
        Returns True when two points in raster coordinates are "close enough" to
        consider the polygon to be closed.  The measure depends on the scale of
        the display, since it's easy to be accurate when zoomed in, but very
        difficult to be accurate when zoomed out.
        '''
        rasterview = self._raster_pane.get_rasterview()
        scale = rasterview.get_scale()
        d = distance(p1, p2)
        return (d < PolygonSelectionCreator.SENSITIVITY / scale)

    def onMousePress(self, widget, mouse_event):
        return self.done

    def onMouseRelease(self, widget, mouse_event):
        rasterview = self._raster_pane.get_rasterview()
        point = rasterview.image_coord_to_raster_coord(mouse_event.localPos())

        if len(self.points) > 0 and self._close_enough(point, self.points[0]):
            # User clicked on the first point in the polygon (or they clicked
            # close enough to it), closing it.  Therefore we are done.
            self.done = True

        else:
            # User is attempting to add another point to the polygon.
            # TODO(donnie):  Only accept if adding the point won't generate a
            #     weird mangled polygon.
            self.points.append(point)

        return self.done

    def onMouseMove(self, widget, mouse_event):
        # Store the current mouse location in raster coordinates,
        # for drawing the current state.

        rasterview = self._raster_pane.get_rasterview()
        point = rasterview.image_coord_to_raster_coord(mouse_event.localPos())
        self.cursor_position = point
        return False

    def onKeyPress(self, widget, key_event):
        return False

    def onKeyRelease(self, widget, key_event):
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
        if len(self.points) == 0:
            return

        rasterview = self._raster_pane.get_rasterview()
        scale = rasterview.get_scale()

        bad_point = False
        closed_loop = False

        points_scaled = [p * scale for p in self.points]
        if self.cursor_position is not None:
            cp_scaled = self.cursor_position * scale

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
            if bad_point and (i == len(self.points) - 1):
                pen = QPen(Qt.red)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)

            painter.drawLine(points_scaled[i], points_scaled[i + 1])

        # Draw boxes on the points themselves.

        color = Qt.yellow
        painter.setPen(QPen(color))

        for p_scaled in points_scaled:
            painter.fillRect(p_scaled.x() - 2, p_scaled.y() - 2, 4, 4, color)


    def get_selection(self):
        return PolygonSelection(self.points)


class MultiPixelSelectionCreator:
    class State(Enum):
        GET_POINT = 1
        DONE = 3
        CANCEL = -1


    def __init__(self, raster_pane):
        self._raster_pane = raster_pane

        self.points = set()


    def onMousePress(self, widget, mouse_event):
        return False

    def onMouseRelease(self, widget, mouse_event):
        rasterview = self._raster_pane.get_rasterview()
        point = rasterview.image_coord_to_raster_coord(mouse_event.localPos())

        if point in self.points:
            self.points.remove(point)

        else:
            self.points.add(point)

        return False


    def onMouseMove(self, widget, mouse_event):
        return False

    def onKeyPress(self, widget, key_event):
        return False

    def onKeyRelease(self, widget, key_event):
        '''
        In the rectangle selection creator, the Esc key allows the user to
        cancel the points that have been entered, in the order they were
        entered.
        '''
        return key_event.key() == Qt.Key_Escape


    def draw_state(self, painter):
        if len(self.points) == 0:
            return

        rasterview = self._raster_pane.get_rasterview()
        scale = rasterview.get_scale()

        points_scaled = [p * scale for p in self.points]

        color = Qt.white # self._app_state.get_color_of('create-selection')
        painter.setPen(QPen(color))

        for p in points_scaled:
            if scale >= 6:
                painter.drawRect(p.x() + 1, p.y() + 1, scale - 2, scale - 2)
            else:
                painter.drawRect(p.x(), p.y(), scale, scale)


    def get_selection(self):
        return MultiPixelSelection(self.points)
