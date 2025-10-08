from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .task_delegate import TaskDelegate
from wiser.raster.selection import PolygonSelection

from .geom import distance, lines_cross, manhattan_distance

from .ui_selection import CONTROL_POINT_SIZE

from .util import scale_qpoint_by_float


def draw_polygon_selection(rasterview, painter, poly_sel, color, active=False):
    """
    Draw a polygon selection, with various options such as whether it is
    active (user-selected), etc.
    """

    scale = rasterview.get_scale()

    pen = QPen(color)
    if not active:
        pen.setStyle(Qt.DashLine)
    painter.setPen(pen)

    points = poly_sel.get_points()
    points_scaled = [scale_qpoint_by_float(p, scale) for p in points]
    for i in range(len(points)):
        painter.drawLine(points_scaled[i - 1], points_scaled[i])

    if active:
        # Draw boxes on all control-points.
        color = self._app_state.get_config("raster.selection.edit_points")
        painter.setPen(QPen(color))
        for cp in points_scaled:
            cp_scaled = cp * scale
            painter.fillRect(
                cp_scaled.x() - CONTROL_POINT_SIZE / 2,
                cp_scaled.y() - CONTROL_POINT_SIZE / 2,
                CONTROL_POINT_SIZE,
                CONTROL_POINT_SIZE,
                color,
            )


class PolygonSelectionCreator(TaskDelegate):
    # This variable tunes the sensitivity of closing the polygon.  Lower values
    # require the start and end points to be closer together to close a polygon.
    SENSITIVITY = 10

    def __init__(self, rasterpane, rasterview=None):
        super().__init__(rasterpane, rasterview)

        self._points = []
        self._cursor_position = None

        self._done = False

        self._app_state.show_status_text(
            "Left-click to add points to polygon.  Esc key discards last "
            + "point added.  Close polygon to finish entry."
        )

    def _close_enough(self, p1, p2):
        """
        Returns True when two points in raster coordinates are "close enough" to
        consider the polygon to be closed.  The measure depends on the scale of
        the display, since it's easy to be accurate when zoomed in, but very
        difficult to be accurate when zoomed out.
        """
        scale = self._rasterview.get_scale()
        d = distance(p1, p2)
        return d < PolygonSelectionCreator.SENSITIVITY / scale

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
        """
        In the rectangle selection creator, the Esc key allows the user to
        cancel the points that have been entered, in the order they were
        entered.
        """
        if key_event.key() != Qt.Key_Escape:
            return False

        if len(self._points) > 0:
            del self._points[-1]

        return False

    def draw_state(self, painter):
        if len(self._points) == 0:
            return

        scale = self._rasterview.get_scale()

        bad_point = False
        closed_loop = False
        points_scaled = [scale_qpoint_by_float(p, scale) for p in self._points]
        if self._cursor_position is not None:
            cp_scaled = scale_qpoint_by_float(self._cursor_position, scale)

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

        color = self._app_state.get_config("raster.selection.edit_outline")
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

        color = self._app_state.get_config("raster.selection.edit_points")
        painter.setPen(QPen(color))

        for p_scaled in points_scaled:
            painter.fillRect(
                p_scaled.x() - CONTROL_POINT_SIZE / 2,
                p_scaled.y() - CONTROL_POINT_SIZE / 2,
                CONTROL_POINT_SIZE,
                CONTROL_POINT_SIZE,
                color,
            )

    def finish(self):
        sel = PolygonSelection(self._points)
        roi = self._rasterpane.get_current_roi()
        roi.add_selection(sel)

        # Signal that the ROI changed, so that everyone can be notified.
        self._rasterpane.roi_selection_changed.emit(roi, sel)

        self._app_state.clear_status_text()


class PolygonSelectionEditor(TaskDelegate):
    def __init__(self, roi, poly_sel, rasterpane, rasterview=None):
        super().__init__(rasterpane, rasterview)
        self._roi = roi
        self._poly_sel = poly_sel
        self._editing_cp_index = None

        # Initialize the control-points for adjusting the polygon selection.
        self._control_points = list(self._poly_sel.get_points())

        # Give the user some directions.
        self._app_state.show_status_text(
            "Left-click and drag control points to adjust the polygon."
            + "  Press Esc key to finish edits."
        )

    def _pick_control_point(self, p):
        for idx, cp in enumerate(self._control_points):
            # TODO(donnie):  May be too difficult to pick control-points if we
            #     only check equality, not "is this point within a certain
            #     distance".  Note that this picking occurs within data-set
            #     coordinate space.
            if manhattan_distance(p, cp) <= 2:
                return idx

        return None

    def on_mouse_press(self, mouse_event):
        # Figure out which control-point was chosen, if any.
        p = self._rasterview.image_coord_to_raster_coord(mouse_event.localPos())
        self._editing_cp_index = self._pick_control_point(p)
        return False

    def on_mouse_move(self, mouse_event):
        self._handle_mouse_update(mouse_event)
        return False

    def on_mouse_release(self, mouse_event):
        self._handle_mouse_update(mouse_event)
        self._editing_cp_index = None
        return False

    def _handle_mouse_update(self, mouse_event):
        if self._editing_cp_index is None:
            # Not editing a control-point, so don't do anything.
            return False

        # Update the control-points based on the mouse operation.
        p = self._rasterview.image_coord_to_raster_coord(mouse_event.localPos())
        self._adjust_control_points(p)

    def _adjust_control_points(self, p):
        """
        This helper adjusts all affected control points based on the input point
        p, which should be in the data-set coordinate system.

        Note that this function assumes that self._editing_cp_index is set to
        the index of the control-point that is being manipulated!
        """
        assert self._editing_cp_index is not None

        # Local variables for these values, since these names are long!
        i = self._editing_cp_index
        edit_cp = self._control_points[i]

        # Adjust the specific control point being edited.  Use the mutators on
        # the object, rather than replacing it with a new object.
        edit_cp.setX(p.x())
        edit_cp.setY(p.y())

    def on_key_release(self, key_event):
        """
        In the rectangle selection editor, the Esc key ends the edit operation.
        """
        return key_event.key() == Qt.Key_Escape

    def draw_state(self, painter):
        scale = self._rasterview.get_scale()
        points_scaled = [scale_qpoint_by_float(p, scale) for p in self._control_points]

        # Draw the polygon specified by all the points, using a dotted line.

        color = self._app_state.get_config("raster.selection.edit_outline")
        pen = QPen(color)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)

        for i in range(len(points_scaled)):
            painter.drawLine(points_scaled[i - 1], points_scaled[i])

        # Draw boxes on all control-points.
        color = self._app_state.get_config("raster.selection.edit_points")
        painter.setPen(QPen(color))
        for cp in points_scaled:
            painter.fillRect(
                cp.x() - CONTROL_POINT_SIZE / 2,
                cp.y() - CONTROL_POINT_SIZE / 2,
                CONTROL_POINT_SIZE,
                CONTROL_POINT_SIZE,
                color,
            )

    def finish(self):
        # Signal that the ROI changed, so that everyone can be notified.
        self._rasterpane.roi_selection_changed.emit(self._roi, self._poly_sel)

        self._app_state.clear_status_text()

    def get_selection(self):
        return PolygonSelection(self._control_points)
