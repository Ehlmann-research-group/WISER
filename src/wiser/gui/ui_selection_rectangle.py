from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .task_delegate import TaskDelegate
from wiser.raster.selection import RectangleSelection

from .geom import get_rectangle, scale_rectangle, manhattan_distance

from .ui_selection import CONTROL_POINT_SIZE

def is_rect_sel_picked(rect_sel, p):
    '''
    Returns True if the specified point (in dataset coordinates) falls within
    the rectangle selection.
    '''
    return rect_sel.get_rect().contains(p)


def draw_rectangle_selection(rasterview, painter, rect_sel, color, active=False):
    '''
    Draw a rectangle selection, with various options such as whether it is
    active (user-selected), etc.
    '''
    scale = rasterview.get_scale()

    pen = QPen(color)
    if not active:
        pen.setStyle(Qt.DashLine)
    painter.setPen(pen)

    rect = rect_sel.get_rect()
    rect_scaled = scale_rectangle(rect, scale)
    painter.drawRect(rect_scaled)

    if active:
        # Draw boxes on all control-points.
        color = self._app_state.get_config('raster.selection.edit_points')
        painter.setPen(QPen(color))
        for cp in self._control_points:
            cp_scaled = cp * scale
            painter.fillRect(cp_scaled.x() - CONTROL_POINT_SIZE / 2,
                             cp_scaled.y() - CONTROL_POINT_SIZE / 2,
                             CONTROL_POINT_SIZE, CONTROL_POINT_SIZE, color)


class RectangleSelectionCreator(TaskDelegate):
    def __init__(self, rasterpane, rasterview=None):
        super().__init__(rasterpane, rasterview)
        self._point1 = None
        self._point2 = None

        self._app_state.show_status_text(
            'Left-click and drag to create a rectangle selection.')

    def on_mouse_press(self, mouse_event):
        point = self._rasterview.image_coord_to_raster_coord(mouse_event.localPos())
        self._point1 = point
        self._point2 = point
        return False

    def on_mouse_release(self, mouse_event):
        point = self._rasterview.image_coord_to_raster_coord(mouse_event.localPos())
        self._point2 = point
        return True

    def on_mouse_move(self, mouse_event):
        point = self._rasterview.image_coord_to_raster_coord(mouse_event.localPos())
        self._point2 = point
        return False

    def draw_state(self, painter):
        if self._point1 is None or self._point2 is None:
            return

        scale = self._rasterview.get_scale()
        print(f"before point1: {self._point1} | point2: {self._point2}")
        # p1_scaled = self._point1 * scale
        p1_scaled = QPointF(float(self._point1.x() * scale), float(self._point1.y() * scale))
        # p2_scaled = self._point2 * scale
        p2_scaled = QPointF(float(self._point2.x() * scale), float(self._point2.y() * scale))

        # Draw a box between the two points, using a dotted rectangle.

        color = self._app_state.get_config('raster.selection.edit_outline')
        pen = QPen(color)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        rect = get_rectangle(p1_scaled, p2_scaled)
        print(f"scale before: {scale}")
        print(f"P1: {p1_scaled} | P2: {p2_scaled}")
        print(f"Rect: {rect.getCoords()}")
        painter.drawRect(rect)

        # Draw boxes on the two points themselves.

        color = self._app_state.get_config('raster.selection.edit_points')
        painter.setPen(QPen(color))
        painter.fillRect(p1_scaled.x() - CONTROL_POINT_SIZE / 2,
                         p1_scaled.y() - CONTROL_POINT_SIZE / 2,
                         CONTROL_POINT_SIZE, CONTROL_POINT_SIZE, color)
        painter.fillRect(p2_scaled.x() - CONTROL_POINT_SIZE / 2,
                         p2_scaled.y() - CONTROL_POINT_SIZE / 2,
                         CONTROL_POINT_SIZE, CONTROL_POINT_SIZE, color)

    def finish(self):
        if self._point1 is None or self._point2 is None:
            return

        print(f"point1: {self._point1} | point2: {self._point2}")
        scale = self._rasterview.get_scale()
        print(f"scale after: {scale}")
        sel = RectangleSelection(self._point1, self._point2)
        roi = self._rasterpane.get_current_roi()
        roi.add_selection(sel)

        # Signal that the ROI changed, so that everyone can be notified.
        self._rasterpane.roi_selection_changed.emit(roi, sel)

        self._app_state.clear_status_text()


class RectangleSelectionEditor(TaskDelegate):
    def __init__(self, roi, rect_sel, rasterpane, rasterview=None):
        super().__init__(rasterpane, rasterview)
        self._roi = roi
        self._rect_sel = rect_sel
        self._control_points = []
        self._editing_cp_index = None

        self._init_control_points()

    def _init_control_points(self):
        '''
        Initialize the control-points for adjusting the rectangle selection.
        '''
        top_left     = self._rect_sel.get_top_left()
        bottom_right = self._rect_sel.get_bottom_right()

        self._control_points.append(top_left)
        self._control_points.append(bottom_right)

        self._control_points.append(QPoint(top_left.x(), bottom_right.y()))
        self._control_points.append(QPoint(bottom_right.x(), top_left.y()))

        # These lists specify how the control points are "connected together".
        # The position in the array is the index of the "source" control point
        # in self._control_points, and the value at that index specifies the
        # index of the control point that is affected in either the X or Y
        # dimension.  Since each control point only affects one other control
        # point in the X or Y direction, this is an easy and efficient way to
        # encode these relationships.
        self._cp_affects_x = [2, 3, 0, 1]
        self._cp_affects_y = [3, 2, 1, 0]

        # Give the user some directions.
        self._app_state.show_status_text(
            'Left-click and drag control points to adjust the rectangle.' +
            '  Press Esc key to finish edits.')


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
        '''
        This helper adjusts all affected control points based on the input point
        p, which should be in the data-set coordinate system.

        Note that this function assumes that self._editing_cp_index is set to
        the index of the control-point that is being manipulated!
        '''
        assert self._editing_cp_index is not None

        # Local variables for these values, since these names are long!
        i = self._editing_cp_index
        edit_cp = self._control_points[i]

        # Adjust the specific control point being edited.  Use the mutators on
        # the object, rather than replacing it with a new object.
        edit_cp.setX(p.x())
        edit_cp.setY(p.y())

        # Also need to adjust the adjacent control points.  Use the
        # _cp_affects_x/_cp_affects_y members to drive this interaction,
        # to keep the code simple.
        self._control_points[self._cp_affects_x[i]].setX(p.x())
        self._control_points[self._cp_affects_y[i]].setY(p.y())


    def on_key_release(self, key_event):
        '''
        In the rectangle selection editor, the Esc key ends the edit operation.
        '''
        return key_event.key() == Qt.Key_Escape

    def draw_state(self, painter):
        scale = self._rasterview.get_scale()
        p1_scaled = self._control_points[0] * scale
        p2_scaled = self._control_points[1] * scale

        # Draw a box between the two points, using a dotted rectangle.

        color = self._app_state.get_config('raster.selection.edit_outline')
        pen = QPen(color)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        rect = get_rectangle(p1_scaled, p2_scaled)
        painter.drawRect(rect)

        # Draw boxes on all control-points.
        color = self._app_state.get_config('raster.selection.edit_points')
        painter.setPen(QPen(color))
        for cp in self._control_points:
            cp_scaled = cp * scale
            painter.fillRect(cp_scaled.x() - CONTROL_POINT_SIZE / 2,
                             cp_scaled.y() - CONTROL_POINT_SIZE / 2,
                             CONTROL_POINT_SIZE, CONTROL_POINT_SIZE, color)

    def finish(self):
        # Signal that the ROI changed, so that everyone can be notified.
        self._rasterpane.roi_selection_changed.emit(self._roi, self._rect_sel)

        self._app_state.clear_status_text()

    def get_selection(self):
        return RectangleSelection(self._control_points[0], self._control_points[1])
