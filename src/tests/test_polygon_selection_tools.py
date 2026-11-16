"""
Tests for the polygon selection creator and editor, driven through the raster
view's mouse and key handlers at the default zoom.
"""

import unittest

import tests.context  # noqa: F401

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from test_utils.test_model import WiserTestModel
from wiser.gui.ui_selection_polygon import PolygonSelectionCreator, PolygonSelectionEditor
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import PolygonSelection, SelectionType


class _Action:
    def __init__(self, selection_type):
        self._selection_type = selection_type

    def data(self):
        return self._selection_type


@pytest.mark.functional
class TestPolygonSelectionTools(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel(use_gui=False)
        self.test_model.load_dataset(np.ones((3, 20, 20), dtype=np.float32))
        self.roi = RegionOfInterest(name="roi")
        self.test_model.app_state.add_roi(self.roi)
        self.pane = self.test_model.main_view
        self.rv = self.test_model.get_main_view_rv((0, 0))
        self.changed = []
        self.pane.roi_selection_changed.connect(lambda roi, sel: self.changed.append(sel))

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _mouse(self, event_type, raster_xy):
        scale = self.rv.get_scale()
        pos = QPointF((raster_xy[0] + 0.5) * scale, (raster_xy[1] + 0.5) * scale)
        buttons = Qt.MouseButtons(Qt.LeftButton) if event_type != QEvent.MouseButtonRelease else Qt.NoButton
        event = QMouseEvent(event_type, pos, Qt.LeftButton, buttons, Qt.NoModifier)
        widget = self.rv._image_widget
        if event_type == QEvent.MouseButtonPress:
            widget.mousePressEvent(event)
        elif event_type == QEvent.MouseMove:
            widget.mouseMoveEvent(event)
        else:
            widget.mouseReleaseEvent(event)

    def _click(self, raster_xy):
        self._mouse(QEvent.MouseButtonPress, raster_xy)
        self._mouse(QEvent.MouseButtonRelease, raster_xy)

    def _drag(self, start_xy, end_xy):
        self._mouse(QEvent.MouseButtonPress, start_xy)
        self._mouse(QEvent.MouseMove, end_xy)
        self._mouse(QEvent.MouseButtonRelease, end_xy)

    def _press_escape(self):
        event = QKeyEvent(QEvent.KeyRelease, Qt.Key_Escape, Qt.NoModifier)
        self.rv._image_widget.keyReleaseEvent(event)

    def _start_polygon(self):
        self.pane._on_add_selection_to_roi(_Action(SelectionType.POLYGON))

    def test_default_zoom_second_click_near_first_adds_vertex(self):
        self.assertEqual(self.rv.get_scale(), 1)
        self._start_polygon()
        self._click((5, 5))
        self._click((6, 6))
        self.assertIsInstance(self.pane._task_delegate, PolygonSelectionCreator)
        self.assertEqual([p.toTuple() for p in self.pane._task_delegate._points], [(5, 5), (6, 6)])
        self.assertEqual(self.roi.get_selections(), [])
        self.assertEqual(self.changed, [])

    def test_default_zoom_click_near_first_closes_three_point_polygon(self):
        self._start_polygon()
        self._click((5, 5))
        self._click((15, 5))
        self._click((15, 15))
        self._click((6, 6))
        self.assertIsNone(self.pane._task_delegate)
        sels = self.roi.get_selections()
        self.assertEqual(len(sels), 1)
        self.assertEqual([p.toTuple() for p in sels[0].get_points()], [(5, 5), (15, 5), (15, 15)])
        self.assertEqual(self.changed, sels)

    def test_edited_polygon_pixels_match_edited_points(self):
        sel = PolygonSelection([QPoint(2, 2), QPoint(10, 2), QPoint(10, 10)])
        self.roi.add_selection(sel)
        before = self.roi.get_all_pixels()
        self.pane._on_edit_roi_selection_geometry(self.roi, 0, self.rv)
        self.assertIsInstance(self.pane._task_delegate, PolygonSelectionEditor)
        self._drag((10, 10), (16, 17))
        self._press_escape()
        self.assertIsNone(self.pane._task_delegate)
        self.assertEqual([p.toTuple() for p in sel.get_points()], [(2, 2), (10, 2), (16, 17)])
        expected = PolygonSelection([QPoint(2, 2), QPoint(10, 2), QPoint(16, 17)]).get_all_pixels()
        self.assertNotEqual(expected, before)
        self.assertEqual(self.roi.get_all_pixels(), expected)


@pytest.mark.unit
def test_restored_polygon_pixels_match_restored_points():
    sel = PolygonSelection([QPoint(2, 2), QPoint(10, 2), QPoint(10, 10)])
    before = sel.get_all_pixels()
    sel.get_points()[2].setX(16)
    sel.get_points()[2].setY(17)
    assert sel.get_all_pixels() != before
    sel.get_points()[2].setX(10)
    sel.get_points()[2].setY(10)
    assert sel.get_all_pixels() == before


if __name__ == "__main__":
    unittest.main()
