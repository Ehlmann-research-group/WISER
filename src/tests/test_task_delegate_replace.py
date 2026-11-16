"""
Tests that starting a selection tool while another is active cancels the active
one: a creator adds nothing, and an editor restores its pre-edit geometry.
"""

import unittest

import tests.context  # noqa: F401

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from test_utils.test_model import WiserTestModel
from wiser.gui.ui_selection_multi_pixel import (
    MultiPixelSelectionCreator,
    MultiPixelSelectionEditor,
)
from wiser.gui.ui_selection_polygon import PolygonSelectionCreator, PolygonSelectionEditor
from wiser.gui.ui_selection_rectangle import RectangleSelectionCreator, RectangleSelectionEditor
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import (
    MultiPixelSelection,
    PolygonSelection,
    RectangleSelection,
    SelectionType,
)


class _Action:
    def __init__(self, selection_type):
        self._selection_type = selection_type

    def data(self):
        return self._selection_type


@pytest.mark.functional
class TestTaskDelegateReplace(unittest.TestCase):
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

    def _mouse(self, event_type, raster_xy, rv=None):
        rv = self.rv if rv is None else rv
        scale = rv.get_scale()
        pos = QPointF((raster_xy[0] + 0.5) * scale, (raster_xy[1] + 0.5) * scale)
        buttons = Qt.MouseButtons(Qt.LeftButton) if event_type != QEvent.MouseButtonRelease else Qt.NoButton
        event = QMouseEvent(event_type, pos, Qt.LeftButton, buttons, Qt.NoModifier)
        widget = rv._image_widget
        if event_type == QEvent.MouseButtonPress:
            widget.mousePressEvent(event)
        elif event_type == QEvent.MouseMove:
            widget.mouseMoveEvent(event)
        else:
            widget.mouseReleaseEvent(event)

    def _click(self, raster_xy, rv=None):
        self._mouse(QEvent.MouseButtonPress, raster_xy, rv)
        self._mouse(QEvent.MouseButtonRelease, raster_xy, rv)

    def _drag(self, start_xy, end_xy):
        self._mouse(QEvent.MouseButtonPress, start_xy)
        self._mouse(QEvent.MouseMove, end_xy)
        self._mouse(QEvent.MouseButtonRelease, end_xy)

    def _start(self, selection_type):
        self.pane._on_add_selection_to_roi(_Action(selection_type))

    def _edit(self, sel, rv=None):
        index = self.roi.get_selections().index(sel)
        self.pane._on_edit_roi_selection_geometry(self.roi, index, self.rv if rv is None else rv)

    def test_polygon_creator_replaced_adds_nothing(self):
        self.rv.scale_image(8)
        self._start(SelectionType.POLYGON)
        self._click((2, 2))
        self._click((8, 2))
        self._click((8, 8))
        self._start(SelectionType.RECTANGLE)
        self.assertEqual(self.roi.get_selections(), [])
        self.assertEqual(self.changed, [])
        self.assertIsInstance(self.pane._task_delegate, RectangleSelectionCreator)

    def test_rectangle_creator_replaced_mid_drag_adds_nothing(self):
        self._start(SelectionType.RECTANGLE)
        self._mouse(QEvent.MouseButtonPress, (2, 2))
        self._mouse(QEvent.MouseMove, (6, 6))
        self._start(SelectionType.POLYGON)
        self.assertEqual(self.roi.get_selections(), [])
        self.assertEqual(self.changed, [])
        self.assertIsInstance(self.pane._task_delegate, PolygonSelectionCreator)

    def test_multi_pixel_creator_replaced_adds_nothing(self):
        self._start(SelectionType.MULTI_PIXEL)
        self._click((1, 1))
        self._click((3, 3))
        self._start(SelectionType.RECTANGLE)
        self.assertEqual(self.roi.get_selections(), [])
        self.assertEqual(self.changed, [])
        self.assertIsInstance(self.pane._task_delegate, RectangleSelectionCreator)

    def test_multi_pixel_editor_replaced_restores_pixels(self):
        sel = MultiPixelSelection([QPoint(1, 1), QPoint(5, 5)])
        self.roi.add_selection(sel)
        self._edit(sel)
        self.assertIsInstance(self.pane._task_delegate, MultiPixelSelectionEditor)
        self._click((1, 1))
        self._click((3, 3))
        self._start(SelectionType.POLYGON)
        self.assertEqual(sel.get_all_pixels(), {(1, 1), (5, 5)})
        self.assertEqual(self.changed, [sel])
        self.assertIsInstance(self.pane._task_delegate, PolygonSelectionCreator)
        self.assertTrue(
            self.test_model.main_window.statusBar().currentMessage().startswith("Left-click to add points")
        )

    def test_rectangle_editor_replaced_restores_corners(self):
        sel = RectangleSelection(QPoint(2, 2), QPoint(8, 8))
        self.roi.add_selection(sel)
        self._edit(sel)
        self.assertIsInstance(self.pane._task_delegate, RectangleSelectionEditor)
        self._drag((8, 8), (12, 14))
        self.assertEqual(sel.get_bottom_right().toTuple(), (12, 14))
        self._start(SelectionType.MULTI_PIXEL)
        self.assertEqual(sel.get_top_left().toTuple(), (2, 2))
        self.assertEqual(sel.get_bottom_right().toTuple(), (8, 8))
        self.assertEqual(self.changed, [sel])
        self.assertIsInstance(self.pane._task_delegate, MultiPixelSelectionCreator)

    def test_polygon_editor_replaced_restores_points(self):
        sel = PolygonSelection([QPoint(2, 2), QPoint(10, 2), QPoint(10, 10)])
        self.roi.add_selection(sel)
        self._edit(sel)
        self.assertIsInstance(self.pane._task_delegate, PolygonSelectionEditor)
        self._drag((10, 10), (15, 16))
        self.assertEqual(sel.get_points()[2].toTuple(), (15, 16))
        self._start(SelectionType.RECTANGLE)
        self.assertEqual([p.toTuple() for p in sel.get_points()], [(2, 2), (10, 2), (10, 10)])
        self.assertEqual(self.changed, [sel])
        self.assertIsInstance(self.pane._task_delegate, RectangleSelectionCreator)

    def test_edit_geometry_from_another_view_restores_first_edit(self):
        self.test_model.set_main_view_layout((1, 2))
        rv0 = self.test_model.get_main_view_rv((0, 0))
        rv1 = self.test_model.get_main_view_rv((0, 1))
        self.rv = rv0
        mp_sel = MultiPixelSelection([QPoint(1, 1), QPoint(5, 5)])
        rect_sel = RectangleSelection(QPoint(2, 2), QPoint(8, 8))
        self.roi.add_selection(mp_sel)
        self.roi.add_selection(rect_sel)
        self._edit(mp_sel, rv0)
        self._click((1, 1), rv0)
        self._click((3, 3), rv0)
        self._edit(rect_sel, rv1)
        self.assertEqual(mp_sel.get_all_pixels(), {(1, 1), (5, 5)})
        self.assertEqual(self.changed, [mp_sel])
        self.assertIsInstance(self.pane._task_delegate, RectangleSelectionEditor)
        self.assertIs(self.pane._task_delegate.get_rasterview(), rv1)
