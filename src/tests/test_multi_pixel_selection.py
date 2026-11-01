"""
Tests the multi-pixel selection tool: left-click toggles pixels, Enter or
right-click finishes, Esc cancels, and picking hits only the selected pixels.
"""

import unittest

import tests.context  # noqa: F401

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QContextMenuEvent, QKeyEvent, QMouseEvent

from test_utils.test_model import WiserTestModel
from wiser.gui.ui_roi import get_picked_roi_selections
from wiser.gui.ui_selection_multi_pixel import (
    MultiPixelSelectionCreator,
    MultiPixelSelectionEditor,
)
from wiser.gui.ui_selection_rectangle import RectangleSelectionCreator
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import MultiPixelSelection, RectangleSelection


@pytest.mark.functional
class TestMultiPixelSelection(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel(use_gui=False)
        self.test_model.load_dataset(np.ones((3, 20, 20), dtype=np.float32))
        self.roi = RegionOfInterest(name="roi")
        self.test_model.app_state.add_roi(self.roi)
        self.pane = self.test_model.main_view
        self.rv = self.test_model.get_main_view_rv((0, 0))
        self.widget = self.rv._image_widget
        self.changed = []
        self.pane.roi_selection_changed.connect(lambda roi, sel: self.changed.append(sel))

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _display_pos(self, raster_xy):
        scale = self.rv.get_scale()
        return QPointF((raster_xy[0] + 0.5) * scale, (raster_xy[1] + 0.5) * scale)

    def _click(self, raster_xy, button=Qt.LeftButton):
        event = QMouseEvent(
            QEvent.MouseButtonRelease,
            self._display_pos(raster_xy),
            button,
            Qt.MouseButtons(button),
            Qt.NoModifier,
        )
        self.widget.mouseReleaseEvent(event)

    def _right_click_menu(self, raster_xy):
        display = self._display_pos(raster_xy)
        self._right_click_menu_at(QPoint(int(display.x()), int(display.y())))

    def _right_click_menu_at(self, pos):
        event = QContextMenuEvent(QContextMenuEvent.Mouse, pos, pos)
        self.pane._onRasterContextMenu(self.rv, event, testing=True)

    def _key(self, key):
        self.widget.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease, key, Qt.NoModifier))

    def _start_creator(self):
        self.pane._task_delegate = MultiPixelSelectionCreator(self.pane)

    def _committed_pixels(self):
        sels = self.roi.get_selections()
        self.assertEqual(len(sels), 1)
        return sels[0].get_all_pixels()

    def test_left_click_toggles_and_right_click_does_not(self):
        self._start_creator()
        self._click((2, 3))
        self._click((5, 5))
        self._click((5, 5))
        self._click((7, 7), button=Qt.RightButton)
        self.assertEqual({p.toTuple() for p in self.pane._task_delegate._points}, {(2, 3)})

    def test_right_click_finishes_without_context_menu(self):
        self._start_creator()
        self._click((2, 3))
        self._click((9, 9))
        menu_before = getattr(self.pane, "_menu", None)
        self._click((4, 4), button=Qt.RightButton)
        self._right_click_menu((4, 4))
        self.assertIsNone(self.pane._task_delegate)
        self.assertIs(getattr(self.pane, "_menu", None), menu_before)
        self.assertEqual(self._committed_pixels(), {(2, 3), (9, 9)})
        self.assertEqual(len(self.changed), 1)

    def test_right_click_press_triggered_menu_finishes(self):
        clicked = []
        self.pane.click_pixel.connect(lambda pos, coord: clicked.append(coord))
        self._start_creator()
        self._click((2, 3))
        self._right_click_menu((4, 4))
        self._click((4, 4), button=Qt.RightButton)
        self.assertIsNone(self.pane._task_delegate)
        self.assertEqual(self._committed_pixels(), {(2, 3)})
        self.assertEqual(clicked, [])

    def test_enter_and_keypad_enter_finish(self):
        for count, key in enumerate((Qt.Key_Return, Qt.Key_Enter), start=1):
            with self.subTest(key=key):
                self._start_creator()
                self._click((count, count))
                self._key(key)
                self.assertIsNone(self.pane._task_delegate)
                sels = self.roi.get_selections()
                self.assertEqual(len(sels), count)
                self.assertEqual(sels[-1].get_all_pixels(), {(count, count)})

    def test_escape_cancels(self):
        self._start_creator()
        self._click((1, 1))
        self._click((2, 2))
        self._key(Qt.Key_Escape)
        self.assertIsNone(self.pane._task_delegate)
        self.assertEqual(self.roi.get_selections(), [])
        self.assertEqual(self.changed, [])

    def test_finish_with_no_pixels_adds_nothing(self):
        self._start_creator()
        self._click((1, 1))
        self._click((1, 1))
        self._key(Qt.Key_Return)
        self.assertIsNone(self.pane._task_delegate)
        self.assertEqual(self.roi.get_selections(), [])
        self.assertEqual(self.changed, [])

    def test_editor_escape_restores_pixels(self):
        sel = MultiPixelSelection([QPoint(1, 1), QPoint(5, 5)])
        self.roi.add_selection(sel)
        self.pane._task_delegate = MultiPixelSelectionEditor(self.roi, sel, self.pane, self.rv)
        self._click((1, 1))
        self._click((3, 3))
        self._key(Qt.Key_Escape)
        self.assertIsNone(self.pane._task_delegate)
        self.assertEqual(sel.get_all_pixels(), {(1, 1), (5, 5)})
        self.assertEqual(self.changed, [])

    def test_editor_enter_commits(self):
        sel = MultiPixelSelection([QPoint(1, 1), QPoint(5, 5)])
        self.roi.add_selection(sel)
        self.pane._task_delegate = MultiPixelSelectionEditor(self.roi, sel, self.pane, self.rv)
        self._click((1, 1))
        self._click((3, 3))
        self._key(Qt.Key_Return)
        self.assertEqual(sel.get_all_pixels(), {(3, 3), (5, 5)})
        self.assertEqual(self.changed, [sel])

    def test_editor_empty_finish_keeps_pixels(self):
        sel = MultiPixelSelection([QPoint(1, 1), QPoint(5, 5)])
        self.roi.add_selection(sel)
        self.pane._task_delegate = MultiPixelSelectionEditor(self.roi, sel, self.pane, self.rv)
        self._click((1, 1))
        self._click((5, 5))
        self._key(Qt.Key_Return)
        self.assertIsNone(self.pane._task_delegate)
        self.assertEqual(sel.get_all_pixels(), {(1, 1), (5, 5)})
        self.assertEqual(self.changed, [])

    def _context_menu_roi_titles(self, raster_xy):
        self._right_click_menu(raster_xy)
        return self._menu_roi_titles()

    def _context_menu_roi_titles_at(self, pos):
        self._right_click_menu_at(pos)
        return self._menu_roi_titles()

    def _menu_roi_titles(self):
        return [a.text() for a in self.pane._menu.actions() if a.menu() is not None]

    def test_context_menu_picks_selected_pixel_not_gap(self):
        self.roi.add_selection(MultiPixelSelection([QPoint(1, 1), QPoint(9, 9)]))
        self.assertIn("roi", self._context_menu_roi_titles((9, 9)))
        self.assertNotIn("roi", self._context_menu_roi_titles((5, 5)))

    def test_context_menu_picks_selected_pixel_when_zoomed_out(self):
        self.roi.add_selection(MultiPixelSelection([QPoint(9, 9)]))
        self.rv.scale_image(0.25)
        self.assertIn("roi", self._context_menu_roi_titles((9, 9)))
        self.assertNotIn("roi", self._context_menu_roi_titles((1, 1)))

    def test_context_menu_picks_marker_at_non_integer_zoom(self):
        self.roi.add_selection(MultiPixelSelection([QPoint(9, 9)]))
        scale = 1 / 2.3
        self.rv.scale_image(scale)
        marker = int(9 * scale)
        self.assertIn("roi", self._context_menu_roi_titles_at(QPoint(marker, marker)))
        self.assertNotIn("roi", self._context_menu_roi_titles_at(QPoint(marker + 2, marker + 2)))

    def test_zoomed_out_rectangle_picking_unchanged(self):
        self.roi.add_selection(RectangleSelection(QPoint(5, 5), QPoint(7, 7)))
        self.rv.scale_image(0.25)
        self.assertNotIn("roi", self._context_menu_roi_titles_at(QPoint(1, 1)))

    def test_rectangle_creator_right_click_still_opens_context_menu(self):
        self.pane._task_delegate = RectangleSelectionCreator(self.pane)
        menu_before = getattr(self.pane, "_menu", None)
        self._right_click_menu((4, 4))
        self.assertIsNotNone(self.pane._task_delegate)
        self.assertIsNot(self.pane._menu, menu_before)


@pytest.mark.unit
def test_multi_pixel_picking_hits_pixels_not_gap():
    sel = MultiPixelSelection([QPoint(0, 0), QPoint(10, 10)])
    assert sel.is_picked_by(QPoint(0, 0))
    assert sel.is_picked_by(QPoint(10, 10))
    assert not sel.is_picked_by(QPoint(5, 5))
    assert not sel.is_picked_by(QPoint(1, 0))

    roi = RegionOfInterest(name="roi")
    roi.add_selection(sel)
    assert get_picked_roi_selections(roi, QPoint(10, 10)) == [0]
    assert get_picked_roi_selections(roi, QPoint(5, 5)) == []
