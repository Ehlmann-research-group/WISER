"""GUI tests for the Stretch Builder across a band-count change (issues #780, #781).

A view's stretches are looked up per ``(dataset, band)`` on ApplicationState.  When a view
switches between grayscale and RGB, it must hold one stretch per display band, so that
reopening the Stretch Builder loads a stretch for every channel; and redrawing a channel
histogram must not try to remove stretch lines that were already cleared from its axes.
"""

import unittest

import numpy as np

import tests.context  # noqa: F401

from PySide6.QtWidgets import QApplication

from test_utils.test_model import WiserTestModel


class TestStretchBandCountChange(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        self.app_state = self.test_model.app_state
        cube = np.arange(3 * 6 * 5, dtype=np.float32).reshape(3, 6, 5)
        self.ds = self.test_model.load_dataset(cube)
        QApplication.processEvents()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _set_bands(self, bands):
        self.test_model.main_window._on_display_bands_change(self.ds.get_id(), bands, None, True)
        QApplication.processEvents()

    def _assert_view_stretches_match_app_state(self, bands):
        rv = self.test_model.get_main_view_rv()
        self.assertEqual(rv.get_display_bands(), bands)
        self.assertEqual(rv.get_stretches(), self.app_state.get_stretches(self.ds.get_id(), bands))

    def test_grayscale_then_rgb_reopens_stretch_builder(self):
        self._set_bands((0,))
        self.test_model.click_stretch_linear_2_5()
        self.test_model.close_stretch_builder()

        self._set_bands((0, 1, 2))
        self._assert_view_stretches_match_app_state((0, 1, 2))
        self.test_model.click_stretch_linear_2_5()
        self.test_model.close_stretch_builder()

        stretches = self.app_state.get_stretches(self.ds.get_id(), (0, 1, 2))
        self.assertTrue(all(s is not None for s in stretches))

    def test_rgb_then_grayscale_reopens_stretch_builder(self):
        self._set_bands((0, 1, 2))
        self.test_model.click_stretch_linear_2_5()
        self.test_model.close_stretch_builder()

        self._set_bands((1,))
        self._assert_view_stretches_match_app_state((1,))
        self.test_model.click_stretch_linear_2_5()
        self.test_model.close_stretch_builder()

        self.assertIsNotNone(self.app_state.get_stretches(self.ds.get_id(), (1,))[0])

    def test_show_rejects_stretches_not_matching_display_bands(self):
        builder = self.test_model.get_stretch_builder()
        with self.assertRaises(ValueError):
            builder.show(self.ds, (0, 1, 2), [None])

    def test_histogram_redraw_after_axes_cleared_does_not_remove_detached_lines(self):
        self._set_bands((0, 1, 2))
        self.test_model.click_stretch_linear_2_5()
        channel = self.test_model.get_channel_stretch(0)

        channel._draw_stretch_lines = False
        channel._show_histogram()
        channel._draw_stretch_lines = True
        channel._show_histogram(update_lines_only=True)

        self.assertIs(channel._low_line.axes, channel._histogram_axes)
        self.assertIs(channel._high_line.axes, channel._histogram_axes)


if __name__ == "__main__":
    unittest.main()
