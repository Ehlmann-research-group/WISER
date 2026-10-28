"""Integration tests for display bands and stretches across main-view dataset switches.

Switching the main view's dataset must not reset the display bands, or the
per-band stretches keyed on them, that the user chose for a dataset in any
raster pane (issue #645).
"""

import unittest
from pathlib import Path

import tests.context  # noqa: F401

import numpy as np
import pytest
from PySide6.QtGui import QAction

from test_utils.test_model import WiserTestModel
from wiser.raster.stretch import StretchLinear
from wiser.raster.dataset import find_display_bands

pytestmark = [pytest.mark.integration]

_DATASETS = Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets"
DATASET_A = str(_DATASETS / "jpl_15_7_7.hdr")
DATASET_B = str(_DATASETS / "caltech_4_100_150_nm.hdr")


class TestDisplayBandsDatasetSwitch(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _panes(self):
        return {
            "main_view": self.test_model.main_view,
            "zoom_pane": self.test_model.zoom_pane,
            "context_pane": self.test_model.context_pane,
        }

    def _switch_main_view(self, ds_id: int):
        action = QAction()
        action.setData(((0, 0), ds_id))
        self.test_model.main_view._on_dataset_changed(action)
        self.test_model.run()

    def test_bands_and_stretches_survive_main_view_switch(self):
        ds_a = self.test_model.load_dataset(DATASET_A)
        ds_b = self.test_model.load_dataset(DATASET_B)
        a_id = ds_a.get_id()
        self._switch_main_view(a_id)

        bands = (10, 5, 2)
        self.assertNotEqual(bands, tuple(find_display_bands(ds_a)))
        self.test_model.main_window._on_display_bands_change(a_id, bands, None, True)
        stretches = [StretchLinear(0.1 * i, 0.5 + 0.1 * i) for i in range(len(bands))]
        self.test_model.app_state.set_stretches(a_id, bands, stretches)
        self.test_model.run()

        for _ in range(2):
            self._switch_main_view(ds_b.get_id())
            self._switch_main_view(a_id)

        for name, pane in self._panes().items():
            with self.subTest(pane=name):
                self.assertEqual(tuple(pane._display_bands[a_id]), bands)
                rv = pane.get_rasterview()
                self.assertIs(rv.get_raster_data(), ds_a)
                self.assertEqual(tuple(rv.get_display_bands()), bands)
                self.assertEqual(rv.get_stretches(), stretches)

    def test_added_dataset_gets_default_bands(self):
        self.test_model.load_dataset(DATASET_A)
        ds_b = self.test_model.load_dataset(DATASET_B)
        expected = tuple(find_display_bands(ds_b))

        for name, pane in self._panes().items():
            with self.subTest(pane=name):
                self.assertEqual(tuple(pane._display_bands[ds_b.get_id()]), expected)

    def test_reused_dataset_id_gets_default_bands(self):
        ds_a = self.test_model.load_dataset(DATASET_A)
        a_id = ds_a.get_id()
        bands = (10, 5, 2)
        self.test_model.main_window._on_display_bands_change(a_id, bands, None, True)
        self.test_model.run()

        self.test_model.app_state.clear_session()
        self.test_model.run()
        ds_b = self.test_model.load_dataset(DATASET_B)
        self.assertEqual(ds_b.get_id(), a_id)
        expected = tuple(find_display_bands(ds_b))
        self.assertNotEqual(expected, bands)

        for name, pane in self._panes().items():
            with self.subTest(pane=name):
                self.assertEqual(tuple(pane._display_bands[a_id]), expected)

    def test_colormap_survives_main_view_switch(self):
        ds_gray = self.test_model.load_dataset(
            np.linspace(0.0, 1.0, 600, dtype=np.float32).reshape(1, 20, 30)
        )
        ds_b = self.test_model.load_dataset(DATASET_B)
        gray_id = ds_gray.get_id()
        self._switch_main_view(gray_id)
        self.test_model.main_window._on_display_bands_change(gray_id, (0,), "viridis", True)
        self.test_model.run()

        for _ in range(2):
            self._switch_main_view(ds_b.get_id())
            self._switch_main_view(gray_id)

        for name, pane in self._panes().items():
            with self.subTest(pane=name):
                rv = pane.get_rasterview()
                self.assertIs(rv.get_raster_data(), ds_gray)
                self.assertEqual(rv.get_colormap(), "viridis")

    def test_reopened_dataset_has_no_stale_colormap(self):
        arr = np.linspace(0.0, 1.0, 600, dtype=np.float32).reshape(1, 20, 30)
        ds_gray = self.test_model.load_dataset(arr)
        self.test_model.load_dataset(DATASET_B)
        self.test_model.main_window._on_display_bands_change(ds_gray.get_id(), (0,), "viridis", True)
        self.test_model.run()

        self.test_model.app_state.clear_session()
        self.test_model.run()
        ds_new = self.test_model.load_dataset(arr)
        self.assertEqual(ds_new.get_id(), ds_gray.get_id())

        for name, pane in self._panes().items():
            with self.subTest(pane=name):
                rv = pane.get_rasterview()
                self.assertIs(rv.get_raster_data(), ds_new)
                self.assertIsNone(rv.get_colormap())


if __name__ == "__main__":
    unittest.main()
