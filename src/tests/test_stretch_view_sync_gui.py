"""GUI sync test: clearing a stretch resets the view image (issue #689).

A contrast stretch applied to a ``(dataset, band)`` is stored on ApplicationState and
reaches the displayed image through the ``stretch_changed`` signal.  This test drives
the real path (offscreen ``WiserTestModel``): a stretch applied through the Stretch
Builder changes the rendered image, and clearing the dataset's stretches on
ApplicationState -- setting them back to ``None`` -- reverts the image to its
unstretched baseline and leaves no stretch on the dataset or the view.
"""

import unittest

import numpy as np

import tests.context  # noqa: F401

from PySide6.QtWidgets import QApplication

from test_utils.test_model import WiserTestModel


class TestStretchViewSync(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        self.app_state = self.test_model.app_state

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _load_gradient_dataset(self):
        band = [
            [0.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75, 0.75],
            [1.0, 1.0, 1.0, 1.0],
        ]
        cube = np.array([band, band, band])
        return self.test_model.load_dataset(cube)

    def test_clearing_stretch_resets_view_image(self):
        ds = self._load_gradient_dataset()
        ds_id = ds.get_id()
        QApplication.processEvents()

        rv = self.test_model.get_main_view_rv()
        self.assertIsNotNone(rv.get_raster_data())
        self.assertEqual(rv.get_raster_data().get_id(), ds_id)
        bands = rv.get_display_bands()
        self.assertIsNotNone(bands)

        baseline = self.test_model.get_main_view_rv_image_data()
        self.assertIsNotNone(baseline)
        baseline = baseline.copy()

        # Apply a stretch through the real Stretch Builder path; the view re-renders.
        self.test_model.click_stretch_hist_equalize()
        self.test_model.click_log_conditioner()
        with_stretch = self.test_model.get_main_view_rv_image_data().copy()
        self.assertFalse(np.array_equal(baseline, with_stretch))
        self.assertTrue(any(s is not None for s in self.app_state.get_stretches(ds_id, bands)))

        # Clearing the dataset's stretches on app state resets the view to baseline.
        self.app_state.set_stretches(ds_id, bands, [None] * len(bands))
        QApplication.processEvents()
        after_clear = self.test_model.get_main_view_rv_image_data().copy()
        self.assertTrue(np.array_equal(after_clear, baseline))
        self.assertTrue(all(s is None for s in self.app_state.get_stretches(ds_id, bands)))
        self.assertTrue(all(s is None for s in rv.get_stretches()))


if __name__ == "__main__":
    unittest.main()
