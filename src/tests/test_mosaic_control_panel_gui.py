"""GUI smoke tests for the mosaic control panel (EPIC #629, issue #638).

Drive the real :class:`MosaicPane` controls — drag-to-reorder, resolution mode,
resampling method, and the band-metadata chooser — and assert they drive the
:class:`MosaicController` and :class:`MosaicView` correctly. The controller math is
unit-tested in ``test_mosaic_controller.py``; here we check the panel wiring end to end
through the real ingestion path (as the #635/#636 GUI tests do).
"""
import os
import tempfile
import unittest
from unittest import mock

from osgeo import gdal, osr
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

import numpy as np

import tests.context  # noqa: F401  (adds src/ to sys.path)
import wiser.gui.mosaic_view as mosaic_view
from wiser.gui.mosaic_view import _PIXEL_READ_DEBOUNCE_MS
from wiser.raster.mosaic_controller import ResolutionMode
from test_utils.test_model import WiserTestModel

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.integration,
]


def _write_tiff(path, origin, epsg=32611, pixel=30.0, size=20, collar=2, bands=3):
    """Write a small georeferenced GeoTIFF at ``origin`` with a nodata collar."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds = gdal.GetDriverByName("GTiff").Create(
        path,
        size,
        size,
        bands,
        gdal.GDT_Float32,
        options=["TILED=YES", "BLOCKXSIZE=16", "BLOCKYSIZE=16"],
    )
    ox, oy = origin
    ds.SetGeoTransform([ox, pixel, 0.0, oy, 0.0, -pixel])
    ds.SetProjection(srs.ExportToWkt())
    for b in range(1, bands + 1):
        arr = np.full((size, size), 7, dtype=np.float32)
        arr[:collar, :] = -9999
        arr[-collar:, :] = -9999
        arr[:, :collar] = -9999
        arr[:, -collar:] = -9999
        band = ds.GetRasterBand(b)
        band.WriteArray(arr)
        band.SetNoDataValue(-9999.0)
    ds.FlushCache()
    ds = None


class TestMosaicPaneGui(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _wait_for(self, predicate, timeout_ms=15000, step_ms=50):
        waited = 0
        while waited < timeout_ms:
            if predicate():
                return True
            QTest.qWait(step_ms)
            waited += step_ms
        return predicate()

    def _load(self, name, origin):
        path = os.path.join(self._tmp.name, name)
        _write_tiff(path, origin)
        return self.test_model.load_dataset(path)

    def _ingest(self, pane, controller, dataset, expected_count):
        index = pane._dataset_combo.findData(dataset.get_id())
        self.assertGreaterEqual(index, 0)
        pane._dataset_combo.setCurrentIndex(index)
        pane._add_scene_button.click()
        self.assertTrue(
            self._wait_for(lambda: controller.scene_count() == expected_count),
            "scene was not ingested within the timeout",
        )

    def _open_with_two_scenes(self):
        """Open the dialog and ingest two same-CRS scenes: bottom-to-top [a, b]."""
        ds_a = self._load("a.tif", origin=(400000.0, 3800000.0))
        ds_b = self._load("b.tif", origin=(400300.0, 3799700.0))
        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()
        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)
        return dlg, pane, controller

    # -- drag-to-reorder ------------------------------------------------------

    def test_drag_reorder_updates_controller_z_order(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        scenes_before = controller.get_scenes()  # [bottom, top]

        # The list shows top-first: row 0 = top scene, row 1 = bottom scene. Dragging
        # row 0 to the bottom is Qt's rowsMoved(parent, start=0, end=0, dest, row=2).
        pane._on_scene_rows_moved(None, 0, 0, None, 2)

        scenes_after = controller.get_scenes()
        # The top scene (visual row 0) is now the bottom of the z-order.
        self.assertIs(scenes_after[0], scenes_before[1])
        self.assertIs(scenes_after[1], scenes_before[0])
        self.test_model.close_seamless_mosaic_dialog()

    def test_drag_reorder_triggers_no_pixel_reads(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        view = pane.get_mosaic_view()
        view.resize(800, 600)

        real_render = mosaic_view.render_scene_argb
        with mock.patch.object(mosaic_view, "render_scene_argb", side_effect=real_render) as spy:
            view.grab()
            self.assertTrue(self._wait_for(lambda: len(view._tile_cache) > 0))
            spy.reset_mock()

            # A reorder through the drag handler is a pure restack — no GDAL reads.
            pane._on_scene_rows_moved(None, 0, 0, None, 2)
            view.grab()
            QTest.qWait(2 * _PIXEL_READ_DEBOUNCE_MS)  # past the debounce window
            self.assertEqual(spy.call_count, 0)
        self.test_model.close_seamless_mosaic_dialog()

    # -- resolution mode ------------------------------------------------------

    def test_resolution_mode_change_rebuilds_grid(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        grid_before = controller.build_common_grid()

        idx = pane._resolution_combo.findData(ResolutionMode.HIGHEST)
        self.assertGreaterEqual(idx, 0)
        pane._resolution_combo.setCurrentIndex(idx)

        self.assertEqual(controller.get_resolution_mode(), ResolutionMode.HIGHEST)
        # The handler rebuilds the grid, so the cached object is replaced.
        self.assertIsNot(controller.build_common_grid(), grid_before)
        self.test_model.close_seamless_mosaic_dialog()

    # -- band-metadata chooser -----------------------------------------------

    def test_band_metadata_selection_sets_source_without_changing_band_count(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        scenes = controller.get_scenes()  # [a(bottom), b(top)]
        band_counts_before = [s.dataset.num_bands() for s in scenes]

        # Select the bottom scene "a" as the canonical band-metadata source.
        combo = pane._band_metadata_combo
        target_row = next((i for i in range(combo.count()) if combo.itemData(i) is scenes[0]), None)
        self.assertIsNotNone(target_row)
        combo.setCurrentIndex(target_row)

        self.assertIs(controller.get_band_metadata_source(), scenes[0])
        # Labeling-only: no scene's band count changed.
        band_counts_after = [s.dataset.num_bands() for s in controller.get_scenes()]
        self.assertEqual(band_counts_after, band_counts_before)
        self.test_model.close_seamless_mosaic_dialog()

    # -- resampling method ----------------------------------------------------

    def test_non_nn_resampling_warns_and_invalidates_pixels(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        view = pane.get_mosaic_view()

        idx = pane._resample_combo.findData(gdal.GRA_Bilinear)
        self.assertGreaterEqual(idx, 0)
        with mock.patch.object(QMessageBox, "warning") as warn:
            pane._resample_combo.setCurrentIndex(idx)

        warn.assert_called_once()  # non-Nearest-Neighbor always warns
        self.assertEqual(controller.get_resample_alg(), gdal.GRA_Bilinear)
        self.assertTrue(view._pixels_dirty)  # forces a fresh read
        self.test_model.close_seamless_mosaic_dialog()

    def test_nearest_neighbor_resampling_does_not_warn(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        # Switch to Bilinear first (patched, since that path warns), then back to
        # Nearest Neighbor and assert the NN switch itself does not warn.
        with mock.patch.object(QMessageBox, "warning"):
            pane._resample_combo.setCurrentIndex(pane._resample_combo.findData(gdal.GRA_Bilinear))
        idx_nn = pane._resample_combo.findData(gdal.GRA_NearestNeighbour)
        with mock.patch.object(QMessageBox, "warning") as warn:
            pane._resample_combo.setCurrentIndex(idx_nn)
        warn.assert_not_called()
        self.assertEqual(controller.get_resample_alg(), gdal.GRA_NearestNeighbour)
        self.test_model.close_seamless_mosaic_dialog()


if __name__ == "__main__":
    unittest.main()
