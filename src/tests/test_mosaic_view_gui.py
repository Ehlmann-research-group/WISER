"""GUI smoke tests for the mosaic vector overlay (EPIC #629, issue #636).

Drive real scene ingestion through a :class:`MosaicPane` (as the #635 CRS tests do),
then assert the :class:`MosaicView` overlay geometry cache populates and invalidates
correctly. The overlap-geometry math itself is unit-tested in
``test_mosaic_controller.py``; here we check the GUI wiring end to end.
"""
import os
import tempfile
import unittest
from unittest import mock

from osgeo import gdal, osr
from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

import numpy as np

import tests.context  # noqa: F401  (adds src/ to sys.path)
import wiser.gui.mosaic_view as mosaic_view
from wiser.gui.mosaic_view import MosaicView, _PIXEL_READ_DEBOUNCE_MS
from test_utils.test_model import WiserTestModel

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.integration,
]


def _write_overlapping_tiff(path, origin, epsg=32611, pixel=30.0, size=20, collar=2, bands=3):
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


class TestMosaicViewGui(unittest.TestCase):
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
        _write_overlapping_tiff(path, origin)
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

    def test_overlay_geometry_populates_for_overlapping_scenes(self):
        # Two same-CRS scenes whose valid footprints overlap in one corner.
        ds_a = self._load("a.tif", origin=(400000.0, 3800000.0))
        ds_b = self._load("b.tif", origin=(400300.0, 3799700.0))

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()
        view = pane.get_mosaic_view()

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)

        # Ingestion invalidates the overlay; the rebuild is what paintEvent runs.
        self.assertTrue(view._geometry_dirty)
        view._rebuild_overlay_geometry()

        self.assertEqual(len(view._footprint_paths), 2)
        self.assertEqual(len(view._hidden_paths), 2)
        self.assertIsNotNone(view._bbox_extent)
        # The scenes overlap, so at least one scene has a hidden region.
        self.assertTrue(any(p is not None for p in view._hidden_paths))
        # Every footprint path has real geometry.
        self.assertTrue(all(not p.isEmpty() for p in view._footprint_paths))

        self.test_model.close_seamless_mosaic_dialog()

    def test_overlay_invalidates_on_scene_removal(self):
        ds_a = self._load("a.tif", origin=(400000.0, 3800000.0))
        ds_b = self._load("b.tif", origin=(400300.0, 3799700.0))

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()
        view = pane.get_mosaic_view()

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)

        # Simulate a completed paint clearing the dirty flag.
        view._rebuild_overlay_geometry()
        view._geometry_dirty = False

        # Removing a scene must re-dirty the overlay so it is rebuilt next paint.
        pane._scene_list.setCurrentRow(0)
        pane._on_remove_scene_clicked()
        self.assertTrue(view._geometry_dirty)

    def test_overlay_invalidates_on_target_crs_change(self):
        ds_a = self._load("a.tif", origin=(400000.0, 3800000.0))
        ds_b = self._load("b.tif", origin=(400300.0, 3799700.0))

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()
        view = pane.get_mosaic_view()

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)

        view._rebuild_overlay_geometry()
        view._geometry_dirty = False

        target_geo = osr.SpatialReference()
        target_geo.ImportFromEPSG(4326)
        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog") as Dlg:
            inst = Dlg.return_value
            inst.exec.return_value = QDialog.Accepted
            inst.selected_target_wkt.return_value = target_geo.ExportToWkt()
            pane._on_choose_target_crs()

        self.assertTrue(view._geometry_dirty)

        self.test_model.close_seamless_mosaic_dialog()

    def test_camera_reframes_after_emptying_and_readding(self):
        # Regression: add a scene, remove it (mosaic empties), then add a *different*
        # scene far away. The camera must re-frame onto the new scene rather than stay
        # parked on the removed scene's extent (which left the new footprint off-screen).
        ds_a = self._load("a.tif", origin=(400000.0, 3800000.0))
        ds_b = self._load("b.tif", origin=(700000.0, 3500000.0))  # far from A

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()
        view = pane.get_mosaic_view()
        view.resize(800, 600)

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
        view.grab()  # force a paint -> fit camera to A
        self.assertTrue(view._has_fitted)

        # Remove A: the mosaic is now empty, which re-arms the fit.
        pane._scene_list.setCurrentRow(0)
        pane._on_remove_scene_clicked()
        view.grab()
        self.assertFalse(view._has_fitted)

        # Add B far away: the camera re-frames onto it.
        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_b, 1)
        view.grab()

        self.assertTrue(view._has_fitted)
        min_x, min_y, max_x, max_y = controller.get_common_grid().extent
        cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        # Camera centered on B's extent...
        self.assertAlmostEqual(view._transform.center_x, cx, places=3)
        self.assertAlmostEqual(view._transform.center_y, cy, places=3)
        # ...and B's footprint maps on-screen (the user-visible symptom).
        screen = view._transform.world_to_screen(view.size()).map(QPointF(cx, cy))
        self.assertGreaterEqual(screen.x(), 0.0)
        self.assertLessEqual(screen.x(), 800.0)
        self.assertGreaterEqual(screen.y(), 0.0)
        self.assertLessEqual(screen.y(), 600.0)

        self.test_model.close_seamless_mosaic_dialog()

    # -- pixel layer (#637) ---------------------------------------------------

    def test_composite_stacks_bottom_to_top_with_alpha(self):
        """composite() stacks known ARGB layers: top opaque wins, holes reveal below."""
        view = MosaicView()  # QApplication exists (WiserTestModel); no controller needed
        w = h = 4
        bottom = QImage(w, h, QImage.Format_ARGB32)
        bottom.fill(QColor(255, 0, 0, 255))  # opaque red
        top = QImage(w, h, QImage.Format_ARGB32)
        top.fill(Qt.transparent)
        painter = QPainter(top)
        painter.fillRect(QRect(0, 0, 2, h), QColor(0, 0, 255, 255))  # left half opaque blue
        painter.end()

        result = view.composite([bottom, top])  # bottom-to-top
        self.assertIsNotNone(result)
        # Left: top (blue) is opaque, so it wins.
        self.assertEqual(result.pixelColor(0, 0), QColor(0, 0, 255, 255))
        # Right: top is a transparent hole, so the bottom (red) shows through.
        self.assertEqual(result.pixelColor(3, 0), QColor(255, 0, 0, 255))

        # Nothing drawable -> None (hidden scenes contribute None entries).
        self.assertIsNone(view.composite([None, None]))
        self.assertIsNone(view.composite([]))

    def test_scene_layers_populate_and_composite(self):
        ds_a = self._load("a.tif", origin=(400000.0, 3800000.0))
        ds_b = self._load("b.tif", origin=(400300.0, 3799700.0))

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()
        view = pane.get_mosaic_view()
        view.resize(800, 600)

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)

        # The read is off the UI thread and debounced, so force a paint to schedule it
        # then pump the loop until the layers arrive.
        view.grab()
        self.assertTrue(
            self._wait_for(lambda: len(view._scene_layers) == 2),
            "per-scene layers were not read within the timeout",
        )
        self.assertIsNotNone(view._composite_pixmap)
        self.assertIsNotNone(view._composite_world_extent)

        self.test_model.close_seamless_mosaic_dialog()

    def test_reorder_and_visibility_trigger_no_reads(self):
        ds_a = self._load("a.tif", origin=(400000.0, 3800000.0))
        ds_b = self._load("b.tif", origin=(400300.0, 3799700.0))

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()
        view = pane.get_mosaic_view()
        view.resize(800, 600)

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)

        real_render = mosaic_view.render_scene_argb
        with mock.patch.object(mosaic_view, "render_scene_argb", side_effect=real_render) as spy:
            # Initial debounced read populates the cache.
            view.grab()
            self.assertTrue(self._wait_for(lambda: len(view._scene_layers) == 2))
            self.assertGreater(spy.call_count, 0)

            def _assert_no_read(mutate):
                spy.reset_mock()
                mutate()
                view.recomposite_only()
                view.grab()
                QTest.qWait(2 * _PIXEL_READ_DEBOUNCE_MS)  # past the debounce window — nothing should read
                self.assertEqual(spy.call_count, 0)

            # Z-order reorder, hide, and unhide-still-cached are all pure restacks.
            _assert_no_read(lambda: controller.move_scene(0, 1))
            _assert_no_read(lambda: controller.set_visibility(0, False))
            _assert_no_read(lambda: controller.set_visibility(0, True))

        self.test_model.close_seamless_mosaic_dialog()

    def test_pan_debounces_into_a_single_read(self):
        ds_a = self._load("a.tif", origin=(400000.0, 3800000.0))
        ds_b = self._load("b.tif", origin=(400300.0, 3799700.0))

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()
        view = pane.get_mosaic_view()
        view.resize(800, 600)

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)

        view.grab()
        self.assertTrue(self._wait_for(lambda: len(view._scene_layers) == 2))

        # A burst of pans must collapse into one background read, not one per event.
        real_worker = mosaic_view._render_scene_layers
        with mock.patch.object(mosaic_view, "_render_scene_layers", side_effect=real_worker) as worker_spy:
            initial_extent = view._composite_world_extent
            for _ in range(5):
                view._transform.pan(40, 0)
                view.grab()  # each paint restarts the debounce timer
                QTest.qWait(_PIXEL_READ_DEBOUNCE_MS / 10)  # well under the debounce window
            # Still mid-gesture: the debounce timer keeps restarting, so no read yet.
            self.assertEqual(worker_spy.call_count, 0)

            # Once the gesture settles, the composite updates to the panned viewport...
            self.assertTrue(self._wait_for(lambda: view._composite_world_extent != initial_extent))
            # ...via a single coalesced read, not one per pan event.
            QTest.qWait(2 * _PIXEL_READ_DEBOUNCE_MS)
            self.assertEqual(worker_spy.call_count, 1)

        self.test_model.close_seamless_mosaic_dialog()


if __name__ == "__main__":
    unittest.main()
