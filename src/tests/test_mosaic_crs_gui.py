"""End-to-end CRS-resolution tests through the mosaic pane (EPIC #629, issue #635).

These drive the real ingestion path (``MosaicPane._on_scene_ingested`` ->
``_ensure_common_grid``) and the manual "Choose Target CRS" action, asserting the
*actual* behavior of the wired-up controller + dialog.

Behavioral note (verified, intentional for these tests): ingestion resolves the
common grid after **every** scene, so the first scene auto-resolves and *persists*
its CRS as the target (``build_common_grid`` adopts the shared/only scene CRS). A
later scene with a different but mappable CRS therefore reprojects onto that locked
target **without** popping the reproject dialog. The dialog is instead reached via
the manual "Choose Target CRS" button, which is what the dialog cases below cover.
"""
import os
import tempfile
import unittest
from unittest import mock

from osgeo import gdal, osr
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

import numpy as np

import tests.context  # noqa: F401  (adds src/ to sys.path)
from test_utils.test_model import WiserTestModel

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.integration,
]

# Per-EPSG georeferencing so the two scenes cover roughly the same ground (UTM 11N
# and its geographic equivalent), keeping cross-CRS reprojection well-defined.
_GEO_BY_EPSG = {
    32611: (400000.0, 30.0, 0.0, 3800000.0, 0.0, -30.0),
    4326: (-117.0, 0.001, 0.0, 34.0, 0.0, -0.001),
}


def _write_georeffed_tiff(path, epsg=32611, nodata=-9999, bands=3, collar=2, size=20):
    """Write a small tiled, georeferenced GeoTIFF (given EPSG) with a nodata collar."""
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
    ds.SetGeoTransform(list(_GEO_BY_EPSG[epsg]))
    ds.SetProjection(srs.ExportToWkt())
    for b in range(1, bands + 1):
        arr = np.full((size, size), 7, dtype=np.float32)
        if collar > 0:
            arr[:collar, :] = nodata
            arr[-collar:, :] = nodata
            arr[:, :collar] = nodata
            arr[:, -collar:] = nodata
        band = ds.GetRasterBand(b)
        band.WriteArray(arr)
        band.SetNoDataValue(float(nodata))
    ds.FlushCache()
    ds = None


def _same_as_epsg(wkt, epsg):
    got = osr.SpatialReference()
    got.ImportFromWkt(wkt)
    want = osr.SpatialReference()
    want.ImportFromEPSG(epsg)
    return bool(got.IsSame(want))


class TestMosaicCrsGui(unittest.TestCase):
    """Drive scene ingestion + target-CRS resolution through a real MosaicPane."""

    def setUp(self):
        self.test_model = WiserTestModel()
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # -- helpers --------------------------------------------------------------

    def _wait_for(self, predicate, timeout_ms=15000, step_ms=50):
        waited = 0
        while waited < timeout_ms:
            if predicate():
                return True
            QTest.qWait(step_ms)
            waited += step_ms
        return predicate()

    def _select_dataset(self, pane, ds_id):
        index = pane._dataset_combo.findData(ds_id)
        self.assertGreaterEqual(index, 0, "dataset not present in the Add-Scene combo")
        pane._dataset_combo.setCurrentIndex(index)

    def _load(self, name, epsg):
        path = os.path.join(self._tmp.name, name)
        _write_georeffed_tiff(path, epsg=epsg, bands=3, collar=2, size=20)
        return self.test_model.load_dataset(path)

    def _ingest(self, pane, controller, dataset, expected_count):
        """Add one scene and pump the loop until it lands in the controller."""
        self._select_dataset(pane, dataset.get_id())
        pane._add_scene_button.click()
        added = self._wait_for(lambda: controller.scene_count() == expected_count)
        self.assertTrue(added, "scene was not ingested within the timeout")

    # -- ingestion: differing CRS auto-locks to the first scene, no dialog ----

    def test_differing_crs_ingest_autolocks_without_dialog(self):
        ds_a = self._load("a_utm.tif", 32611)
        ds_b = self._load("b_geo.tif", 4326)

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog") as Dlg:
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)
            # No prompt: the first scene locked the target to 32611, and the second
            # (4326) maps onto it, so build_common_grid never raises TargetCrsRequired.
            Dlg.assert_not_called()

        self.assertTrue(_same_as_epsg(controller.get_target_crs(), 32611))
        self.assertIsNotNone(controller.get_common_grid().geotransform)

        self.test_model.close_seamless_mosaic_dialog()

    # -- ingestion: same CRS auto-resolves to the shared CRS, no dialog -------

    def test_same_crs_ingest_resolves_without_dialog(self):
        ds_a = self._load("a_utm.tif", 32611)
        ds_b = self._load("b_utm.tif", 32611)

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog") as Dlg:
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)
            Dlg.assert_not_called()

        self.assertTrue(_same_as_epsg(controller.get_target_crs(), 32611))

        self.test_model.close_seamless_mosaic_dialog()

    # -- manual "Choose Target CRS": accept overrides the locked target -------

    def test_manual_choose_target_crs_accept(self):
        ds_a = self._load("a_utm.tif", 32611)
        ds_b = self._load("b_geo.tif", 4326)

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)

        target_geo = osr.SpatialReference()
        target_geo.ImportFromEPSG(4326)
        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog") as Dlg:
            inst = Dlg.return_value
            inst.exec.return_value = QDialog.Accepted
            inst.selected_target_wkt.return_value = target_geo.ExportToWkt()
            pane._on_choose_target_crs()
            Dlg.assert_called_once()

        # The user's choice (4326) replaces the auto-locked 32611 target.
        self.assertTrue(_same_as_epsg(controller.get_target_crs(), 4326))

        self.test_model.close_seamless_mosaic_dialog()

    # -- manual "Choose Target CRS": cancel leaves the target unchanged -------

    def test_manual_choose_target_crs_cancel(self):
        ds_a = self._load("a_utm.tif", 32611)
        ds_b = self._load("b_geo.tif", 4326)

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog") as Dlg:
            Dlg.return_value.exec.return_value = QDialog.Rejected
            pane._on_choose_target_crs()

        # Cancelled: the target stays at the auto-locked 32611.
        self.assertTrue(_same_as_epsg(controller.get_target_crs(), 32611))

        self.test_model.close_seamless_mosaic_dialog()

    # -- manual "Choose Target CRS": incompatible choice commits + marks pending --

    def test_manual_choose_target_crs_incompatible_marks_pending(self):
        ds_a = self._load("a_utm.tif", 32611)
        ds_b = self._load("b_geo.tif", 4326)

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)

        target_geo = osr.SpatialReference()
        target_geo.ImportFromEPSG(4326)
        # Force every scene to be un-transformable to the chosen target so the change
        # exercises the "no live scenes" path. The new behavior *accepts* the CRS and
        # marks the scenes pending instead of rejecting the choice.
        # Pending status is computed live, so the incompatibility patch must stay active
        # while we assert on it (it reverts the moment the patch exits).
        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog") as Dlg, mock.patch(
            "wiser.gui.mosaic_pane.QMessageBox.warning"
        ) as warn, mock.patch(
            "wiser.raster.mosaic_controller.can_transform_between_srs",
            return_value=False,
        ):
            inst = Dlg.return_value
            inst.exec.return_value = QDialog.Accepted
            inst.selected_target_wkt.return_value = target_geo.ExportToWkt()
            pane._on_choose_target_crs()

            # The choice is committed (not rejected), every scene is now pending, and the
            # user is warned that the preview has nothing left to show.
            self.assertTrue(_same_as_epsg(controller.get_target_crs(), 4326))
            self.assertFalse(controller.has_live_scenes())
            self.assertTrue(controller.has_pending_scenes())
            warn.assert_called_once()

        self.test_model.close_seamless_mosaic_dialog()

    # -- CRS change reframes the camera when the mosaic lands off-screen -------

    def test_choose_target_crs_reframes_when_mosaic_off_screen(self):
        # Two UTM (32611) scenes -> the camera is framed on eastings ~4e5. Switching the
        # target to geographic (4326) reprojects the footprints to ~(-117, 34), far from
        # the parked camera, so the mosaic lands off-screen and must be reframed.
        ds_a = self._load("a_utm.tif", 32611)
        ds_b = self._load("b_utm.tif", 32611)

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()
        view = pane._mosaic_view

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)

        # Frame the camera on the UTM mosaic, as the user would be viewing it.
        view.resize(300, 300)
        view.zoom_to_extent(controller.get_common_grid().extent)
        self.assertGreater(view._transform.center_x, 100000.0)  # parked over UTM easting

        target_geo = osr.SpatialReference()
        target_geo.ImportFromEPSG(4326)
        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog") as Dlg:
            inst = Dlg.return_value
            inst.exec.return_value = QDialog.Accepted
            inst.selected_target_wkt.return_value = target_geo.ExportToWkt()
            pane._on_choose_target_crs()

        self.assertTrue(_same_as_epsg(controller.get_target_crs(), 4326))
        # The camera reframed to the new (geographic) union extent center.
        geo_extent = controller.get_common_grid().extent
        self.assertAlmostEqual(view._transform.center_x, (geo_extent[0] + geo_extent[2]) / 2.0, places=3)
        self.assertAlmostEqual(view._transform.center_y, (geo_extent[1] + geo_extent[3]) / 2.0, places=3)

        self.test_model.close_seamless_mosaic_dialog()

    def test_ensure_scenes_in_view_is_noop_when_a_scene_is_visible(self):
        # No CRS change: the scene is already framed, so ensure_scenes_in_view must leave
        # the user's pan/zoom untouched (only an off-screen mosaic gets reframed).
        ds_a = self._load("a_utm.tif", 32611)

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()
        view = pane._mosaic_view

        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)

        view.resize(300, 300)
        view.zoom_to_extent(controller.get_common_grid().extent)
        before = (
            view._transform.center_x,
            view._transform.center_y,
            view._transform.world_units_per_pixel,
        )

        view.ensure_scenes_in_view()

        after = (
            view._transform.center_x,
            view._transform.center_y,
            view._transform.world_units_per_pixel,
        )
        self.assertEqual(before, after)

        self.test_model.close_seamless_mosaic_dialog()


if __name__ == "__main__":
    unittest.main()
