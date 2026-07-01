import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from osgeo import gdal, osr
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

import tests.context  # noqa: F401  (adds src/ to sys.path)
from test_utils.test_model import WiserTestModel
from wiser.gui.mosaic_dialog import SeamlessMosaicDialog

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.integration,
]

_GOOD_GEO_TRANSFORM = (100.0, 1.0, 0.0, 200.0, 0.0, -1.0)


def _write_georeffed_tiff(path, nodata=-9999, bands=3, collar=2, size=20):
    """Write a small tiled, georeferenced GeoTIFF with a nodata collar."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32611)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        path,
        size,
        size,
        bands,
        gdal.GDT_Float32,
        options=["TILED=YES", "BLOCKXSIZE=16", "BLOCKYSIZE=16"],
    )
    ds.SetGeoTransform(list(_GOOD_GEO_TRANSFORM))
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


class TestSeamlessMosaicDialogSmoke(unittest.TestCase):
    """
    GUI smoke test for the Seamless Mosaic scaffolding (issue #633): the menu action
    opens the dialog and it closes again without error, with the dialog -> pane -> view
    widget tree wired up.
    """

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_dialog_opens_and_closes(self):
        dlg = self.test_model.open_seamless_mosaic_dialog()

        self.assertIsNotNone(dlg)
        self.assertIsInstance(dlg, SeamlessMosaicDialog)

        # The full scaffolding tree is present: dialog -> pane -> view, sharing one
        # controller between the pane and its view.
        pane = dlg.get_mosaic_pane()
        self.assertIsNotNone(pane)
        view = pane.get_mosaic_view()
        self.assertIsNotNone(view)
        self.assertIs(pane.get_controller(), view.get_controller())

        # Closing must not raise.
        self.test_model.close_seamless_mosaic_dialog()


class TestSeamlessMosaicAddScene(unittest.TestCase):
    """
    Integration test for the Add-Scene ingestion path (issue #634): a georeferenced
    dataset is picked and ingested on a background thread into the controller, while
    an ungeoreferenced one is rejected before any scene is added.
    """

    def setUp(self):
        self.test_model = WiserTestModel()
        # ignore_cleanup_errors: the loaded RasterDataSet keeps the source GeoTIFF
        # open (a GDAL handle), and Windows refuses to unlink an open file, so a
        # strict cleanup would raise a spurious PermissionError at teardown.
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _wait_for(self, predicate, timeout_ms=15000, step_ms=50):
        """Pump the Qt event loop until ``predicate()`` is true or the timeout hits."""
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

    def test_add_valid_scene(self):
        tiff_path = os.path.join(self._tmp.name, "scene.tif")
        _write_georeffed_tiff(tiff_path, bands=3, collar=2, size=20)
        dataset = self.test_model.load_dataset(tiff_path)

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()

        self._select_dataset(pane, dataset.get_id())
        pane._add_scene_button.click()

        added = self._wait_for(lambda: controller.scene_count() == 1)
        self.assertTrue(added, "scene was not ingested within the timeout")

        scene = controller.get_scenes()[0]
        self.assertTrue(scene.has_overviews)
        self.assertIsNotNone(scene.footprint_wkt)
        self.assertIsNotNone(scene.gdal_path)

        self.test_model.close_seamless_mosaic_dialog()

    def test_add_invalid_scene_rejected(self):
        # A numpy-array dataset is ungeoreferenced (identity transform, no SRS), so
        # validation must reject it before any background work runs.
        ungeoreffed = self.test_model.load_dataset(np.zeros((3, 10, 10), dtype=np.float32))

        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()

        self._select_dataset(pane, ungeoreffed.get_id())
        # The rejection surfaces a modal QMessageBox.warning; patch it so the test
        # does not block, and assert it fired.
        with mock.patch("wiser.gui.mosaic_pane.QMessageBox.warning") as warn:
            pane._add_scene_button.click()
            QTest.qWait(100)

        warn.assert_called_once()
        self.assertEqual(controller.scene_count(), 0)

        self.test_model.close_seamless_mosaic_dialog()


if __name__ == "__main__":
    unittest.main()
