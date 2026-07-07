"""GUI integration test for the mosaic Export path (EPIC #629, issue #639).

Drives the real :class:`MosaicPane` "Export / Finish…" button end to end through the
ingestion + export pipeline: ingest two overlapping same-CRS scenes, pick an output
path (patched file dialog), click Export, and assert the ENVI mosaic is written on the
common grid while nothing is loaded back into WISER (the user opens the file manually).
"""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pytest
from osgeo import gdal, osr
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog, QMessageBox

import tests.context  # noqa: F401  (adds src/ to sys.path)
from test_utils.test_model import WiserTestModel

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.integration,
]


def _write_tiff(path, origin, epsg=32611, pixel=30.0, size=20, collar=2, bands=3, base=7.0):
    """Write a small georeferenced GeoTIFF at ``origin`` with a nodata collar."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds = gdal.GetDriverByName("GTiff").Create(path, size, size, bands, gdal.GDT_Float32)
    ox, oy = origin
    ds.SetGeoTransform([ox, pixel, 0.0, oy, 0.0, -pixel])
    ds.SetProjection(srs.ExportToWkt())
    for b in range(1, bands + 1):
        arr = np.full((size, size), base + b, dtype=np.float32)
        arr[:collar, :] = -9999
        arr[-collar:, :] = -9999
        arr[:, :collar] = -9999
        arr[:, -collar:] = -9999
        band = ds.GetRasterBand(b)
        band.WriteArray(arr)
        band.SetNoDataValue(-9999.0)
    ds.FlushCache()
    ds = None


class TestMosaicExportGui(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _wait_for(self, predicate, timeout_ms=30000, step_ms=50):
        waited = 0
        while waited < timeout_ms:
            if predicate():
                return True
            QTest.qWait(step_ms)
            waited += step_ms
        return predicate()

    def _load(self, name, origin, base=7.0):
        path = os.path.join(self._tmp.name, name)
        _write_tiff(path, origin, base=base)
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
        """Open the dialog and ingest two same-CRS overlapping scenes: bottom-to-top [a, b]."""
        ds_a = self._load("a.tif", origin=(400000.0, 3800000.0), base=7.0)
        ds_b = self._load("b.tif", origin=(400300.0, 3799700.0), base=100.0)
        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()
        controller = pane.get_controller()
        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            self._ingest(pane, controller, ds_a, 1)
            self._ingest(pane, controller, ds_b, 2)
        return dlg, pane, controller

    def test_export_writes_envi_and_does_not_load_back(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        grid = controller.get_common_grid()
        self.assertIsNotNone(grid.extent)

        datasets_before = self.test_model.app_state.num_datasets()
        out_path = os.path.join(self._tmp.name, "mosaic_out.img")

        info_seen = {"count": 0}

        def _fake_info(*_args, **_kwargs):
            info_seen["count"] += 1
            return QMessageBox.Ok

        with mock.patch.object(
            QFileDialog, "getSaveFileName", return_value=(out_path, "")
        ), mock.patch.object(QMessageBox, "information", side_effect=_fake_info), mock.patch.object(
            QMessageBox, "warning"
        ) as warn:
            pane._export_button.click()
            self.assertTrue(
                self._wait_for(lambda: os.path.exists(out_path) and info_seen["count"] > 0),
                "export did not finish within the timeout",
            )

        # No warning dialog (a warning would mean the export failed / was blocked).
        warn.assert_not_called()
        # The ENVI image + header were written.
        self.assertTrue(os.path.exists(out_path))
        hdr = out_path[:-4] + ".hdr"
        self.assertTrue(os.path.exists(hdr) or os.path.exists(out_path + ".hdr"))
        # A success dialog was shown.
        self.assertEqual(info_seen["count"], 1)

        # Nothing was loaded back into WISER — the user opens the file manually.
        self.assertEqual(self.test_model.app_state.num_datasets(), datasets_before)

        # The mosaic is on the common grid with nodata carried through.
        exported = gdal.OpenEx(out_path, gdal.OF_RASTER, allowed_drivers=["ENVI"])
        try:
            self.assertEqual(exported.RasterXSize, grid.width)
            self.assertEqual(exported.RasterYSize, grid.height)
            self.assertEqual(exported.RasterCount, controller.get_scenes()[0].dataset.num_bands())
            self.assertEqual(exported.GetRasterBand(1).GetNoDataValue(), -9999.0)
        finally:
            exported = None

        self.test_model.close_seamless_mosaic_dialog()

    def test_export_with_no_scenes_informs_and_writes_nothing(self):
        dlg = self.test_model.open_seamless_mosaic_dialog()
        pane = dlg.get_mosaic_pane()

        with mock.patch.object(QMessageBox, "information") as info, mock.patch.object(
            QFileDialog, "getSaveFileName"
        ) as save_dialog:
            pane._export_button.click()
            QTest.qWait(100)

        # It should inform "nothing to export" and never reach the file picker.
        info.assert_called_once()
        save_dialog.assert_not_called()
        self.test_model.close_seamless_mosaic_dialog()


if __name__ == "__main__":
    unittest.main()
