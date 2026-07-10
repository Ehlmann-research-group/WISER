"""GUI tests for re-georeferencing a mosaic scene in place (EPIC #629, issue #685).

Drive the real :class:`MosaicPane` right-click "Georeference…" entry point end to end
through the real ingestion path (as the #634/#638 GUI tests do). Chunk 1 covers the
entry point: the context menu offers the action, and it opens a task-scoped
:class:`GeoReferencerDialog` locked onto the chosen scene (target + save path locked)
without disturbing the controller. The live reingest/swap and save/revert semantics are
exercised in later chunks.
"""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from osgeo import gdal, osr
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest

import tests.context  # noqa: F401  (adds src/ to sys.path)
from test_utils.test_model import WiserTestModel
from wiser.gui.geo_reference_config import GeoReferencerConfig
from wiser.utils.primitives import temp_dir

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


class TestMosaicRegeoreferenceEntryPoint(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        # ignore_cleanup_errors: a loaded RasterDataSet keeps its GeoTIFF open (a GDAL
        # handle), and Windows refuses to unlink an open file.
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

    def test_scene_list_has_context_menu_policy(self):
        _dlg, pane, _controller = self._open_with_two_scenes()
        self.assertEqual(pane._scene_list.contextMenuPolicy(), Qt.CustomContextMenu)
        self.test_model.close_seamless_mosaic_dialog()

    def test_context_menu_offers_georeference(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        # Row 0 is the top scene; it carries its real controller index in UserRole.
        item = pane._scene_list.item(0)
        index = item.data(Qt.UserRole)

        # Patch QMenu so exec_ does not block, and _on_georeference_scene so we can
        # confirm the action targets the right controller index.
        with mock.patch("wiser.gui.mosaic_pane.QMenu") as MenuCls, mock.patch.object(
            pane._scene_list, "itemAt", return_value=item
        ), mock.patch.object(pane, "_on_georeference_scene") as geo:
            menu = MenuCls.return_value
            action = menu.addAction.return_value
            pane._on_scene_context_menu(QPoint(0, 0))

            menu.addAction.assert_called_once()
            (label,), _ = menu.addAction.call_args
            self.assertIn("Georeference", label)
            menu.exec_.assert_called_once()

            # Firing the action re-georeferences the clicked row's scene.
            (slot,), _ = action.triggered.connect.call_args
            slot()
            geo.assert_called_once_with(index)
        self.test_model.close_seamless_mosaic_dialog()

    def test_context_menu_no_op_without_item(self):
        _dlg, pane, _controller = self._open_with_two_scenes()
        with mock.patch("wiser.gui.mosaic_pane.QMenu") as MenuCls, mock.patch.object(
            pane._scene_list, "itemAt", return_value=None
        ):
            pane._on_scene_context_menu(QPoint(0, 0))
            MenuCls.assert_not_called()
        self.test_model.close_seamless_mosaic_dialog()

    def test_georeference_opens_locked_dialog(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        scenes_before = controller.get_scenes()
        index = 0  # georeference the bottom scene
        scene = scenes_before[index]

        with mock.patch("wiser.gui.mosaic_pane.GeoReferencerDialog") as DlgCls:
            pane._on_georeference_scene(index)

        # A fresh dialog was constructed with the shared app state / services.
        DlgCls.assert_called_once()
        args, kwargs = DlgCls.call_args
        self.assertIs(args[0], pane._app_state)
        self.assertIs(args[1], pane._app_services)
        self.assertIn("parent", kwargs)

        inst = DlgCls.return_value
        inst.show.assert_called_once()
        (config,), _ = inst.show.call_args
        self.assertIsInstance(config, GeoReferencerConfig)

        # The target is locked to this exact (original) dataset; save path is locked to
        # a mosaic-owned temp GeoTIFF; the reference stays user-chosen.
        self.assertIs(config.target_dataset, scene.dataset)
        self.assertFalse(config.allow_change_target)
        self.assertFalse(config.allow_change_save_path)
        self.assertIsNone(config.reference_dataset)
        self.assertEqual(config.accept_button_text, "Save to Mosaic")
        self.assertTrue(config.save_path.endswith(".tif"))
        self.assertTrue(
            os.path.abspath(config.save_path).startswith(os.path.abspath(str(temp_dir()))),
            "warp output should live under the session temp dir",
        )

        # The live result/teardown are wired to the pane's handlers.
        inst.warp_completed.connect.assert_called_once_with(pane._on_scene_rewarped)
        inst.finished.connect.assert_called_once_with(pane._on_geodialog_finished)

        # The in-flight context holds the original scene aside at its z-order slot.
        ctx = pane._regeoref_ctx
        self.assertIsNotNone(ctx)
        self.assertIs(ctx.orig_scene, scene)
        self.assertEqual(ctx.orig_index, index)
        self.assertIsNone(ctx.warped_scene)

        # Nothing about the mosaic changed just by opening the dialog.
        self.assertEqual(controller.scene_count(), 2)
        scenes_after = controller.get_scenes()
        self.assertIs(scenes_after[0], scenes_before[0])
        self.assertIs(scenes_after[1], scenes_before[1])
        self.test_model.close_seamless_mosaic_dialog()

    def test_finished_clears_context_and_deletes_dialog(self):
        _dlg, pane, _controller = self._open_with_two_scenes()
        with mock.patch("wiser.gui.mosaic_pane.GeoReferencerDialog") as DlgCls:
            pane._on_georeference_scene(0)
            inst = DlgCls.return_value

        pane._on_geodialog_finished(0)
        self.assertIsNone(pane._regeoref_ctx)
        inst.deleteLater.assert_called_once()
        self.test_model.close_seamless_mosaic_dialog()

    # -- reingest + swap on Run Warp (Chunk 2) --------------------------------

    def _open_georef(self, pane, index):
        """Open the real (locked) georeference dialog on the scene at ``index``."""
        pane._on_georeference_scene(index)
        ctx = pane._regeoref_ctx
        self.assertIsNotNone(ctx)
        return ctx

    def _rewarp_and_wait(self, pane, ctx, name, origin=(400000.0, 3800000.0)):
        """Feed a real georeferenced GeoTIFF as a completed warp and wait for the swap."""
        warp_path = os.path.join(self._tmp.name, name)
        _write_tiff(warp_path, origin=origin)
        prev = ctx.warped_scene
        # _ensure_common_grid runs in the swap; same-CRS scenes never prompt, but patch
        # the reproject dialog defensively so a stray prompt can't block the test.
        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            pane._on_scene_rewarped(warp_path)
            self.assertTrue(
                self._wait_for(lambda: ctx.warped_scene is not None and ctx.warped_scene is not prev),
                "warped scene was not swapped in within the timeout",
            )
        return ctx.warped_scene

    def test_warp_completed_swaps_scene_in_place(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        scenes_before = controller.get_scenes()  # bottom-to-top [A, B]
        index = 0  # georeference the bottom scene, A
        orig = scenes_before[index]

        ctx = self._open_georef(pane, index)

        warp_path = os.path.join(self._tmp.name, "warp1.tif")
        _write_tiff(warp_path, origin=(400000.0, 3800000.0))
        with mock.patch("wiser.gui.mosaic_pane.ReprojectPromptDialog"):
            pane._on_scene_rewarped(warp_path)
            # Reingest blocks the georeference dialog, not the mosaic window.
            self.assertFalse(ctx.dialog.isEnabled())
            self.assertTrue(_dlg.isEnabled())
            self.assertTrue(
                self._wait_for(lambda: ctx.warped_scene is not None),
                "warped scene was not swapped in within the timeout",
            )

        scenes_after = controller.get_scenes()
        self.assertEqual(controller.scene_count(), 2)  # swapped in place, not added
        # The warped scene occupies the original z-order slot; the other is unchanged.
        self.assertIs(scenes_after[index], ctx.warped_scene)
        self.assertIsNot(scenes_after[index], orig)
        self.assertIs(scenes_after[1], scenes_before[1])
        # The reingest produced fully-derived state (overviews + footprint).
        self.assertTrue(ctx.warped_scene.has_overviews)
        self.assertIsNotNone(ctx.warped_scene.footprint_wkt)
        # The original scene / dataset is held aside untouched.
        self.assertIs(ctx.orig_scene, orig)
        self.assertIs(ctx.orig_scene.dataset, orig.dataset)
        self.assertFalse(any(s is orig for s in scenes_after))

        ctx.dialog.reject()
        self.test_model.close_seamless_mosaic_dialog()

    def test_repeated_warp_replaces_not_stacks(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        index = 0
        ctx = self._open_georef(pane, index)

        first = self._rewarp_and_wait(pane, ctx, "warp_a.tif")
        self.assertEqual(controller.scene_count(), 2)

        second = self._rewarp_and_wait(pane, ctx, "warp_b.tif")
        self.assertIsNot(second, first)
        # Replaced, not stacked: still two scenes, and the first warped one is gone.
        self.assertEqual(controller.scene_count(), 2)
        scenes_after = controller.get_scenes()
        self.assertFalse(any(s is first for s in scenes_after))
        self.assertIs(scenes_after[index], second)

        ctx.dialog.reject()
        self.test_model.close_seamless_mosaic_dialog()

    def test_rewarp_does_not_register_output_dataset(self):
        _dlg, pane, controller = self._open_with_two_scenes()
        before = len(pane._app_state.get_datasets())
        ctx = self._open_georef(pane, 0)
        self._rewarp_and_wait(pane, ctx, "warp_noreg.tif")
        # The warped output is mosaic-internal: it must not enter the global dataset
        # list (and thus never appears in the Add-Scene combo).
        self.assertEqual(len(pane._app_state.get_datasets()), before)

        ctx.dialog.reject()
        self.test_model.close_seamless_mosaic_dialog()


if __name__ == "__main__":
    unittest.main()
