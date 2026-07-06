"""GUI tests for the mosaic reproject prompt dialog (EPIC #629, issue #635).

The :class:`ReprojectPromptDialog` is data-driven: it takes the scene->CRS summary
and the distinct-CRS choices as plain lists (not the controller), so these tests
construct it directly. A :class:`WiserTestModel` is used only to provide a running
(offscreen) QApplication and a real :class:`ApplicationState` for the user-created
CRS lookup.
"""
import unittest
from unittest import mock

from osgeo import osr

import tests.context  # noqa: F401  (adds src/ to sys.path)
from test_utils.test_model import WiserTestModel
from wiser.gui.geo_reference_dialog import COMMON_SRS
from wiser.gui.mosaic_crs_dialog import ReprojectPromptDialog

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.integration,
]


def _wkt_for_epsg(epsg: int) -> str:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs.ExportToWkt()


def _same_as_epsg(wkt: str, epsg: int) -> bool:
    got = osr.SpatialReference()
    got.ImportFromWkt(wkt)
    want = osr.SpatialReference()
    want.ImportFromEPSG(epsg)
    return bool(got.IsSame(want))


class TestReprojectPromptDialog(unittest.TestCase):
    """Exercise the reproject prompt dialog's target chooser and lookups."""

    def setUp(self):
        self.test_model = WiserTestModel()
        # bottom scene = 32611, top scene = 4326. scene_crs_choices is ordered so the
        # last entry (the top scene's CRS) becomes the default target selection.
        self.wkt_utm = _wkt_for_epsg(32611)
        self.wkt_geo = _wkt_for_epsg(4326)
        self.summary = [
            ("bottom", "WGS 84 / UTM zone 11N (EPSG:32611)"),
            ("top", "WGS 84 (EPSG:4326)"),
        ]
        self.choices = [
            ("WGS 84 / UTM zone 11N (EPSG:32611)", self.wkt_utm),
            ("WGS 84 (EPSG:4326)", self.wkt_geo),
        ]

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _make_dialog(self):
        return ReprojectPromptDialog(self.summary, self.choices, self.test_model.app_state)

    def test_table_combo_and_default_selection(self):
        dlg = self._make_dialog()

        # One table row per scene.
        self.assertEqual(dlg._table.rowCount(), 2)

        # Target combo: the two distinct scene CRSs + the common CRSs.
        self.assertEqual(dlg._cbox_target.count(), len(self.choices) + len(COMMON_SRS))

        # Default selection is the top scene's CRS (4326, the last choice).
        self.assertEqual(dlg._cbox_target.currentIndex(), len(self.choices) - 1)
        self.assertTrue(_same_as_epsg(dlg.selected_target_wkt(), 4326))

    def test_user_created_crs_appears_as_option(self):
        user_srs = osr.SpatialReference()
        user_srs.ImportFromEPSG(26915)
        self.test_model.app_state._user_created_crs = {"MyUserCRS": (user_srs, None)}

        dlg = self._make_dialog()
        self.assertEqual(dlg._cbox_target.count(), len(self.choices) + len(COMMON_SRS) + 1)
        self.assertGreaterEqual(dlg._cbox_target.findText("MyUserCRS"), 0)

    def test_add_authority_crs_valid(self):
        dlg = self._make_dialog()
        before = dlg._cbox_target.count()

        idx = dlg._cbox_authority.findText("EPSG")
        self.assertGreaterEqual(idx, 0)
        dlg._cbox_authority.setCurrentIndex(idx)
        dlg._ledit_code.setText("3857")
        dlg._on_add_authority_crs()

        self.assertEqual(dlg._cbox_target.count(), before + 1)
        # The new item is auto-selected and resolves to Web Mercator.
        self.assertEqual(dlg._cbox_target.currentIndex(), dlg._cbox_target.count() - 1)
        self.assertTrue(_same_as_epsg(dlg.selected_target_wkt(), 3857))

    def test_add_authority_crs_bad_code_warns(self):
        dlg = self._make_dialog()
        before = dlg._cbox_target.count()

        idx = dlg._cbox_authority.findText("EPSG")
        dlg._cbox_authority.setCurrentIndex(idx)
        dlg._ledit_code.setText("999999999")

        with mock.patch("wiser.gui.mosaic_crs_dialog.QMessageBox.warning") as warn:
            dlg._on_add_authority_crs()

        warn.assert_called_once()
        self.assertEqual(dlg._cbox_target.count(), before)

    def test_accept_with_nothing_selectable_warns_and_stays_open(self):
        dlg = self._make_dialog()
        # Force the combo empty so there is nothing to select.
        dlg._cbox_target.clear()
        self.assertIsNone(dlg.selected_target_wkt())

        with mock.patch("wiser.gui.mosaic_crs_dialog.QMessageBox.warning") as warn:
            dlg.accept()

        warn.assert_called_once()
        # accept() bailed before super().accept(), so the dialog was not accepted.
        self.assertNotEqual(dlg.result(), ReprojectPromptDialog.Accepted)


if __name__ == "__main__":
    unittest.main()
