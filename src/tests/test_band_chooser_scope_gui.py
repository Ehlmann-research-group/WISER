"""Integration tests for the band chooser's "apply to every pane" option.

Display bands are held per dataset in each raster pane, so every panel of one
pane that shows a dataset draws it with the same bands. The dialog's checkbox
chooses between the originating pane only and every pane (Context, main window
and Zoom) showing the dataset.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import tests.context  # noqa: F401

import pytest
from PySide6.QtWidgets import QDialog

from test_utils.test_model import WiserTestModel
from wiser.gui import rasterpane as rasterpane_module
from wiser.gui.band_chooser import BandChooserDialog
from wiser.gui.geo_reference_pane import GeoReferencerPane
from wiser.gui.geo_reference_task_delegate import PointSelectorType

pytestmark = [pytest.mark.integration]

_DATASETS = Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets"
DATASET = str(_DATASETS / "jpl_15_7_7.hdr")

NEW_BANDS = (10, 5, 2)


def _scripted_dialog(apply_everywhere: bool, opened: list):
    """
    A BandChooserDialog whose ``exec_`` records the checkbox's initial
    ``(enabled, checked)`` state in ``opened``, then picks ``NEW_BANDS`` and
    accepts.
    """

    class ScriptedDialog(BandChooserDialog):
        def exec_(self):
            chk = self._ui.chk_apply_globally
            opened.append((chk.isEnabled(), chk.isChecked()))
            self._ui.cbox_red_band.setCurrentIndex(NEW_BANDS[0])
            self._ui.cbox_green_band.setCurrentIndex(NEW_BANDS[1])
            self._ui.cbox_blue_band.setCurrentIndex(NEW_BANDS[2])
            self._ui.chk_apply_globally.setChecked(apply_everywhere)
            return QDialog.Accepted

    return ScriptedDialog


class TestBandChooserScope(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        self.dataset = self.test_model.load_dataset(DATASET)
        self.ds_id = self.dataset.get_id()
        self.test_model.set_main_view_layout((1, 2))
        self.test_model.set_main_view_rv((0, 0), self.ds_id)
        self.test_model.set_main_view_rv((0, 1), self.ds_id)
        self.test_model.run()
        self.original = tuple(self.test_model.main_view.get_rasterview((0, 0)).get_display_bands())
        self.assertNotEqual(self.original, NEW_BANDS)

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _choose_from_tile(self, pos, apply_everywhere: bool):
        opened = []
        with patch.object(rasterpane_module, "BandChooserDialog", _scripted_dialog(apply_everywhere, opened)):
            self.test_model.main_view._on_band_chooser(rasterview_pos=pos)
        self.test_model.run()
        self.assertEqual(opened, [(True, True)])

    def _bands(self, pane, pos=(0, 0)):
        return tuple(pane.get_rasterview(pos).get_display_bands())

    def test_option_defaults_to_every_pane(self):
        dialog = BandChooserDialog(
            self.test_model.app_state, self.dataset, list(self.original), parent=self.test_model.main_view
        )
        chk = dialog._ui.chk_apply_globally
        self.assertTrue(chk.isEnabled())
        self.assertTrue(chk.isChecked())
        self.assertIn("every pane", chk.text().replace("&", ""))
        self.assertIn("same bands", chk.toolTip())

    def test_checked_updates_every_pane_and_panel(self):
        self._choose_from_tile((0, 1), apply_everywhere=True)

        main = self.test_model.main_view
        self.assertEqual(self._bands(main, (0, 0)), NEW_BANDS)
        self.assertEqual(self._bands(main, (0, 1)), NEW_BANDS)
        self.assertEqual(self._bands(self.test_model.context_pane), NEW_BANDS)
        self.assertEqual(self._bands(self.test_model.zoom_pane), NEW_BANDS)

    def test_unchecked_updates_every_panel_of_the_pane_only(self):
        self._choose_from_tile((0, 1), apply_everywhere=False)

        main = self.test_model.main_view
        self.assertEqual(self._bands(main, (0, 0)), NEW_BANDS)
        self.assertEqual(self._bands(main, (0, 1)), NEW_BANDS)
        self.assertEqual(self._bands(self.test_model.context_pane), self.original)
        self.assertEqual(self._bands(self.test_model.zoom_pane), self.original)

    def test_georeferencer_pane_disables_the_option(self):
        pane = GeoReferencerPane(self.test_model.app_state, PointSelectorType.TARGET_POINT_SELECTOR)
        pane.show_dataset(self.dataset)
        self.test_model.run()

        opened = []
        with patch.object(rasterpane_module, "BandChooserDialog", _scripted_dialog(False, opened)):
            pane._on_band_chooser()
        self.test_model.run()

        self.assertEqual(opened, [(False, False)])
        self.assertEqual(self._bands(pane), NEW_BANDS)
        self.assertEqual(self._bands(self.test_model.main_view), self.original)
        pane.deleteLater()


if __name__ == "__main__":
    unittest.main()
