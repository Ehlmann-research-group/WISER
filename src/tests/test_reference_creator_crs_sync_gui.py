"""GUI sync tests: Reference System Creator CRS combo <-> ApplicationState (issue #689).

The Reference System Creator's starting-CRS combo box (``cbox_user_crs``) is populated
from ApplicationState's user-created-CRS store.  These tests drive the real dialog
against a real ApplicationState (offscreen ``WiserTestModel``) and assert the combo
reflects the store: a dialog constructed after the store is populated shows the stored
CRS, and refreshing an open dialog after the store gains an entry surfaces it.  The
store emits no change signal, so the dialog re-reads it on construction and on the
explicit refresh the creation flow already calls.
"""

import unittest

import tests.context  # noqa: F401

from osgeo import osr
from PySide6.QtCore import Qt

from test_utils.test_model import WiserTestModel
from wiser.gui.reference_creator_dialog import ReferenceCreatorDialog


def _make_srs(epsg):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs


class TestReferenceCreatorCrsSync(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        self.app_state = self.test_model.app_state

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _make_dialog(self):
        dialog = ReferenceCreatorDialog(self.app_state, parent=self.test_model.main_window)
        dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
        self.addCleanup(dialog.close)
        return dialog

    def test_dialog_seeds_crs_combo_from_app_state(self):
        self.app_state.add_user_created_crs("MyTestCRS", _make_srs(26915), None)
        dialog = self._make_dialog()
        self.assertGreaterEqual(dialog._ui.cbox_user_crs.findText("MyTestCRS"), 0)

    def test_crs_combo_refreshes_after_store_change(self):
        dialog = self._make_dialog()
        self.assertEqual(dialog._ui.cbox_user_crs.findText("LaterCRS"), -1)

        self.app_state.add_user_created_crs("LaterCRS", _make_srs(32611), None)
        dialog._update_user_created_crs_cbox()

        self.assertGreaterEqual(dialog._ui.cbox_user_crs.findText("LaterCRS"), 0)


if __name__ == "__main__":
    unittest.main()
