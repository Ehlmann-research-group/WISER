"""GUI sync tests: BandMathDialog saved expressions <-> ApplicationState (issue #625).

The saved-expressions combo box is backed by ``ApplicationState`` so the list
survives the dialog closing and is persisted with the project file.  These tests
drive the real ``BandMathDialog`` against a real ``ApplicationState`` (via
``WiserTestModel``'s offscreen ``QApplication``) and assert the two stay in sync:
clicking "Save expression" and "Load from file..." pushes into application state,
and a dialog seeded from a pre-populated application state shows the stored
expressions -- including a re-seed on show, so a project load is reflected in a
cached dialog.
"""

import os
import tempfile
import unittest
from unittest import mock

import tests.context  # noqa: F401

from PySide6.QtCore import Qt
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QMessageBox

from test_utils.test_model import WiserTestModel
from wiser.gui.bandmath_dialog import BandMathDialog


class TestBandMathSavedExprsSync(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        self.app_state = self.test_model.app_state

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _make_dialog(self):
        dialog = BandMathDialog(self.app_state, parent=self.test_model.main_window)
        dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
        self.addCleanup(dialog.close)
        return dialog

    @staticmethod
    def _combo_items(dialog):
        cbox = dialog._ui.cbox_saved_exprs
        return [cbox.itemText(i) for i in range(cbox.count())]

    def test_save_expression_updates_app_state(self):
        dialog = self._make_dialog()
        dialog._ui.ledit_expression.setText("b1 + b2")
        dialog._ui.btn_add_to_saved.click()

        self.assertEqual(self.app_state.get_bandmath_expressions(), ["b1 + b2"])
        self.assertEqual(self._combo_items(dialog), ["b1 + b2"])

    @mock.patch("wiser.gui.bandmath_dialog.QMessageBox.question", return_value=QMessageBox.Yes)
    def test_load_from_file_updates_app_state(self, _mock_question):
        fd, path = tempfile.mkstemp(suffix=".txt")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("B1 + B2\n\n  B3 / B4  \n")

            dialog = self._make_dialog()
            with mock.patch(
                "wiser.gui.bandmath_dialog.QFileDialog.getOpenFileName",
                return_value=(path, ""),
            ):
                dialog._ui.btn_load_saved_exprs.click()
        finally:
            os.unlink(path)

        # The handler strips, lowercases, and drops blank lines.
        self.assertEqual(self.app_state.get_bandmath_expressions(), ["b1 + b2", "b3 / b4"])
        self.assertEqual(self._combo_items(dialog), ["b1 + b2", "b3 / b4"])

    def test_dialog_seeds_combo_from_app_state(self):
        self.app_state.set_bandmath_expressions(["b1 + b2", "b3 / b4"])
        dialog = self._make_dialog()
        self.assertEqual(self._combo_items(dialog), ["b1 + b2", "b3 / b4"])

    def test_show_event_reseeds_after_app_state_change(self):
        dialog = self._make_dialog()
        self.assertEqual(self._combo_items(dialog), [])

        # A project load repopulates application state while the cached dialog exists;
        # the next show must reflect it rather than the stale (empty) list.
        self.app_state.set_bandmath_expressions(["nir / red"])
        dialog.showEvent(QShowEvent())

        self.assertEqual(self._combo_items(dialog), ["nir / red"])


if __name__ == "__main__":
    unittest.main()
