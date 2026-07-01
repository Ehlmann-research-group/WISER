import unittest

import tests.context  # noqa: F401  (adds src/ to sys.path)
from test_utils.test_model import WiserTestModel
from wiser.gui.mosaic_dialog import SeamlessMosaicDialog

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.integration,
]


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


if __name__ == "__main__":
    unittest.main()
