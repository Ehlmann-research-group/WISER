"""GUI tests: binding band-math band variables by wavelength, and index presets.

These drive the real ``BandMathDialog`` against a real ``ApplicationState``
(``WiserTestModel``'s offscreen ``QApplication``).  A band variable in
wavelength mode resolves to the nearest band within the tolerance, shows that
band (with its actual wavelength) in the band chooser, and binds a plain
``RasterDataBand``; a wavelength with no band in tolerance, or a dataset without
wavelengths, is reported as an error rather than bound to some other band.
"""

import os
import unittest
from unittest import mock

import tests.context  # noqa: F401

from PySide6.QtCore import Qt

from test_utils.test_model import WiserTestModel
from wiser.bandmath import VariableType
from wiser.bandmath.presets import DEFAULT_WAVELENGTH_TOLERANCE_NM
from wiser.gui.bandmath_dialog import BandMathDialog, DatasetBandChooserWidget, ImageBandBatchChooserWidget
from wiser.raster.dataset import RasterDataBand

TEST_DATASETS = os.path.join(os.path.dirname(__file__), "..", "test_utils", "test_datasets")
EMIT_PATH = os.path.join(TEST_DATASETS, "EMIT_L2A_RFL_001_20260821T180123_2623312_016_subset_aid0000.nc")
NO_WAVELENGTHS_PATH = os.path.join(TEST_DATASETS, "envi.hdr")


class TestBandMathWavelengthBinding(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        self.app_state = self.test_model.app_state

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _load(self, path):
        dataset = self.test_model.raster_data_loader.load_from_file(
            path=path, data_cache=self.test_model.data_cache, interactive=False
        )[0]
        self.app_state.add_dataset(dataset)
        return dataset

    def _make_dialog(self):
        dialog = BandMathDialog(self.app_state, parent=self.test_model.main_window)
        dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
        self.addCleanup(dialog.close)
        return dialog

    def _set_expression(self, dialog, expr):
        dialog._ui.ledit_expression.setText(expr)
        dialog._analyze_expr()

    def _band_widget(self, dialog, var) -> DatasetBandChooserWidget:
        row = dialog._find_variable_in_bindings(var)
        self.assertNotEqual(row, -1)
        widget = dialog._ui.tbl_variables.cellWidget(row, 2)
        self.assertIsInstance(widget, DatasetBandChooserWidget)
        return widget

    def _enter_wavelength(self, widget, value, unit_key, tolerance):
        widget.mode_chooser.setCurrentIndex(
            widget.mode_chooser.findData(ImageBandBatchChooserWidget.Mode.WAVELENGTH.value)
        )
        widget.cbox_wavelength_unit.setCurrentText(unit_key)
        widget.ledit_wavelength.setText(value)
        widget.ledit_tolerance.setText(tolerance)
        widget.ledit_wavelength.editingFinished.emit()

    def test_wavelength_mode_defaults(self):
        self._load(EMIT_PATH)
        dialog = self._make_dialog()
        self._set_expression(dialog, "nir + 1")
        widget = self._band_widget(dialog, "nir")
        widget.mode_chooser.setCurrentIndex(
            widget.mode_chooser.findData(ImageBandBatchChooserWidget.Mode.WAVELENGTH.value)
        )

        self.assertEqual(widget.cbox_wavelength_unit.currentText(), "nm")
        self.assertEqual(float(widget.ledit_tolerance.text()), DEFAULT_WAVELENGTH_TOLERANCE_NM)
        self.assertFalse(widget.band_chooser.isEnabled())

    def test_bind_by_wavelength_shows_and_binds_nearest_band(self):
        dataset = self._load(EMIT_PATH)
        dialog = self._make_dialog()
        self._set_expression(dialog, "nir + 1")
        widget = self._band_widget(dialog, "nir")

        self._enter_wavelength(widget, "860", "nm", "10")

        self.assertEqual(widget.band_chooser.currentData(), 64)
        self.assertEqual(widget.band_chooser.currentText(), "Band 64: 857.45 nm")
        self.assertEqual(widget.lbl_status.text(), "")

        var_type, value = dialog.get_variable_bindings()["nir"]
        self.assertEqual(var_type, VariableType.IMAGE_BAND)
        self.assertIsInstance(value, RasterDataBand)
        self.assertIs(value.get_dataset(), dataset)
        self.assertEqual(value.get_band_index(), 64)
        self.assertIsNotNone(dialog.get_expression_info())

    def test_bind_by_wavelength_in_micrometers(self):
        self._load(EMIT_PATH)
        dialog = self._make_dialog()
        self._set_expression(dialog, "nir + 1")
        widget = self._band_widget(dialog, "nir")

        self._enter_wavelength(widget, "0.86", "um", "10")

        self.assertEqual(dialog.get_variable_bindings()["nir"][1].get_band_index(), 64)

    def test_wavelength_out_of_range_reports_error_and_does_not_bind(self):
        self._load(EMIT_PATH)
        dialog = self._make_dialog()
        self._set_expression(dialog, "nir + 1")
        widget = self._band_widget(dialog, "nir")

        self._enter_wavelength(widget, "3000", "nm", "10")

        self.assertIn("No band", widget.lbl_status.text())
        self.assertEqual(widget.band_chooser.currentIndex(), -1)
        with self.assertRaisesRegex(ValueError, "No band"):
            dialog.get_variable_bindings()
        self.assertIsNone(dialog.get_expression_info())
        self.assertIn("No band", dialog._ui.lbl_result_info.text())

    @mock.patch("wiser.gui.bandmath_dialog.QMessageBox.critical")
    def test_accept_refuses_unresolved_wavelength(self, mock_critical):
        self._load(EMIT_PATH)
        dialog = self._make_dialog()
        self._set_expression(dialog, "nir + 1")
        self._enter_wavelength(self._band_widget(dialog, "nir"), "3000", "nm", "10")

        with mock.patch.object(BandMathDialog.__bases__[0], "accept") as base_accept:
            dialog.accept()

        base_accept.assert_not_called()
        mock_critical.assert_called_once()
        self.assertIn("No band", mock_critical.call_args.args[2])

    def test_dataset_without_wavelengths_reports_error(self):
        self._load(NO_WAVELENGTHS_PATH)
        dialog = self._make_dialog()
        self._set_expression(dialog, "nir + 1")
        widget = self._band_widget(dialog, "nir")

        self._enter_wavelength(widget, "860", "nm", "10")

        self.assertIn("has no band wavelengths", widget.lbl_status.text())
        with self.assertRaisesRegex(ValueError, "has no band wavelengths"):
            dialog.get_variable_bindings()

    def test_index_mode_is_unchanged(self):
        self._load(EMIT_PATH)
        dialog = self._make_dialog()
        self._set_expression(dialog, "nir + 1")
        widget = self._band_widget(dialog, "nir")

        widget.band_chooser.setCurrentIndex(52)

        self.assertTrue(widget.band_chooser.isEnabled())
        self.assertEqual(dialog.get_variable_bindings()["nir"][1].get_band_index(), 52)

    def test_band_chooser_labels_carry_units(self):
        self._load(EMIT_PATH)
        dialog = self._make_dialog()
        self._set_expression(dialog, "nir + 1")
        widget = self._band_widget(dialog, "nir")

        self.assertEqual(widget.band_chooser.itemText(52), "Band 52: 767.91 nm")

    def test_ndvi_preset_sets_expression_and_wavelength_bindings(self):
        dataset = self._load(EMIT_PATH)
        dialog = self._make_dialog()
        presets = dialog._ui.cbox_index_presets
        index = presets.findData("NDVI")
        self.assertNotEqual(index, -1)

        presets.setCurrentIndex(index)
        presets.activated.emit(index)

        self.assertEqual(dialog.get_expression(), "(nir - red) / (nir + red)")
        for var, wavelength, band in (("nir", "860", 64), ("red", "660", 38)):
            widget = self._band_widget(dialog, var)
            self.assertEqual(
                widget.mode_chooser.currentData(), ImageBandBatchChooserWidget.Mode.WAVELENGTH.value
            )
            self.assertEqual(float(widget.ledit_wavelength.text()), float(wavelength))
            self.assertEqual(widget.band_chooser.currentData(), band)

        bindings = dialog.get_variable_bindings()
        self.assertEqual(bindings["nir"][1].get_band_index(), 64)
        self.assertEqual(bindings["red"][1].get_band_index(), 38)
        self.assertIs(bindings["red"][1].get_dataset(), dataset)
        self.assertIsNotNone(dialog.get_expression_info())

    def test_msi_preset_overrides_guessed_variable_types(self):
        self._load(EMIT_PATH)
        dialog = self._make_dialog()
        presets = dialog._ui.cbox_index_presets
        index = presets.findData("MSI")
        presets.setCurrentIndex(index)
        presets.activated.emit(index)

        bindings = dialog.get_variable_bindings()
        self.assertEqual(bindings["r1600"][1].get_band_index(), 164)
        self.assertEqual(bindings["r820"][1].get_band_index(), 59)

    def test_presets_are_not_saved_expressions(self):
        self._load(EMIT_PATH)
        dialog = self._make_dialog()
        presets = dialog._ui.cbox_index_presets
        index = presets.findData("NDWI")
        presets.setCurrentIndex(index)
        presets.activated.emit(index)

        self.assertEqual(dialog._ui.cbox_saved_exprs.count(), 0)
        self.assertEqual(self.app_state.get_bandmath_expressions(), [])


if __name__ == "__main__":
    unittest.main()
