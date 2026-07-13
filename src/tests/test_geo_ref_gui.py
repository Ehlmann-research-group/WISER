"""Integration tests for WISER's GeoReferencer GUI.

This module validates the complete georeferencing workflow using the WISER GUI.
It simulates user actions like setting GCPs, choosing CRS values, and applying 
geometric transformations. It verifies the output against known ground truth.

What's covered:
- End-to-end raster georeferencing with image-based and manual CRS entry.
- Enabling/disabling/removing GCPs.
- Validating geo-transformation output against expected results.

What's not covered:
- Internal filtering logic for filename/path selection.
"""
import unittest

import os

import tests.context
# import context

from test_utils.test_model import WiserTestModel

from wiser.gui.geo_reference_dialog import (
    AuthorityCodeCRS,
    UserGeneratedCRS,
    GeneralCRS,
)
from wiser.gui.geo_reference_config import GeoReferencerConfig

import numpy as np

from PySide6.QtTest import QTest
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import pytest

pytestmark = [
    pytest.mark.functional,
]


class TestGeoReferencerGUI(unittest.TestCase):
    """Test case for validating the GeoReferencer UI workflow in WISER.

    This class simulates user-driven georeferencing operations via the GUI,
    including GCP input, CRS selection, interpolation setting, and file export.
    It ensures that warping is correctly applied and results in an expected
    affine transform.

    Attributes:
        test_model (WiserTestModel): Interface to simulate GUI operations in WISER.
    """

    def setUp(self):
        """Initializes the test model before each test by launching the WISER app."""
        self.test_model = WiserTestModel()

    def tearDown(self):
        """Cleans up after each test by closing the WISER application."""
        self.test_model.close_app()
        del self.test_model

    def test_rasterpane_e2e(self):
        """Performs an end-to-end test of image-based georeferencing through the GUI.

        This test:
        - Loads a dataset.
        - Opens the GeoReferencer dialog.
        - Adds GCPs with both target and reference image clicks.
        - Sets interpolation type, CRS, polynomial order, and output path.
        - Disables and removes invalid GCPs.
        - Runs the warp operation.
        - Compares the resulting geo-transform to a known ground truth.
        """
        rel_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "caltech_4_100_150_nm",
        )
        ds = self.test_model.load_dataset(rel_path)

        self.test_model.open_geo_referencer()

        self.test_model.set_geo_ref_target_dataset(ds.get_id())
        self.test_model.set_geo_ref_reference_dataset(ds.get_id())

        gcp_list = [
            [(12, 2), (395615.1733312448, 3778287.4352373052)],
            [(22, 133), (395625.94761462574, 3778024.940670322)],
            [(142, 131), (395865.7763863942, 3778020.691346924)],
            [(144, 4), (395878.48156122735, 3778274.252152171)],
            [(80, 84), (395745.30016513495, 3778118.96401564)],
            [(44, 99), (395672.2851735266, 3778091.6713232244)],
            [
                (56, 28),
                (395751.2234671218, 3778276.7707845406),
            ],  # This point is incorrect, we disable it
            [
                (64, 122),
                (395751.2234671218, 3778276.7707845406),
            ],  # This point is incorrect, we remove it
        ]
        for gcp in gcp_list:
            target = gcp[0]
            ref = gcp[1]
            self.test_model.click_target_image(target)
            self.test_model.press_enter_target_image()
            self.test_model.click_reference_image_spatially(ref)
            self.test_model.press_enter_reference_image()

        self.test_model.set_interpolation_type("GRA_Bilinear")

        self.test_model.set_geo_ref_output_crs(AuthorityCodeCRS("EPSG", 4326))

        self.test_model.set_geo_ref_polynomial_order("2")

        rel_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "artifacts",
            "test_warp_output.tif",
        )
        self.test_model.set_geo_ref_file_save_path(rel_path)

        ground_truth_geo_transform = (
            -118.13251959615042,
            1.9876715549174167e-05,
            0.0,
            34.14031070448446,
            0.0,
            -1.9876715549174167e-05,
        )

        # Disables and re-enables
        self.test_model.click_gcp_enable_btn_geo_ref(0)
        self.test_model.click_gcp_enable_btn_geo_ref(0)
        # Disables invalid point
        self.test_model.click_gcp_enable_btn_geo_ref(6)

        # Removes invalid point
        self.test_model.remove_gcp_geo_ref(7)

        self.test_model.click_run_warp()

        ds_warp = self.test_model.load_dataset(rel_path)

        warped_transform = ds_warp.get_geo_transform()
        self.assertTrue(np.allclose(warped_transform, ground_truth_geo_transform))

    def test_manual_entry_e2e(self):
        """Performs an end-to-end test of manually entering geocoordinates in the GeoReferencer.

        This test:
        - Loads a dataset and opens the GeoReferencer.
        - Adds GCPs using manual latitude/longitude input instead of spatial clicks.
        - Selects and verifies CRS entries.
        - Sets interpolation and warp options.
        - Disables and removes invalid GCPs.
        - Executes the warp and checks the output affine transform.
        """
        rel_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "caltech_4_100_150_nm",
        )
        ds = self.test_model.load_dataset(rel_path)

        self.test_model.open_geo_referencer()

        self.test_model.set_geo_ref_target_dataset(ds.get_id())

        gcp_list = [
            [(12, 2), (395615.1733312448, 3778287.4352373052)],
            [(22, 133), (395625.94761462574, 3778024.940670322)],
            [(142, 131), (395865.7763863942, 3778020.691346924)],
            [(144, 4), (395878.48156122735, 3778274.252152171)],
            [(80, 84), (395745.30016513495, 3778118.96401564)],
            [(44, 99), (395672.2851735266, 3778091.6713232244)],
            [
                (56, 28),
                (395751.2234671218, 3778276.7707845406),
            ],  # This point is incorrect, we disable it
            [
                (64, 122),
                (395751.2234671218, 3778276.7707845406),
            ],  # This point is incorrect, we remove it
        ]

        self.test_model.set_interpolation_type("GRA_Bilinear")

        self.test_model.set_geo_ref_output_crs(AuthorityCodeCRS("EPSG", 4326))

        self.test_model.set_geo_ref_polynomial_order("2")

        self.test_model.select_manual_authority_ref("EPSG")

        self.test_model.enter_manual_authority_code_ref("32611")

        self.test_model.click_find_crs_ref()

        self.test_model.choose_manual_crs_geo_ref(AuthorityCodeCRS("EPSG", 4326))
        self.test_model.choose_manual_crs_geo_ref(AuthorityCodeCRS("EPSG", 32611))

        for gcp in gcp_list:
            target = gcp[0]
            ref = gcp[1]
            lat_north = ref[1]
            lon_east = ref[0]
            self.test_model.click_target_image(target)
            self.test_model.press_enter_target_image()
            self.test_model.enter_lat_north_geo_ref(lat_north)
            self.test_model.press_enter_lat_north_geo_ref()
            self.test_model.enter_lon_east_geo_ref(lon_east)
            self.test_model.press_enter_lon_east_geo_ref()

        rel_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "artifacts",
            "test_warp_output.tif",
        )
        self.test_model.set_geo_ref_file_save_path(rel_path)

        ground_truth_geo_transform = (
            -118.13251959615042,
            1.9876715549174167e-05,
            0.0,
            34.14031070448446,
            0.0,
            -1.9876715549174167e-05,
        )

        # Disables and re-enables
        self.test_model.click_gcp_enable_btn_geo_ref(0)
        self.test_model.click_gcp_enable_btn_geo_ref(0)
        # Disables invalid point
        self.test_model.click_gcp_enable_btn_geo_ref(6)

        # Removes invalid point
        self.test_model.remove_gcp_geo_ref(7)

        self.test_model.click_run_warp()

        ds_warp = self.test_model.load_dataset(rel_path)

        warped_transform = ds_warp.get_geo_transform()
        self.assertTrue(np.allclose(warped_transform, ground_truth_geo_transform))

    def test_config_driven_locking(self):
        """A GeoReferencerConfig presets + locks the choosers; config=None restores them.

        Verifies:
        - presetting target dataset + save path populates the panes/fields,
        - lock flags disable exactly the right widgets,
        - a custom accept_button_text relabels the OK button,
        - re-applying config=None (the Tools-menu path) re-enables everything.
        """
        rel_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "caltech_4_100_150_nm",
        )
        ds = self.test_model.load_dataset(rel_path)
        self.test_model.open_geo_referencer()
        dialog = self.test_model.main_window._geo_ref_dialog

        save_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "artifacts",
            "config_locked_output.tif",
        )
        config = GeoReferencerConfig(
            target_dataset=ds,
            save_path=save_path,
            allow_change_target=False,
            allow_change_save_path=False,
            accept_button_text="Save to Mosaic",
        )
        self.test_model.apply_geo_ref_config(config)

        # Presets applied.
        self.assertEqual(dialog._target_rasterpane.get_rasterview().get_raster_data().get_id(), ds.get_id())
        self.assertEqual(dialog._ui.ledit_save_path.text(), save_path)

        # Locks disable exactly the right widgets.
        self.assertFalse(dialog._target_cbox.isEnabled())
        self.assertTrue(dialog._ui.ledit_save_path.isReadOnly())
        self.assertFalse(dialog._ui.btn_save_path.isEnabled())
        # Reference was left unlocked.
        self.assertTrue(dialog._reference_cbox.isEnabled())

        # Accept button relabeled.
        ok_btn = dialog._ui.buttonBox.button(QDialogButtonBox.Ok)
        self.assertEqual(ok_btn.text(), "Save to Mosaic")

        # config=None restores the classic Tools-menu state.
        self.test_model.apply_geo_ref_config(None)
        self.assertTrue(dialog._target_cbox.isEnabled())
        self.assertFalse(dialog._ui.ledit_save_path.isReadOnly())
        self.assertTrue(dialog._ui.btn_save_path.isEnabled())
        self.assertEqual(ok_btn.text(), self._default_ok_text(dialog))

    def test_zoom_to_fit(self):
        """The georeferencer panes auto-fit on load and expose a Zoom-to-fit button.

        Verifies:
        - each pane has a "Zoom to fit" toolbar action,
        - loading a dataset auto-fits the target pane,
        - the button re-frames the image after the user has zoomed away, back to
          the same auto-fit scale.
        """
        rel_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "caltech_4_100_150_nm",
        )
        ds = self.test_model.load_dataset(rel_path)
        self.test_model.open_geo_referencer()
        dialog = self.test_model.main_window._geo_ref_dialog

        # Both panes expose the Zoom-to-fit action.
        self.assertTrue(hasattr(dialog._target_rasterpane, "_act_zoom_to_fit"))
        self.assertTrue(hasattr(dialog._reference_rasterpane, "_act_zoom_to_fit"))

        # Loading a dataset auto-fits the target pane.  The fit is deferred to a
        # single-shot timer that fires in the event loop pumped by
        # apply_geo_ref_config; an extra run() gives it a safety turn.
        config = GeoReferencerConfig(target_dataset=ds)
        self.test_model.apply_geo_ref_config(config)
        self.test_model.run()

        auto_fit_scale = self.test_model.get_geo_ref_target_scale()
        self.assertGreater(auto_fit_scale, 0.0)

        # Zoom the user well away from the fit...
        self.test_model.set_geo_ref_target_scale(8.0)
        self.assertAlmostEqual(self.test_model.get_geo_ref_target_scale(), 8.0, places=5)

        # ...then the Zoom-to-fit button re-frames the image, back to the same
        # scale the auto-fit produced (proving the button is not a no-op and that
        # the auto-fit ran).
        self.test_model.click_geo_ref_target_zoom_to_fit()
        button_fit_scale = self.test_model.get_geo_ref_target_scale()
        self.assertNotAlmostEqual(button_fit_scale, 8.0, places=5)
        self.assertAlmostEqual(button_fit_scale, auto_fit_scale, places=5)

    @staticmethod
    def _default_ok_text(dialog):
        # The remembered original label of the OK button.
        return dialog._default_accept_button_text


"""
Code to make sure tests work as desired
"""
if __name__ == "__main__":
    test_model = WiserTestModel(use_gui=True)

    rel_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test_utils",
        "test_datasets",
        "caltech_4_100_150_nm",
    )
    ds = test_model.load_dataset(rel_path)

    test_model.open_geo_referencer()

    test_model.set_geo_ref_target_dataset(ds.get_id())

    gcp_list = [
        [(12, 2), (395615.1733312448, 3778287.4352373052)],
        [(22, 133), (395625.94761462574, 3778024.940670322)],
        [(142, 131), (395865.7763863942, 3778020.691346924)],
        [(144, 4), (395878.48156122735, 3778274.252152171)],
        [(80, 84), (395745.30016513495, 3778118.96401564)],
        [(44, 99), (395672.2851735266, 3778091.6713232244)],
        [
            (56, 28),
            (395751.2234671218, 3778276.7707845406),
        ],  # This point is incorrect, we disable it
        [
            (64, 122),
            (395751.2234671218, 3778276.7707845406),
        ],  # This point is incorrect, we remove it
    ]

    test_model.set_interpolation_type("GRA_Bilinear")

    test_model.set_geo_ref_output_crs(AuthorityCodeCRS("EPSG", 4326))

    test_model.set_geo_ref_polynomial_order("2")

    test_model.select_manual_authority_ref("EPSG")

    test_model.enter_manual_authority_code_ref("32611")

    test_model.click_find_crs_ref()

    test_model.choose_manual_crs_geo_ref(AuthorityCodeCRS("EPSG", 4326))
    test_model.choose_manual_crs_geo_ref(AuthorityCodeCRS("EPSG", 32611))

    for gcp in gcp_list:
        target = gcp[0]
        ref = gcp[1]
        lat_north = ref[1]
        lon_east = ref[0]
        test_model.click_target_image(target)
        test_model.press_enter_target_image()
        test_model.enter_lat_north_geo_ref(lat_north)
        test_model.press_enter_lat_north_geo_ref()
        test_model.enter_lon_east_geo_ref(lon_east)
        test_model.press_enter_lon_east_geo_ref()

    rel_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test_utils",
        "test_datasets",
        "artifacts",
        "test_warp_output.tif",
    )
    test_model.set_geo_ref_file_save_path(rel_path)

    # Disables and re-enables
    test_model.click_gcp_enable_btn_geo_ref(0)
    test_model.click_gcp_enable_btn_geo_ref(0)
    # Disables invalid point
    test_model.click_gcp_enable_btn_geo_ref(6)

    # Removes invalid point
    test_model.remove_gcp_geo_ref(7)

    test_model.click_run_warp()

    ds_warp = test_model.load_dataset(rel_path)

    print(f"Warping geo transform: {ds_warp.get_geo_transform()}")

    test_model.close_geo_referencer()

    test_model.app.exec_()
