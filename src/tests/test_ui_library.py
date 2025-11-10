"""
Tests to make sure the UI library works as intended
"""
import unittest

from matplotlib import use

import tests.context
# import context

from test_utils.test_model import WiserTestModel
from test_utils.test_function_decorator import run_in_wiser_decorator

import numpy as np

from wiser.gui.ui_library import (
    DatasetChooserDialog,
    SpectrumChooserDialog,
    ROIChooserDialog,
    BandChooserDialog,
)

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import time

import pytest

pytestmark = [
    pytest.mark.unit,
]

np_arr = np.array(
    [
        [
            [0.0, np.nan, 0.0, 1.0],
            [0.25, 0.25, 0.25, np.nan],
            [0.5, 0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75, 0.75],
            [1.0, 1.0, 1.0, 1.0],
        ],
        [
            [0.0, 1.0, np.nan, 0.0],
            [0.25, 0.25, np.nan, 0.25],
            [0.5, 0.5, np.nan, 0.5],
            [0.75, 0.75, np.nan, 0.75],
            [1.0, 1.0, np.nan, 1.0],
        ],
        [
            [0.0, 2.0, 1.0, 2.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, np.nan, 0.5, 0.5],
            [0.75, 0.75, np.nan, 0.75],
            [np.nan, 1.0, 1.0, 1.0],
        ],
    ]
)

np_mask = np.array(
    [
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ],
    ]
)

np_impl = np.ma.array(np_arr, mask=np_mask)


class TestUILibrary(unittest.TestCase):
    """
    Tests the UI library features

    Attributes:
        test_model (WiserTestModel): Wrapper for controlling the WISER GUI and accessing state.
    """

    def setUp(self):
        """Initializes the WISER test model before each test."""
        self.test_model = WiserTestModel(use_gui=False)

    def tearDown(self):
        """Closes the WISER application and cleans up after each test."""
        self.test_model.close_app()
        del self.test_model

    def test_dataset_chooser_dialog(self):
        ds1 = self.test_model.load_dataset(np_impl)
        ds1.set_name("Numpy1")

        ds2 = self.test_model.load_dataset(np_impl)
        ds2.set_name("Numpy2")

        # dataset_chooser_dialog = DatasetChooserDialog(app_state=app_state, parent=None)
        self.test_model.open_plugin_dataset_chooser()
        self.test_model.select_plugin_dataset(ds_id=ds2.get_id())
        chosen_ds = self.test_model.accept_plugin_dataset_chooser()

        self.assertTrue(chosen_ds.get_name() == ds2.get_name())

    def test_spectrum_choose_dialog(self):
        ds = self.test_model.load_dataset(np_impl)
        ds.set_name("Numpy")

        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))
        self.test_model.get_active_spectrum()
        self.test_model.collect_active_spectrum()

        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 1))
        sp2 = self.test_model.get_active_spectrum()
        self.test_model.collect_active_spectrum()

        self.test_model.open_plugin_spectrum_chooser()
        self.test_model.select_plugin_spectrum(sp2.get_id())
        chosen_sp = self.test_model.accept_plugin_spectrum_chooser()

        self.assertTrue(chosen_sp.get_name() == sp2.get_name())

    def test_band_choose_dialog(self):
        ds1 = self.test_model.load_dataset(np_impl)
        ds1.set_name("Numpy1")

        ds2 = self.test_model.load_dataset(np_impl)
        ds2.set_name("Numpy2")

        band_idx = 2

        self.test_model.open_plugin_band_chooser()
        self.test_model.select_plugin_band(dataset=ds2, band_idx=band_idx)
        band = self.test_model.accept_plugin_band_chooser()

        assert ds2.get_name() == band.get_dataset().get_name()
        assert band_idx == band.get_band_index()


if __name__ == "__main__":
    test_model = WiserTestModel(use_gui=True)

    ds1 = test_model.load_dataset(np_impl)
    ds1.set_name("Numpy1")

    ds2 = test_model.load_dataset(np_impl)
    ds2.set_name("Numpy2")

    band_idx = 2

    test_model.open_plugin_band_chooser()
    test_model.select_plugin_band(dataset=ds2, band_idx=band_idx)
    band = test_model.accept_plugin_band_chooser()

    assert ds2.get_name() == band.get_dataset().get_name()
    assert band_idx == band.get_band_index()

    test_model.app.exec_()
