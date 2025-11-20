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
from astropy import units as u

from wiser.gui.ui_library import (
    DatasetChooserDialog,
    SpectrumChooserDialog,
    ROIChooserDialog,
    BandChooserDialog,
    TableDisplayWidget,
)

from wiser.raster.spectrum import NumPyArraySpectrum

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

    def test_table_display_widget(self):
        header = ["Header1", "Header2", "Header3"]
        rows = [
            ["r1c1", "r1c2", "r1c3"],
            ["r2c1", "r2c2", "r2c3"],
            ["r3c1", "r3c2", "r3c3"],
        ]
        window_title = "Testing Title"
        description = "Test Description"

        app_state = self.test_model.app_state
        app_state.show_table_widget(
            header=header,
            rows=rows,
            window_title=window_title,
            description=description,
        )

        table_widget_set = app_state._table_display_widgets

        self.assertTrue(len(table_widget_set) == 1)

        table_widget: TableDisplayWidget = next(iter(table_widget_set))

        self.assertTrue(table_widget._table.rowCount() == len(rows))
        self.assertTrue(table_widget._table.columnCount() == len(header))
        self.assertTrue(table_widget.windowTitle() == window_title)
        self.assertTrue(table_widget._description_label.text() == description)

    def test_spectrum_plot_generic(self):
        test_spectrum_y = np.array(
            [
                0.25744912028312683,
                0.2996889650821686,
                0.07340309023857117,
                0.09369881451129913,
            ]
        )
        test_spectrum_x = [
            472.019989 * u.nanometer,
            532.130005 * u.nanometer,
            702.419983 * u.nanometer,
            852.679993 * u.nanometer,
        ]
        truth_x_range = (452.9869888, 871.7129931999999)
        truth_y_range = (0.047104348242282865, 0.6256766721606255)

        double = test_spectrum_y + test_spectrum_y
        spec1 = NumPyArraySpectrum(test_spectrum_y, "Test_spectrum1", wavelengths=test_spectrum_x)
        spec2 = NumPyArraySpectrum(double, "Test_spectrum2", wavelengths=test_spectrum_x)

        app_state = self.test_model.app_state
        app_state.show_spectra_in_plot([spec1, spec2])

        generic_sp = next(iter(app_state._generic_spectrum_plots))

        self.assertTrue(generic_sp._plot_uses_wavelengths is not None and generic_sp._plot_uses_wavelengths)
        self.assertTrue(generic_sp.get_x_range() == truth_x_range)
        self.assertTrue(generic_sp.get_y_range() == truth_y_range)


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
