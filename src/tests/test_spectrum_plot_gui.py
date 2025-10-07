"""Unit tests for the Spectrum Plot UI in WISER.

This module verifies that the spectrum plot behaves correctly when interacting with datasets
via raster view and zoom pane. Tests include spectrum selection, wavelength units, and spectrum
state updates across different datasets with and without spectral metadata.
"""
import os

import unittest

import tests.context
# import context

from test_utils.test_model import WiserTestModel

import numpy as np
from astropy import units as u

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class TestSpectrumPlotUI(unittest.TestCase):
    """Tests for spectrum plotting and user interaction in the WISER GUI.

    Simulates user actions such as clicking on raster views and zoom panes,
    and validates spectrum extraction and unit handling in the spectrum plot UI.

    Attributes:
        test_model (WiserTestModel): A GUI-driven test controller for interacting with WISER.
    """

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_click(self):
        """Tests spectrum extraction and plot click interaction.

        Loads a simple dataset, simulates a click on the raster view, and then on the spectrum plot.
        Verifies the extracted spectrum and clicked coordinate match expected values.
        """
        np_impl = np.array([[[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[1.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.5  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]]])
        
        self.test_model.load_dataset(np_impl)

        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        self.test_model.click_spectrum_plot(1, 0.5)

        active_spectrum = self.test_model.get_active_spectrum()

        active_spectrum_arr = active_spectrum.get_spectrum()

        expected_spectrum_arr = np.array([0, 1, 0.5])

        self.assertTrue(np.array_equal(active_spectrum_arr, expected_spectrum_arr))

        clicked_point = self.test_model.get_clicked_spectrum_plot_point()

        self.assertTrue(np.array_equal(np.array(clicked_point), np.array((1., 1.0))))

    def test_wavelength_main_view(self):
        """Tests that wavelength units are shown in the spectrum plot after clicking in the main view.

        Loads an ENVI dataset and verifies that the x-axis of the spectrum plot is labeled in nanometers.
        """
        rel_path = os.path.join("..", "test_utils", "test_datasets", "caltech_4_100_150_nm")
        self.test_model.load_dataset(rel_path)

        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        spectrum_plot_units = self.test_model.get_spectrum_plot_x_units()

        correct_unit = u.nanometer

        self.assertTrue(spectrum_plot_units == correct_unit)

    def test_wavelength_zoom_pane(self):
        """Tests that wavelength units are shown in the spectrum plot after clicking in the zoom pane.

        Loads an ENVI dataset and verifies that the x-axis of the spectrum plot is labeled in nanometers.
        """
        rel_path = os.path.join("..", "test_utils", "test_datasets", "caltech_4_100_150_nm")
        self.test_model.load_dataset(rel_path)

        self.test_model.click_raster_coord_zoom_pane((0, 0))

        spectrum_plot_units = self.test_model.get_spectrum_plot_x_units()

        correct_unit = u.nanometer

        self.assertTrue(spectrum_plot_units == correct_unit)
    
    def test_changing_wavelengths(self):
        """Tests switching between datasets with different wavelength units.

        Verifies that the spectrum plot correctly updates its x-axis units when switching between
        datasets using nanometers and micrometers.
        """
        self.test_model.set_main_view_layout((1, 2))

        # This will be in the (0, 0) raster view position
        rel_path = os.path.join("..", "test_utils", "test_datasets", "caltech_4_100_150_nm")
        self.test_model.load_dataset(rel_path)

        # This will be in the (0, 1) raster view position
        rel_path = os.path.join("..", "test_utils", "test_datasets", "circuit_4_100_150_um")
        self.test_model.load_dataset(rel_path)

        # Switch back to nanometer units
        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))
        spectrum_plot_units = self.test_model.get_spectrum_plot_x_units()
        correct_unit = u.nanometer
        self.assertTrue(spectrum_plot_units == correct_unit)

        # Switch to micrometer units
        self.test_model.click_raster_coord_main_view_rv((0, 1), (0, 0))
        spectrum_plot_units = self.test_model.get_spectrum_plot_x_units()
        correct_unit = u.micrometer
        self.assertTrue(spectrum_plot_units == correct_unit)

        # Switch back to nanometer units
        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))
        spectrum_plot_units = self.test_model.get_spectrum_plot_x_units()
        correct_unit = u.nanometer
        self.assertTrue(spectrum_plot_units == correct_unit)
    
    def test_no_use_wavelengths(self):
        """Tests behavior when switching from a dataset with wavelengths to one without.

        Verifies that the spectrum plot disables wavelength display after switching to a dataset
        that lacks spectral metadata.
        """
        np_impl = np.array([[[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[1.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.5  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]]])

        np_ds = self.test_model.load_dataset(np_impl)
    
        rel_path = os.path.join("..", "test_utils", "test_datasets", "caltech_4_100_150_nm")
        self.test_model.load_dataset(rel_path)

        # Get a spectrum with wavelength nm in spectrum plot
        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))
        spectrum_plot_units = self.test_model.get_spectrum_plot_x_units()
        correct_unit = u.nanometer
        self.assertTrue(spectrum_plot_units == correct_unit)

        # Ensure the plot uses wavelengths
        plot_use_wavelengths = self.test_model.get_spectrum_plot_use_wavelengths()
        self.assertTrue(True == plot_use_wavelengths)

        # Collect the spectrum
        self.test_model.collect_active_spectrum()

        # Change to view without units and click a spectrum
        self.test_model.set_main_view_rv((0, 0), np_ds.get_id())
        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        # Ensure the plot no longer uses wavelengths 
        plot_use_wavelengths = self.test_model.get_spectrum_plot_use_wavelengths()
        self.assertTrue(False == plot_use_wavelengths)
    
    def test_use_wavelengths(self):
        """Tests spectrum plot update when switching between datasets with and without wavelengths.

        Ensures that adding and removing spectra correctly enables and disables the use of wavelengths
        on the x-axis depending on the presence of spectral metadata.
        """
        np_impl = np.array([[[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[1.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.5  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]]])

        np_ds = self.test_model.load_dataset(np_impl)
    
        rel_path = os.path.join("..", "test_utils", "test_datasets", "caltech_4_100_150_nm")
        self.test_model.load_dataset(rel_path)

        # Get a spectrum with wavelength nm in spectrum plot
        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))
        spectrum_plot_units = self.test_model.get_spectrum_plot_x_units()
        correct_unit = u.nanometer
        self.assertTrue(spectrum_plot_units == correct_unit)

        # Ensure the plot uses wavelengths
        plot_use_wavelengths = self.test_model.get_spectrum_plot_use_wavelengths()
        self.assertTrue(True == plot_use_wavelengths)

        # Collect the spectrum
        self.test_model.collect_active_spectrum()

        # Change to view without units and click a spectrum
        self.test_model.set_main_view_rv((0, 0), np_ds.get_id())
        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        # Ensure the plot no longer uses wavelengths 
        plot_use_wavelengths = self.test_model.get_spectrum_plot_use_wavelengths()
        self.assertTrue(False == plot_use_wavelengths)

        # Get rid of the active spectra which has no wavelengths
        self.test_model.remove_active_spectrum()

        # Ensure we have wavelengths again
        plot_use_wavelengths = self.test_model.get_spectrum_plot_use_wavelengths()
        self.assertTrue(True == plot_use_wavelengths)

    def test_switch_clicked_dataset(self):
        """Tests that the spectrum plot correctly switches active dataset.

        Loads two datasets and simulates clicking in each to ensure the active spectrum reflects
        the correct dataset.
        """
        np_impl = np.array([[[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]]])
    
        
        np_impl2 = np.array([[[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[1.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.5  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]]])
        
        pixel = (0, 0)

        ds1 = self.test_model.load_dataset(np_impl)
    
        self.test_model.load_dataset(np_impl2)

        # Click on ds2 to get spectrum and ensure that that its correct
        self.test_model.click_raster_coord_main_view_rv((0, 0), pixel)

        active_spectrum_arr = self.test_model.get_active_spectrum().get_spectrum()

        self.assertTrue(np.array_equal(active_spectrum_arr, np.array([0., 1., 0.5])))

        # Click on ds2 to get spectrum and ensure that that its correct
        self.test_model.set_spectrum_plot_dataset(ds1.get_id())

        self.test_model.click_raster_coord_main_view_rv((0, 0), pixel)

        active_spectrum_arr = self.test_model.get_active_spectrum().get_spectrum()

        self.assertTrue(np.array_equal(active_spectrum_arr, np.array([0., 0., 0.])))

    def test_switch_clicked_dataset_out_of_bounds(self):
        """Tests behavior when clicking outside dataset bounds.

        Verifies that requesting a spectrum from an out-of-bounds pixel does not crash and
        returns a spectrum of NaNs.
        """
        np_impl = np.array([[[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]]])
    
        
        np_impl2 = np.array([[[0.  , 0.  , 0.  , 0.  , 0.],
                                [0.25, 0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5, 0.5 ],
                                [0.75, 0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1., 1. ]],

                            [[1.  , 0.  , 0.  , 0.  , 0.],
                                [0.25, 0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5, 0.5 ],
                                [0.75, 0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1., 1. ]],

                            [[0.5  , 0.  , 0.  , 0.  , 0.],
                                [0.25, 0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5, 0.5 ],
                                [0.75, 0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1. , 1.]]])

        ds1 = self.test_model.load_dataset(np_impl)
    
        ds2 = self.test_model.load_dataset(np_impl2)
        
        pixel = (ds2.get_width()-1, ds2.get_height()-1)

        self.test_model.set_spectrum_plot_dataset(ds1.get_id())

        # This pixel is out of bounds for ds1
        self.test_model.click_raster_coord_main_view_rv((0, 0), pixel)

        active_spectrum_arr = self.test_model.get_active_spectrum().get_spectrum()

        self.assertTrue(np.array_equal(active_spectrum_arr, np.array([np.nan, np.nan, np.nan]), equal_nan=True))


"""
Code to make sure tests work as desired. Feel free to change to your needs.
"""
if __name__ == '__main__':
        test_model = WiserTestModel(use_gui=True)
        
        # rows, cols, channels = 50, 50, 3
        # # Create a vertical gradient from 0 to 1: shape (50,1)
        # row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # # Tile the values horizontally to get a 50x50 array
        # impl = np.tile(row_values, (1, cols))
        # # Repeat the 2D array across 3 channels to get a 3x50x50 array
        # np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        rel_path = os.path.join("..", "test_utils", "test_datasets", "caltech_4_100_150_nm")
        ds = test_model.load_dataset(rel_path)

        pixel_to_click = (0, 0)

        # axes = test_model.spectrum_plot._axes
        # bbox = axes.get_window_extent()
        # x_value = bbox.x0 + bbox.width/2
        # y_value = bbox.y0 + bbox.height/2

        # mouse_event = QMouseEvent(
        #     QEvent.MouseButtonRelease,            # event type
        #     QPointF(x_value, y_value),           # local (widget) position
        #     Qt.LeftButton,                       # which button changed state
        #     Qt.MouseButtons(Qt.LeftButton),      # state of all mouse buttons
        #     Qt.NoModifier,                         # keyboard modifiers (e.g. Ctrl, Shift)
        # )

        # test_model.app.postEvent(test_model.spectrum_plot._figure_canvas, mouse_event)

        test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        test_model.click_spectrum_plot_display_toggle()
    
        spectrum_plot_units = test_model.get_spectrum_plot_x_units()
        test_model.remove_active_spectrum()

        test_model.set_spectrum_plot_dataset(ds.get_id())

        print(f"spectrum_plot_units: {spectrum_plot_units}")

        test_model.set_zoom_pane_zoom_level(9)
        test_model.scroll_zoom_pane_dx(100)
        test_model.scroll_zoom_pane_dy(100)
    
        test_model.app.exec_()

        print("We can continue on after calling app.exec_()")
