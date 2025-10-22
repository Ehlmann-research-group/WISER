"""
Test to make sure the interactive scatter plot works as intended
"""
import os
import unittest

from matplotlib import use

import tests.context
# import context

from test_utils.test_model import WiserTestModel

import numpy as np


from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import time


class TestInteractiveScatterPlot(unittest.TestCase):
    """
    Tests the interactive scatter plot.

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

    def test_scatter_plot_creation(self):
        """
        Open numpy array in WISER and create a 2D scatter plot for it, create a highlight region.

        Ensure that all of the correct points are plotted in the scatter plot and the highlight
        region is created correctly.
        """
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
        ds = self.test_model.load_dataset(np_impl)
        self.test_model.click_zoom_to_fit()
        self.test_model.open_interactive_scatter_plot_context_menu()
        self.test_model.set_interactive_scatter_x_axis_dataset(ds.get_id())
        self.test_model.set_interactive_scatter_y_axis_dataset(ds.get_id())
        self.test_model.set_interactive_scatter_render_dataset(ds.get_id())
        self.test_model.set_interactive_scatter_x_band(0)
        self.test_model.set_interactive_scatter_y_band(2)
        self.test_model.click_create_scatter_plot()
        xy = self.test_model.get_interactive_scatter_plot_xy_values()
        x_truth = np_impl[0][:, :].flatten()
        y_truth = np_impl[2][:, :].flatten()
        xy_truth = np.column_stack([x_truth, y_truth])
        self.assertTrue(np.array_equal(xy, xy_truth, equal_nan=True))

        highlighted_points_truth = [(1, 0), (1, 1), (1, 2)]
        self.test_model.create_polygon_in_interactive_scatter_plot(
            [(0.1, 0), (0.1, 0.49), (0.49, 0.49), (0.49, 0.1), (0.1, 0)]
        )
        points = self.test_model.main_view._interactive_scatter_highlight_points
        self.assertTrue(highlighted_points_truth == points)


if __name__ == "__main__":
    test_model = WiserTestModel(use_gui=True)

    np_impl = np.array(
        [
            [
                [0.0, np.nan, 0.0, 1.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5, 0.5],
                [0.75, 0.75, 0.75, 0.75],
                [1.0, 1.0, 1.0, 1.0],
            ],
            [
                [0.0, 1.0, 2.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5, 0.5],
                [0.75, 0.75, 0.75, 0.75],
                [1.0, 1.0, 1.0, 1.0],
            ],
            [
                [0.0, 2.0, 1.0, 2.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5, 0.5],
                [0.75, 0.75, np.nan, 0.75],
                [1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    ds = test_model.load_dataset(np_impl)
    test_model.click_zoom_to_fit()
    test_model.open_interactive_scatter_plot_context_menu()
    test_model.set_interactive_scatter_x_axis_dataset(ds.get_id())
    test_model.set_interactive_scatter_y_axis_dataset(ds.get_id())
    test_model.set_interactive_scatter_render_dataset(ds.get_id())
    test_model.set_interactive_scatter_x_band(0)
    test_model.set_interactive_scatter_y_band(2)
    test_model.click_create_scatter_plot()
    xy = test_model.get_interactive_scatter_plot_xy_values()
    test_model.create_polygon_in_interactive_scatter_plot(
        [(0.1, 0), (0.1, 0.49), (0.49, 0.49), (0.49, 0.1), (0.1, 0)]
    )
    points = test_model.main_view._interactive_scatter_highlight_points
    highlighted_points_truth = [(1, 0), (1, 1), (1, 2), (1, 3)]
    assert highlighted_points_truth == points
    x_truth = np_impl[0][:, :].flatten()
    y_truth = np_impl[2][:, :].flatten()
    xy_truth = np.column_stack([x_truth, y_truth])
    assert np.array_equal(xy, xy_truth, equal_nan=True)
    test_model.app.exec_()
