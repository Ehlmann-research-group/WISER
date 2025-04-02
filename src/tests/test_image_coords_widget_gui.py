import os

import unittest

import tests.context
# import context

import numpy as np

from test_utils.test_model import WiserTestModel

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class TestImageCoordsWidget(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_go_to_pixel(self):
        # Get the directory where the current file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Compute the absolute path to the target file
        target_path = os.path.normpath( \
            os.path.join(current_dir, "..", "test_utils", "test_datasets", "caltech_4_100_150_nm.hdr"))

        self.test_model.load_dataset(target_path)

        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        self.test_model.scroll_main_view_rv((0, 0), -100, -100)

        self.test_model.goto_pixel_using_geo_coords_dialog((100, 100))

        self.test_model.cancel_geo_coords_dialog()

        active_spectrum = self.test_model.get_active_spectrum()

        expected_spectrum = np.array([0.13855411, 0.16071412, 0.18646793, 0.18314502])

        self.assertTrue(np.allclose(active_spectrum.get_spectrum(), expected_spectrum))


if __name__ == '__main__':
    test_model = WiserTestModel(use_gui=True)

    # Get the directory where the current file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Compute the absolute path to the target file
    target_path = os.path.normpath( \
        os.path.join(current_dir, "..", "test_utils", "test_datasets", "caltech_4_100_150_nm.hdr"))

    test_model.load_dataset(target_path)

    test_model.set_main_view_zoom_level(10)

    test_model.scroll_main_view_rv((0, 0), -100, -100)

    test_model.goto_pixel_using_geo_coords_dialog((100, 100))

    test_model.cancel_geo_coords_dialog()

    active_spectrum = test_model.get_active_spectrum()

    expected_spectrum = np.array([0.13855411, 0.16071412, 0.18646793, 0.18314502])

    test_model.app.exec_()


