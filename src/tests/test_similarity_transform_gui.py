import unittest

import os

# import context

from test_utils.test_model import WiserTestModel


import numpy as np

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class TestSimliarityTransformGUI(unittest.TestCase):
    """
    Tests the GeoReferencer by going through the GUI.

    Waht it doesn't test:
    1. Doesn't test filtering logic of choosing a file name
    """

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # Write one tests to rotate and scale the caltech dataset by 30 degrees. Get the geo transform and compare
    # Get the array and compare. Get this from
    def test_rotate_scale(self):
        load_path = os.path.join(
            "..", "test_utils", "test_datasets", "caltech_4_100_150_nm"
        )
        ground_truth_path = os.path.join(
            "..",
            "test_utils",
            "test_datasets",
            "caltech_4_100_150_nm_rot_35_scale_2_linear_gt.tif",
        )

        temp_save_path = os.path.join(
            "..",
            "test_utils",
            "test_datasets",
            "artifacts",
            "caltech_4_100_150_nm_rot_35_scale_2_linear.tif",
        )
        ds = self.test_model.load_dataset(load_path)

        self.test_model.open_similarity_transform_dialog()

        self.test_model.switch_sim_tab(to_translate=False)

        self.test_model.select_dataset_rs(ds)
        self.test_model.set_rotation_slider(35)

        self.test_model.set_scale_rs(2)

        self.test_model.choose_interpolation_rs(2)

        self.test_model.set_save_path_rs(temp_save_path)

        self.test_model.run_rotate_scale()

        self.test_model.close_similarity_transform_dialog()

        gt_ds = self.test_model.load_dataset(ground_truth_path)
        test_ds = self.test_model.load_dataset(temp_save_path)

        # Don't use get_image_data here. For some reason the reference to the
        # array gets unreferenced in the github actions linux server, so you
        # must directrly make a copy.
        gt_arr = gt_ds.get_impl().gdal_dataset.ReadAsArray().copy()
        gt_geo_transform = gt_ds.get_geo_transform()

        test_arr = test_ds.get_impl().gdal_dataset.ReadAsArray().copy()
        test_geo_transform = test_ds.get_geo_transform()

        self.assertTrue(
            np.allclose(gt_arr, test_arr),
            "Rotated and scaled array doesn't match ground truth",
        )
        self.assertTrue(
            gt_geo_transform == test_geo_transform,
            "Rotated and scaled geo transform doesn't match ground truth",
        )

    def test_translate(self):
        load_path = os.path.join(
            "..", "test_utils", "test_datasets", "caltech_4_100_150_nm"
        )
        temp_save_path = os.path.join(
            "..",
            "test_utils",
            "test_datasets",
            "artifacts",
            "caltech_4_100_150_nm_translate.tif",
        )
        ds = self.test_model.load_dataset(load_path)

        self.test_model.open_similarity_transform_dialog()

        self.test_model.switch_sim_tab(to_translate=True)

        self.test_model.select_dataset_translate(ds)
        lat_translate_amt = 100
        lon_translate_amt = 200
        self.test_model.set_translate_lat(lat_translate_amt)
        self.test_model.set_translate_lon(lon_translate_amt)

        self.test_model.set_save_path_translate(temp_save_path)

        self.test_model.run_create_translation()

        self.test_model.close_similarity_transform_dialog()

        ds_translate = self.test_model.load_dataset(temp_save_path)

        orig_gt = ds.get_geo_transform()
        ground_truth_gt = (
            orig_gt[0] + lon_translate_amt,
            orig_gt[1],
            orig_gt[2],
            orig_gt[3] + lat_translate_amt,
            orig_gt[4],
            orig_gt[5],
        )
        translated_gt = ds_translate.get_geo_transform()
        self.assertTrue(ground_truth_gt == translated_gt)


if __name__ == "__main__":
    test_model = WiserTestModel(use_gui=True)

    rel_path = os.path.join("..", "test_utils", "test_datasets", "caltech_4_100_150_nm")
    ds = test_model.load_dataset(rel_path)

    test_model.open_similarity_transform_dialog()

    test_model.switch_sim_tab(to_translate=False)

    test_model.select_dataset_rs(ds)
    test_model.set_rotation_slider(35)

    test_model.set_scale_rs(2)

    test_model.choose_interpolation_rs(2)

    save_path = os.path.join(
        "..",
        "test_utils",
        "test_datasets",
        "caltech_4_100_150_nm_rot_35_scale_2_linear_gt.tif",
    )
    # save_path = os.path.join("..", "test_utils", "test_datasets", "artifacts", "test_sim_transform.tif")
    test_model.set_save_path_rs(save_path)

    test_model.run_rotate_scale()

    # test_model.close_similarity_transform_dialog()
    # test_model.switch_sim_tab(to_translate=True)
    # test_model.select_dataset_translate(ds)

    # test_model.click_translation_pixel((12, 12))

    # test_model.set_translate_lat(100)

    # test_model.set_translate_lon(100)

    # save_path = os.path.join("..", "test_utils", "test_datasets", "artifacts", "test_sim_transform_translate.tif")
    # test_model.set_save_path_translate(save_path)

    # print(f"new spatial_coords: {test_model.ge_spatial_coords_translate_pane()}")
    # print(f"old spatial coords: ")
    # print(f"get_lat_north_ul_text: {test_model.get_lat_north_ul_text()}")
    # print(f"get_lon_east_ul_text: {test_model.get_lon_east_ul_text()}")

    # test_model.click_translation_pixel((100, 100))

    # print(f"spatial_coords: {test_model.ge_spatial_coords_translate_pane()}")
    # print(f"get_lat_north_ul_text: {test_model.get_lat_north_ul_text()}")
    # print(f"get_lon_east_ul_text: {test_model.get_lon_east_ul_text()}")

    # test_model.run_create_translation()

    test_model.app.exec_()
