import unittest

import tests.context

import numpy as np

from test_utils.test_model import WiserTestModel
from test_utils.test_arrays import (
    clean_test_arr,
    gt_clean_test_arr,
    clean_test_arr_extra_2_bands,
    test_arr_extra_2_bands_data_ignore,
)

from wiser.gui.permanent_plugins.pca_plugin import PCAPlugin, ESTIMATOR_TYPES

from sklearn.decomposition import PCA


class TestPCA(unittest.TestCase):
    def setUp(self):
        """Initializes the test model before each test."""
        self.test_model = WiserTestModel()

    def tearDown(self):
        """Closes the test model and cleans up resources after each test."""
        self.test_model.close_app()
        del self.test_model

    # def test_clean_dataset(self):
    #     """
    #     Tests that PCA works on a clean dataset (no bad bands, no data_ignore)
    #     """
    #     num_components = 8
    #     print(f"shape of gt_clean_test_arr: {gt_clean_test_arr.shape}")
    #     print(f"shape of clean_test_arr: {clean_test_arr.shape}")
    #     pca = PCA(n_components=num_components)
    #     pca_gt = pca.fit_transform(gt_clean_test_arr)

    #     pca_plugin = PCAPlugin()

    #     clean_test_arr_shaped = clean_test_arr.copy().transpose(2, 0, 1)
    #     print(f"shape of clean_test_arr_shaped: {clean_test_arr_shaped.shape}")
    #     dataset = self.test_model.load_dataset(clean_test_arr_shaped)
    #     pca_test_arr = pca_plugin.run_pca(
    #         dataset=dataset,
    #         num_components=num_components,
    #         estimator=ESTIMATOR_TYPES.COVARIANCE,
    #         app_state=self.test_model.app_state,
    #         test_mode=True,
    #     )

    #     pca_test_arr = pca_test_arr.transpose(1, 2, 0)
    #     pca_test_arr = pca_test_arr.reshape(
    #         (pca_test_arr.shape[0] * pca_test_arr.shape[1], pca_test_arr.shape[2]),
    #     )

    #     print(f"shape of pca_gt: {pca_gt.shape}")
    #     print(f"shape of pca_test_arr: {pca_test_arr.shape}")
    #     print(f"are close?: {np.allclose(pca_gt, pca_test_arr)}")
    #     self.assertTrue(np.allclose(pca_gt, pca_test_arr))

    # def test_dataset_with_bad_bands(self):
    #     num_components = 8
    #     print(f"shape of gt_clean_test_arr: {gt_clean_test_arr.shape}")
    #     print(f"shape of clean_test_arr_extra_2_bands: {clean_test_arr_extra_2_bands.shape}")
    #     pca = PCA(n_components=num_components)
    #     pca_gt = pca.fit_transform(gt_clean_test_arr)

    #     pca_plugin = PCAPlugin()

    #     clean_test_arr_shaped = clean_test_arr_extra_2_bands.copy().transpose(2, 0, 1)
    #     print(f"shape of clean_test_arr_shaped: {clean_test_arr_shaped.shape}")
    #     dataset = self.test_model.load_dataset(clean_test_arr_shaped)
    #     bad_bands = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    #     dataset.set_bad_bands(bad_bands)
    #     pca_test_arr = pca_plugin.run_pca(
    #         dataset=dataset,
    #         num_components=num_components,
    #         estimator=ESTIMATOR_TYPES.COVARIANCE,
    #         app_state=self.test_model.app_state,
    #         test_mode=True,
    #     )

    #     pca_test_arr = pca_test_arr.transpose(1, 2, 0)
    #     pca_test_arr = pca_test_arr.reshape(
    #         (pca_test_arr.shape[0] * pca_test_arr.shape[1], pca_test_arr.shape[2]),
    #     )

    #     print(f"shape of pca_gt: {pca_gt.shape}")
    #     print(f"shape of pca_test_arr: {pca_test_arr.shape}")
    #     print(f"are close?: {np.allclose(pca_gt, pca_test_arr)}")
    #     self.assertTrue(np.allclose(pca_gt, pca_test_arr))

    def test_dataset_bad_bands_data_ignore(self):
        num_components = 8
        print(f"shape of gt_clean_test_arr: {gt_clean_test_arr.shape}")
        print(f"shape of test_arr_extra_2_bands_data_ignore: {test_arr_extra_2_bands_data_ignore.shape}")
        pca = PCA(n_components=num_components)
        pca_gt = pca.fit_transform(gt_clean_test_arr)

        pca_plugin = PCAPlugin()

        clean_test_arr_shaped = test_arr_extra_2_bands_data_ignore.copy().transpose(2, 0, 1)
        print(f"shape of clean_test_arr_shaped: {clean_test_arr_shaped.shape}")
        dataset = self.test_model.load_dataset(clean_test_arr_shaped)
        bad_bands = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        dataset.set_bad_bands(bad_bands)
        dataset.set_data_ignore_value(-9999)
        print(f"type of dataset arr !@#: {type(dataset.get_image_data())}")
        print(f"dtype of dataset: {dataset.get_image_data().dtype}")
        pca_test_arr = pca_plugin.run_pca(
            dataset=dataset,
            num_components=num_components,
            estimator=ESTIMATOR_TYPES.COVARIANCE,
            app_state=self.test_model.app_state,
            test_mode=True,
        )

        pca_test_arr[pca_test_arr.mask]

        # [num_components][y][x] --> [y][x][num_components]
        pca_test_arr = pca_test_arr.transpose(1, 2, 0)
        # pca_test_arr = pca_test_arr[~pca_test_arr.mask].flatten()
        pca_test_arr = pca_test_arr.reshape(
            (pca_test_arr.shape[0] * pca_test_arr.shape[1], pca_test_arr.shape[2]),
        )
        pca_test_arr = pca_test_arr[~pca_test_arr.mask]
        gt_arr = pca_gt.flatten()

        print(f"pca_gt: {pca_gt.flatten()}")
        print(f"pca_test_arr: {pca_test_arr}")
        print(f"pca_test_arr type: {pca_test_arr.dtype}")

        print(f"shape of gt_arr: {pca_gt.shape}")
        print(f"shape of pca_test_arr: {pca_test_arr.shape}")
        print(f"are close?: {np.allclose(gt_arr, pca_test_arr)}")
        self.assertTrue(np.allclose(gt_arr, pca_test_arr))
