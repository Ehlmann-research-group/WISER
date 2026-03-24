import os
import unittest

import tests.context
import numpy as np

from test_utils.test_model import WiserTestModel
from wiser.gui.permanent_plugins.continuum_removal_plugin import (
    ContinuumRemovalPlugin,
)
from wiser.raster.dataset import dict_list_equal
from wiser.utils.primitives import PriorityClass
from wiser.utils.storage_client import StorageClient
from wiser.utils.storage_layer import ExternalRasterHandle
from wiser.utils.task_stage_utils import get_continuum_removal_image_pipeline
from wiser.utils.task_system import SemanticTask


class TestContinuumRemovalTask(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _dataset_path(self, name: str) -> str:
        return os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            name,
        )

    def _compare_datasets(self, actual, expected):
        self.assertTrue(np.allclose(actual.get_image_data(), expected.get_image_data(), equal_nan=True))
        if actual.get_spatial_ref() is None or expected.get_spatial_ref() is None:
            self.assertEqual(actual.get_spatial_ref(), expected.get_spatial_ref())
        else:
            self.assertTrue(actual.get_spatial_ref().IsSame(expected.get_spatial_ref()))
        self.assertEqual(actual.get_geo_transform(), expected.get_geo_transform())
        self.assertEqual(actual.has_wavelengths(), expected.has_wavelengths())
        self.assertEqual(actual._data_ignore_value, expected._data_ignore_value)
        self.assertEqual(actual._default_display_bands, expected._default_display_bands)
        self.assertEqual(actual.get_bad_bands(), expected.get_bad_bands())
        self.assertTrue(
            dict_list_equal(
                actual._band_info,
                expected._band_info,
                ignore_keys=["wavelength_units"],
            )
        )

    def test_continuum_removal_pipeline_matches_direct_numba_subset(self) -> None:
        dataset = self.test_model.load_dataset(self._dataset_path("caltech_425_7_7_nm"))
        plugin = ContinuumRemovalPlugin()
        app_services = self.test_model.app_services
        storage_client = None
        try:
            min_cols = 2
            min_rows = 2
            max_cols = dataset.get_width() - 2
            max_rows = dataset.get_height() - 1
            min_band = 100
            max_band = dataset.num_bands() - 100

            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            output_ref_name = "continuum_removed_subset"
            pipeline = get_continuum_removal_image_pipeline(
                dataset_ref=dataset_ref,
                min_cols=min_cols,
                min_rows=min_rows,
                max_cols=max_cols,
                max_rows=max_rows,
                min_band=min_band,
                max_band=max_band,
                output_ref_name=output_ref_name,
            )
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=pipeline,
            )
            task.id = 4001

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=180)

            expected_dataset = plugin.image(
                min_cols,
                min_rows,
                max_cols,
                max_rows,
                min_band,
                max_band,
                context={"wiser": self.test_model.app_state, "dataset": dataset},
                in_test_mode=True,
            )
            expected = expected_dataset.get_image_data().transpose(1, 2, 0)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            actual_ref = task_plan.bindings[output_ref_name]
            actual, meta = storage_client.read_data(actual_ref, filter_data=False)

            self.assertTrue(np.allclose(actual, expected, equal_nan=True))
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(meta.nodata, dataset.get_data_ignore_value())
        finally:
            if storage_client is not None:
                storage_client.close()
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_continuum_removal_task_matches_old_inline_whole_image(self) -> None:
        dataset = self.test_model.load_dataset(self._dataset_path("caltech_4_100_150_nm"))
        plugin = ContinuumRemovalPlugin()
        app_services = self.test_model.app_services
        try:
            direct_context = {"wiser": self.test_model.app_state, "dataset": dataset}
            expected_dataset = plugin.image(
                min_cols=0,
                min_rows=0,
                max_cols=dataset.get_width(),
                max_rows=dataset.get_height(),
                min_band=0,
                max_band=dataset.num_bands(),
                context=direct_context,
                in_test_mode=True,
            )  # Dataset from the old synchronous method

            future = plugin.image(
                min_cols=0,
                min_rows=0,
                max_cols=dataset.get_width(),
                max_rows=dataset.get_height(),
                min_band=0,
                max_band=dataset.num_bands(),
                context={
                    "wiser": self.test_model.app_state,
                    "dataset": dataset,
                    "app_services": app_services,
                },
            )
            future.result(timeout=180)
            self.test_model.app.processEvents()
            actual_dataset = self.test_model.app_state.get_datasets()[-1]  # Dataset from the future

            self._compare_datasets(actual_dataset, expected_dataset)
        finally:
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_continuum_removal_task_matches_old_inline_subset(self) -> None:
        dataset = self.test_model.load_dataset(self._dataset_path("caltech_425_7_7_nm"))
        plugin = ContinuumRemovalPlugin()
        app_services = self.test_model.app_services
        try:
            min_cols = 2
            min_rows = 2
            max_cols = dataset.get_width() - 2
            max_rows = dataset.get_height() - 1
            min_band = 100
            max_band = dataset.num_bands() - 100

            expected_dataset = plugin.image(
                min_cols=min_cols,
                min_rows=min_rows,
                max_cols=max_cols,
                max_rows=max_rows,
                min_band=min_band,
                max_band=max_band,
                context={"wiser": self.test_model.app_state, "dataset": dataset},
                in_test_mode=True,
            )

            future = plugin.image(
                min_cols=min_cols,
                min_rows=min_rows,
                max_cols=max_cols,
                max_rows=max_rows,
                min_band=min_band,
                max_band=max_band,
                context={
                    "wiser": self.test_model.app_state,
                    "dataset": dataset,
                    "app_services": app_services,
                },
            )
            future.result(timeout=180)
            self.test_model.app.processEvents()
            actual_dataset = self.test_model.app_state.get_datasets()[-1]

            self._compare_datasets(actual_dataset, expected_dataset)
        finally:
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()
