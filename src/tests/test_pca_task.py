from typing import Union
import unittest
from pathlib import Path

import tests.context

import numpy as np

from test_utils.memory_cleanup import release_kept_refs
from test_utils.test_model import WiserTestModel

from wiser.gui.permanent_plugins.pca_plugin import (
    ESTIMATOR_TYPES,
    PCAPlugin,
    PCAPluginTask,
    compute_max_pca_components,
)
from wiser.raster.loader import RasterDataLoader
from wiser.raster.utils import compute_PCA_on_image
from wiser.utils.primitives import DeletePolicy, PriorityClass
from wiser.utils.storage_client import StorageClient
from wiser.utils.primitives import ExternalRasterHandle
from wiser.utils.task_stage_utils import get_pca_pipeline
from wiser.utils.task_system import SemanticTask

from tests.utils import (
    NAN_INF_BAD_BANDS,
    NAN_INF_DATA_IGNORE_VALUE,
    assert_reduction_drops_invalid_pixels,
    build_unmasked_nan_inf_cube,
)


class TestPcaTask(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _assert_componentwise_sign_invariant_match(
        self,
        actual: Union[np.ndarray, np.ma.MaskedArray],
        expected: Union[np.ndarray, np.ma.MaskedArray],
        atol: float = 1e-5,
    ) -> None:
        actual_ma = np.ma.array(actual, copy=False)
        expected_ma = np.ma.array(expected, copy=False)

        self.assertEqual(actual_ma.shape, expected_ma.shape)
        self.assertTrue(np.array_equal(np.ma.getmaskarray(actual_ma), np.ma.getmaskarray(expected_ma)))

        actual_mask = np.ma.getmaskarray(actual_ma)
        expected_mask = np.ma.getmaskarray(expected_ma)
        combined_mask = actual_mask | expected_mask

        for component_idx in range(actual_ma.shape[2]):
            actual_component = np.asarray(np.ma.getdata(actual_ma[:, :, component_idx]), dtype=np.float64)
            expected_component = np.asarray(np.ma.getdata(expected_ma[:, :, component_idx]), dtype=np.float64)
            valid = ~combined_mask[:, :, component_idx]
            self.assertTrue(np.any(valid), f"Expected valid pixels for component {component_idx}")

            actual_valid = actual_component[valid]
            expected_valid = expected_component[valid]
            same_sign = np.allclose(actual_valid, expected_valid, atol=atol)
            flipped_sign = np.allclose(actual_valid, -expected_valid, atol=atol)
            self.assertTrue(
                same_sign or flipped_sign,
                f"PCA component {component_idx} does not match expected values even with sign flip",
            )

    def test_get_pca_pipeline_matches_compute_pca_on_image_on_data_ignore_fixture(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent
            / ".."
            / "test_utils"
            / "test_datasets"
            / "caltech_425_6_6_data_ignore.hdr"
        ).resolve()
        dataset = self.test_model.load_dataset(str(dataset_path))
        app_services = self.test_model.app_services
        storage_client = None
        try:
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            output_ref_name = "pca_data_ignore_output"
            pca_json_ref_name = "pca_data_ignore_model"
            num_components = 3
            pca_pipeline = get_pca_pipeline(
                dataset_ref=dataset_ref,
                num_components=num_components,
                output_ref_name=output_ref_name,
                pca_json_ref_name=pca_json_ref_name,
            )
            pca_pipeline.stages[-1].set_output_delete_policy(output_ref_name, DeletePolicy.KEEP)
            pca_pipeline.stages[-1].set_output_delete_policy(pca_json_ref_name, DeletePolicy.KEEP)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=pca_pipeline,
            )
            task.id = 3001

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=180)

            expected_array, expected_pca = compute_PCA_on_image(
                image_arr=dataset.get_image_data(),
                num_components=num_components,
                bad_bands=dataset.get_bad_bands(),
                data_ignore=dataset.get_data_ignore_value(),
            )

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_ref = task_plan.bindings[output_ref_name]
            actual_array, actual_meta = storage_client.read_data(output_ref)
            pca_payload = storage_client.read_json_value(task_plan.bindings[pca_json_ref_name])

            self._assert_componentwise_sign_invariant_match(actual_array, expected_array)
            self.assertEqual(actual_meta.nodata, dataset.get_data_ignore_value())
            self.assertIsNone(actual_meta.bad_bands)
            self.assertIn("pca", pca_payload)
            self.assertEqual(pca_payload["pca"].n_components_, expected_pca.n_components_)
            self.assertEqual(pca_payload["pca"].components_.shape, expected_pca.components_.shape)
            self.assertTrue(
                np.allclose(
                    np.abs(pca_payload["pca"].components_), np.abs(expected_pca.components_), atol=1e-3
                )
            )
            self.assertTrue(
                np.array_equal(np.ma.getmaskarray(actual_array), np.ma.getmaskarray(expected_array))
            )
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_pca_plugin_task_completion_loads_dataset_and_metadata(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent
            / ".."
            / "test_utils"
            / "test_datasets"
            / "caltech_425_6_6_data_ignore.hdr"
        ).resolve()
        dataset = self.test_model.load_dataset(str(dataset_path))
        app_services = self.test_model.app_services
        try:
            starting_dataset_count = self.test_model.app_state.num_datasets()
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            task = PCAPluginTask(
                app_state=self.test_model.app_state,
                source_dataset=dataset,
                input_ref=dataset_ref,
                num_components=3,
                max_components_available=compute_max_pca_components(dataset),
            )

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=180)
            self.test_model.app.processEvents()

            self.assertEqual(self.test_model.app_state.num_datasets(), starting_dataset_count + 1)
            new_dataset = self.test_model.app_state.get_datasets()[-1]
            self.assertEqual(new_dataset.get_data_ignore_value(), dataset.get_data_ignore_value())
            self.assertEqual(new_dataset.get_spatial_metadata(), dataset.get_spatial_metadata())
            self.assertEqual(new_dataset.get_description(), dataset.get_description())
            self.assertTrue(hasattr(task, "_pca_widget"))
            self.assertIn("PCA Metadata", task._pca_widget._text_edit.toPlainText())
        finally:
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_plugin_run_pca_submits_task_path(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent
            / ".."
            / "test_utils"
            / "test_datasets"
            / "caltech_425_6_6_data_ignore.hdr"
        ).resolve()
        dataset = self.test_model.load_dataset(str(dataset_path))
        plugin = PCAPlugin()
        app_services = self.test_model.app_services
        try:
            future = plugin.run_pca(
                dataset=dataset,
                num_components=3,
                estimator=ESTIMATOR_TYPES.COVARIANCE,
                app_state=self.test_model.app_state,
                app_services=app_services,
            )
            self.assertIsNotNone(future)
            future.result(timeout=180)
            self.test_model.app.processEvents()
            self.assertTrue(hasattr(plugin, "_last_pca_task"))
            self.assertTrue(hasattr(plugin._last_pca_task, "_pca_widget"))
        finally:
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_pca_pipeline_handles_unmasked_nan_and_inf(self) -> None:
        # Synthetic cube with a bad band, a nodata sentinel, and unmasked
        # NaN/+Inf/-Inf in good bands. Before the shared finite_unmasked_row_mask
        # cleaning fix, the PCA pipeline fed those into sklearn and raised
        # "Input X contains NaN".
        dataset = RasterDataLoader().dataset_from_numpy_array(build_unmasked_nan_inf_cube())
        dataset.set_bad_bands(NAN_INF_BAD_BANDS)
        dataset.set_data_ignore_value(NAN_INF_DATA_IGNORE_VALUE)

        app_services = self.test_model.app_services
        storage_client = None
        try:
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            num_components = 3
            output_ref_name = "pca_nan_output"
            pca_json_ref_name = "pca_nan_model"
            pca_pipeline = get_pca_pipeline(
                dataset_ref=dataset_ref,
                num_components=num_components,
                output_ref_name=output_ref_name,
                pca_json_ref_name=pca_json_ref_name,
            )
            pca_pipeline.stages[-1].set_output_delete_policy(output_ref_name, DeletePolicy.KEEP)
            pca_pipeline.stages[-1].set_output_delete_policy(pca_json_ref_name, DeletePolicy.KEEP)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=pca_pipeline,
            )
            task.id = 3002

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=180)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output, _ = storage_client.read_data(task_plan.bindings[output_ref_name])
            assert_reduction_drops_invalid_pixels(self, output, num_components)
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()
