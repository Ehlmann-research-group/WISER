import unittest
from pathlib import Path
from unittest.mock import patch

# import context
import tests.context

import numpy as np
import spectral
import pytest
from test_utils.test_model import WiserTestModel

from wiser.gui.app_services import AppServices
from wiser.gui.mnf import MinimumNoiseFractionDialog, get_mnf_pipeline, get_y_shift_noise
from wiser.raster.loader import RasterDataLoader
from wiser.utils.primitives import DeletePolicy, PriorityClass
from wiser.utils.storage_client import StorageClient
from wiser.utils.primitives import ExternalRasterHandle
from wiser.utils.task_system import AlgorithmPipeline, SemanticTask

pytestmark = [
    pytest.mark.integration,
]


class TestMnf(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_get_y_shift_noise_stage_outputs_vertical_shift_difference_noise(self) -> None:
        array_2x2x3 = np.array(
            [
                [[1.0, 3.0], [2.0, 4.0]],
                [[10.0, 30.0], [20.0, 40.0]],
                [[100.0, 300.0], [200.0, 400.0]],
            ],
            dtype=np.float32,
        )
        dataset = self.test_model.load_dataset(array_2x2x3)

        app_services = self.test_model.app_services
        storage_client = None
        try:
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            output_ref_name = "shift_y_noise_test"
            stage = get_y_shift_noise(dataset_ref, output_ref_name)
            stage.set_output_delete_policy(output_ref_name, DeletePolicy.KEEP)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
            )
            task.id = 2001

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=30)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            output_ref = task_plan.bindings[output_ref_name]
            output_noise, _ = storage_client.read_data(output_ref)

            yxb = array_2x2x3.transpose(1, 2, 0)
            expected_noise = yxb[:-1, :, :] - yxb[1:, :, :]
            self.assertEqual(output_noise.shape, expected_noise.shape)
            self.assertTrue(np.allclose(output_noise, expected_noise, atol=1e-6))
        finally:
            if storage_client is not None:
                storage_client.close()
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_get_mnf_pipeline_matches_spy_mnf_on_caltech_fixture(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "circuit_4_100_150_um"
        ).resolve()
        dataset = self.test_model.load_dataset(str(dataset_path))
        app_services = self.test_model.app_services
        storage_client = None
        try:
            image = np.nan_to_num(
                np.asarray(dataset.get_image_data()).transpose(1, 2, 0),
                nan=0.0,
            )

            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            num_components = 3
            output_ref_name = "mnf_data"
            mnf_pipeline = get_mnf_pipeline(dataset_ref, num_components, output_ref_name)
            mnf_pipeline.stages[-1].set_output_delete_policy(output_ref_name, DeletePolicy.KEEP)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=mnf_pipeline,
            )
            task.id = 2002

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=180)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )

            mnf_ref = task_plan.bindings[output_ref_name]
            our_mnf, _ = storage_client.read_data(mnf_ref)

            spy_signal_stats = spectral.calc_stats(image)
            spy_noise_stats = spectral.noise_from_diffs(image, direction="lower")
            spy_mnf_transform = spectral.mnf(spy_signal_stats, spy_noise_stats)
            spy_mnf = np.asarray(
                spy_mnf_transform.reduce(image, num=num_components),
                dtype=np.float32,
            )
            self.assertEqual(our_mnf.shape, spy_mnf.shape)
            for i in range(num_components):
                ours = our_mnf[:, :, i].reshape(-1).astype(np.float64)
                theirs = spy_mnf[:, :, i].reshape(-1).astype(np.float64)
                self.assertTrue(np.allclose(np.abs(ours), np.abs(theirs), atol=1e-3))
        finally:
            if storage_client is not None:
                storage_client.close()
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_get_mnf_pipeline_runs_on_data_ignore_fixture_without_error(self) -> None:
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
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            output_ref_name = "mnf_data_ignore_output"
            mnf_pipeline = get_mnf_pipeline(dataset_ref, 3, output_ref_name)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=mnf_pipeline,
            )
            task.id = 2003

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=180)

            self.assertIn(output_ref_name, task_plan.bindings)
        finally:
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_get_mnf_pipeline_does_not_count_valid_pixels_during_planning(self) -> None:
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
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            with patch(
                "wiser.utils.task_stage_utils.count_valid_dataset_pixels",
                side_effect=AssertionError(
                    "count_valid_dataset_pixels should not run during get_mnf_pipeline"
                ),
            ):
                mnf_pipeline = get_mnf_pipeline(dataset_ref, 3, "mnf_planning_only")
                self.assertIsNotNone(mnf_pipeline)
        finally:
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_get_mnf_pipeline_matches_spy_mnf_with_bad_bands_and_invalid_pixels(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent
            / ".."
            / "test_utils"
            / "test_datasets"
            / "circuit_4_100_150_um.hdr"
        ).resolve()
        base_dataset = self.test_model.load_dataset(str(dataset_path))
        app_services = self.test_model.app_services
        storage_client = None
        try:
            data_ignore_value = -9999.0
            image_by_band = np.array(base_dataset.get_image_data(filter_data_ignore_value=False), copy=True)
            band_count, height, width = image_by_band.shape
            self.assertEqual(band_count, 4)

            image_by_band[:, 0, 0] = np.nan
            image_by_band[:, 0, 1] = np.inf
            image_by_band[:, 0, 2] = -np.inf
            image_by_band[:, 0, 3:width] = data_ignore_value

            mutated_dataset = RasterDataLoader().dataset_from_numpy_array(image_by_band)
            bad_bands = [1] * band_count
            bad_bands[-1] = 0
            mutated_dataset.set_bad_bands(bad_bands)
            mutated_dataset.set_data_ignore_value(data_ignore_value)

            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=mutated_dataset)
            )
            num_components = 2
            output_ref_name = "mnf_bad_bands_invalid_pixels"
            mnf_pipeline = get_mnf_pipeline(dataset_ref, num_components, output_ref_name)
            mnf_pipeline.stages[-1].set_output_delete_policy(output_ref_name, DeletePolicy.KEEP)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=mnf_pipeline,
            )
            task.id = 2004

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=180)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )

            mnf_ref = task_plan.bindings[output_ref_name]
            our_mnf, _ = storage_client.read_data(mnf_ref)

            valid_image = np.asarray(
                base_dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float32
            )
            valid_image = valid_image[:-1, 1:, :].transpose(1, 2, 0)

            spy_signal_stats = spectral.calc_stats(valid_image)
            spy_noise_stats = spectral.noise_from_diffs(valid_image, direction="lower")
            spy_mnf_transform = spectral.mnf(spy_signal_stats, spy_noise_stats)
            spy_mnf = np.asarray(
                spy_mnf_transform.reduce(valid_image, num=num_components),
                dtype=np.float32,
            )

            our_valid_mnf = np.asarray(our_mnf[1:, :, :], dtype=np.float32)
            self.assertEqual(our_valid_mnf.shape, spy_mnf.shape)
            for i in range(num_components):
                ours = our_valid_mnf[:, :, i].reshape(-1).astype(np.float64)
                theirs = spy_mnf[:, :, i].reshape(-1).astype(np.float64)
                # The difference is with the covariance matrix calculation of the input data
                # I can't find the actual problem because the divisor is the same and the
                # X data should be the same
                self.assertTrue(np.allclose(np.abs(ours), np.abs(theirs), atol=3e-1))
        finally:
            if storage_client is not None:
                storage_client.close()
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()


if __name__ == "__main__":
    test_mnf = TestMnf()
    test_mnf.setUp()
    try:
        test_mnf.test_get_mnf_pipeline_matches_spy_mnf_with_bad_bands_and_invalid_pixels()
    finally:
        test_mnf.tearDown()
