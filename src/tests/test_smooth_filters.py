import unittest
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

import tests.context

from test_utils.memory_cleanup import release_kept_refs
from test_utils.test_model import WiserTestModel
from wiser.utils.primitives import DeletePolicy, PriorityClass, ExternalRasterHandle
from wiser.utils.storage_client import StorageClient
from wiser.utils.task_stage_utils import (
    SmoothingFilterSpatial,
    SmoothingFilterSpectral,
    build_smoothing_exclusion_mask,
)
from wiser.utils.task_system import (
    AlgorithmPipeline,
    DatasetPlanMeta,
    ResourceModel,
    SemanticTask,
)

pytestmark = [
    pytest.mark.integration,
]

_JPL_HDR = Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_425_7_7.hdr"

_TASK_ID_COUNTER = iter(range(50000, 51000))


class TestSmoothingFilters(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_stage(self, dataset, stage, output_ref_name: str):
        """Register dataset, plan a single-stage pipeline, run it, and return (array, meta)."""
        app_services = self.test_model.app_services

        dataset_ref = app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=dataset)
        )

        # Override delete policy so output survives after the plan finishes.
        stage.set_output_delete_policy(output_ref_name, DeletePolicy.KEEP)

        task = SemanticTask(
            priority_class=PriorityClass.BACKGROUND,
            input_ref=dataset_ref,
            algorithm_pipeline=AlgorithmPipeline(stages=[stage]),
        )
        task.id = next(_TASK_ID_COUNTER)

        task_plan = app_services.task_planner.plan_semantic_task(task)
        future = app_services.scheduler.run_task_plan(task_plan)
        future.result(timeout=180)

        listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
        storage_client = StorageClient(
            service=None,  # type: ignore[arg-type]
            service_address=listener_address,
            service_authkey=listener_authkey,
        )
        output_ref = task_plan.bindings[output_ref_name]
        output_data, _ = storage_client.read_data(output_ref, filter_data=False)
        output_meta = storage_client.get_meta(output_ref)
        storage_client.close()

        release_kept_refs(app_services)
        app_services.scheduler.shutdown(wait=True)
        app_services.storage_service.close()

        return np.asarray(output_data, dtype=np.float32), output_meta

    def _make_spatial_stage(
        self, dataset, *, filter_registry_key: str, filter_kwargs: dict, output_ref_name: str
    ):
        dataset_meta_shape = (
            np.asarray(dataset.get_image_data(filter_data_ignore_value=False)).transpose(1, 2, 0).shape
        )
        input_meta = DatasetPlanMeta(shape=dataset_meta_shape, dtype=np.dtype(np.float32))
        return SmoothingFilterSpatial(
            default_executor="process",
            input_plan_meta=input_meta,
            _filter_registry_key=filter_registry_key,
            _filter_kwargs=filter_kwargs,
            _output_ref_name=output_ref_name,
        )

    def _make_spectral_stage(
        self, dataset, *, filter_registry_key: str, filter_kwargs: dict, output_ref_name: str
    ):
        dataset_meta_shape = (
            np.asarray(dataset.get_image_data(filter_data_ignore_value=False)).transpose(1, 2, 0).shape
        )
        input_meta = DatasetPlanMeta(shape=dataset_meta_shape, dtype=np.dtype(np.float32))
        return SmoothingFilterSpectral(
            default_executor="process",
            input_plan_meta=input_meta,
            _filter_registry_key=filter_registry_key,
            _filter_kwargs=filter_kwargs,
            _output_ref_name=output_ref_name,
        )

    def _expected_input(self, dataset) -> np.ndarray:
        """
        Return the dataset as float32 [y][x][b] with nodata / bad-band positions and
        any remaining non-finite values set to NaN — exactly what the tile runner does
        before passing the array to scipy.
        """
        arr = np.asarray(dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float32).transpose(
            1, 2, 0
        )

        exclusion = build_smoothing_exclusion_mask(
            arr,
            nodata=dataset.get_data_ignore_value(),
            bad_bands=dataset.get_bad_bands(),
        )
        arr = arr.copy()
        arr[exclusion] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        return arr

    def _check_meta(self, actual_meta, dataset) -> None:
        bad_bands = dataset.get_bad_bands()
        if bad_bands is None:
            self.assertIsNone(actual_meta.bad_bands)
        else:
            self.assertTrue(np.array_equal(np.asarray(actual_meta.bad_bands), np.asarray(bad_bands)))
        self.assertEqual(actual_meta.nodata, dataset.get_data_ignore_value())

    # ------------------------------------------------------------------
    # Spatial: mean (uniform_filter)
    # ------------------------------------------------------------------

    def test_spatial_mean_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        output_ref_name = "spatial_mean_jpl"
        stage = self._make_spatial_stage(
            dataset,
            filter_registry_key="uniform_filter",
            filter_kwargs={"size": 3, "mode": "reflect"},
            output_ref_name=output_ref_name,
        )
        actual, actual_meta = self._run_stage(dataset, stage, output_ref_name)

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.uniform_filter(work, size=(3, 3), axes=(0, 1), mode="reflect"),
            dtype=np.float32,
        )

        self.assertEqual(actual.shape, expected.shape)
        # Restore nodata/bad-band positions from input before comparing (mirrors task runner restore step).
        mask = ~np.isfinite(work)
        expected[mask] = work[mask]
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_meta(actual_meta, dataset)

    # ------------------------------------------------------------------
    # Spatial: median (median_filter)
    # ------------------------------------------------------------------

    def test_spatial_median_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        output_ref_name = "spatial_median_jpl"
        stage = self._make_spatial_stage(
            dataset,
            filter_registry_key="median_filter",
            filter_kwargs={"size": 3, "mode": "reflect"},
            output_ref_name=output_ref_name,
        )
        actual, actual_meta = self._run_stage(dataset, stage, output_ref_name)

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.median_filter(work, size=(3, 3), axes=(0, 1), mode="reflect"),
            dtype=np.float32,
        )

        mask = ~np.isfinite(work)
        expected[mask] = work[mask]
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_meta(actual_meta, dataset)

    # ------------------------------------------------------------------
    # Spatial: gaussian (gaussian_filter)
    # ------------------------------------------------------------------

    def test_spatial_gaussian_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        output_ref_name = "spatial_gaussian_jpl"
        stage = self._make_spatial_stage(
            dataset,
            filter_registry_key="gaussian_filter",
            filter_kwargs={"sigma": 1.0, "mode": "reflect"},
            output_ref_name=output_ref_name,
        )
        actual, actual_meta = self._run_stage(dataset, stage, output_ref_name)

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.gaussian_filter(work, sigma=(1.0, 1.0), axes=(0, 1), mode="reflect", truncate=4.0),
            dtype=np.float32,
        )

        mask = ~np.isfinite(work)
        expected[mask] = work[mask]
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_meta(actual_meta, dataset)

    # ------------------------------------------------------------------
    # Spectral: mean (uniform_filter)
    # ------------------------------------------------------------------

    def test_spectral_mean_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        output_ref_name = "spectral_mean_jpl"
        stage = self._make_spectral_stage(
            dataset,
            filter_registry_key="uniform_filter",
            filter_kwargs={"size": 5, "mode": "reflect"},
            output_ref_name=output_ref_name,
        )
        actual, actual_meta = self._run_stage(dataset, stage, output_ref_name)

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.uniform_filter(work, size=5, axes=(2,), mode="reflect"),
            dtype=np.float32,
        )

        mask = ~np.isfinite(work)
        expected[mask] = work[mask]
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_meta(actual_meta, dataset)

    # ------------------------------------------------------------------
    # Spectral: median (median_filter)
    # ------------------------------------------------------------------

    def test_spectral_median_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        output_ref_name = "spectral_median_jpl"
        stage = self._make_spectral_stage(
            dataset,
            filter_registry_key="median_filter",
            filter_kwargs={"size": 5, "mode": "reflect"},
            output_ref_name=output_ref_name,
        )
        actual, actual_meta = self._run_stage(dataset, stage, output_ref_name)

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.median_filter(work, size=5, axes=(2,), mode="reflect"),
            dtype=np.float32,
        )

        mask = ~np.isfinite(work)
        expected[mask] = work[mask]
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_meta(actual_meta, dataset)

    # ------------------------------------------------------------------
    # Spectral: gaussian (gaussian_filter)
    # ------------------------------------------------------------------

    def test_spectral_gaussian_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        output_ref_name = "spectral_gaussian_jpl"
        stage = self._make_spectral_stage(
            dataset,
            filter_registry_key="gaussian_filter",
            filter_kwargs={"sigma": 2.0, "mode": "reflect"},
            output_ref_name=output_ref_name,
        )
        actual, actual_meta = self._run_stage(dataset, stage, output_ref_name)

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.gaussian_filter(work, sigma=2.0, axes=(2,), mode="reflect", truncate=4.0),
            dtype=np.float32,
        )

        mask = ~np.isfinite(work)
        expected[mask] = work[mask]
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_meta(actual_meta, dataset)


if __name__ == "__main__":
    unittest.main()
