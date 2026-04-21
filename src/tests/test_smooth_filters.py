import unittest
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

import tests.context

from test_utils.memory_cleanup import release_kept_refs
from test_utils.test_model import WiserTestModel
from wiser.raster.dataset import RasterDataSet
from wiser.utils.primitives import DeletePolicy, PriorityClass, ExternalRasterHandle
from wiser.utils.storage_client import StorageClient
from wiser.utils.task_stage_utils import (
    SmoothingFilterSpatial,
    SmoothingFilterSpectral,
    _nan_aware_linear_ndimage_filtered_output,
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
_CALTECH_DATA_IGNORE_HDR = (
    Path(__file__).resolve().parent
    / ".."
    / "test_utils"
    / "test_datasets"
    / "caltech_425_6_6_data_ignore.hdr"
)

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
        """Register dataset, plan a single-stage pipeline, run it, and return (array, meta).

        Does not shut down storage or the scheduler; callers must run
        ``_finalize_pipeline_storage_mnf_style`` once after all pipeline work for this
        test (same pattern as ``test_mnf``).
        """
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
        try:
            output_ref = task_plan.bindings[output_ref_name]
            output_data, _ = storage_client.read_data(output_ref, filter_data=False)
            output_meta = storage_client.get_meta(output_ref)
            return np.asarray(output_data, dtype=np.float32), output_meta
        finally:
            storage_client.close()

    def _finalize_pipeline_storage_mnf_style(self) -> None:
        """Match ``test_mnf`` teardown: release KEEP refs, stop workers, close storage."""
        app_services = self.test_model.app_services
        release_kept_refs(app_services)
        app_services.scheduler.shutdown(wait=True)
        app_services.storage_service.close()

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

    def _assert_output_not_all_nan(self, output: np.ndarray, label: str) -> None:
        self.assertFalse(np.isnan(output).all(), msg=f"{label} output is entirely NaN")

    def _dataset_with_data_ignore_written_as_nan(self, base: RasterDataSet) -> RasterDataSet:
        """
        Rewrite the stored raster so invalid samples are NaN in the buffer:

        - Pixels equal to the data-ignore sentinel become NaN (metadata ignore value unchanged).
        - Every sample in a bad band (``bad_bands[b] == 0``) becomes NaN (bad-band list unchanged).

        ``build_smoothing_exclusion_mask`` excludes by matching nodata or bad-band index, not by
        ``isnan`` alone for finite nodata. Writing NaN into those cells makes them participate in
        the filter like ordinary NaNs for NaN-propagation regression tests.
        """
        ignore = base.get_data_ignore_value()
        if ignore is None:
            raise AssertionError("NaN-propagation fixture must define a data-ignore value.")

        arr = np.asarray(base.get_image_data(filter_data_ignore_value=False), dtype=np.float32)
        if isinstance(ignore, (float, np.floating)) and np.isnan(ignore):
            pass
        else:
            ignore_f = np.float32(ignore)
            arr[np.isclose(arr, ignore_f, rtol=0.0, atol=0.0, equal_nan=True)] = np.nan

        bad_bands = base.get_bad_bands()
        if bad_bands is not None:
            bb = np.asarray(bad_bands)
            if bb.ndim != 1:
                raise AssertionError(f"bad_bands must be 1-D, got shape {bb.shape}")
            if bb.shape[0] != arr.shape[0]:
                raise AssertionError(
                    f"bad_bands length {bb.shape[0]} must match band dimension {arr.shape[0]}"
                )
            for b in range(bb.shape[0]):
                if bb[b] == 0:
                    arr[b, :, :] = np.nan

        out = self.test_model.raster_data_loader.dataset_from_numpy_array(arr, self.test_model.data_cache)
        base_name = base.get_name() or "dataset"
        out.set_name(f"{base_name} (ignore+badbands→NaN)")
        out.set_description(base.get_description())
        out.set_data_ignore_value(ignore)
        if base.get_bad_bands() is not None:
            out.set_bad_bands(base.get_bad_bands())
        if base.has_wavelengths():
            out.copy_spectral_metadata(base.get_spectral_metadata())
        if base.get_spatial_metadata().get_spatial_ref():
            out.copy_spatial_metadata(base.get_spatial_metadata())

        self.test_model.app_state.add_dataset(out)
        return out

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
        try:
            actual, actual_meta = self._run_stage(dataset, stage, output_ref_name)

            work = self._expected_input(dataset)
            expected = np.asarray(
                ndimage.uniform_filter(work, size=(3, 3), axes=(0, 1), mode="reflect"),
                dtype=np.float32,
            )

            self.assertEqual(actual.shape, expected.shape)
            # Restore nodata/bad-band positions from input before comparing
            mask = ~np.isfinite(work)
            expected[mask] = work[mask]
            self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
            self._check_meta(actual_meta, dataset)
        finally:
            self._finalize_pipeline_storage_mnf_style()

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
        try:
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
        finally:
            self._finalize_pipeline_storage_mnf_style()

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
        try:
            actual, actual_meta = self._run_stage(dataset, stage, output_ref_name)

            work = self._expected_input(dataset)
            expected = _nan_aware_linear_ndimage_filtered_output(
                work,
                ndimage.gaussian_filter,
                {"sigma": (1.0, 1.0), "axes": (0, 1), "mode": "reflect", "truncate": 4.0},
            )

            mask = ~np.isfinite(work)
            expected[mask] = work[mask]
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
            self._check_meta(actual_meta, dataset)
        finally:
            self._finalize_pipeline_storage_mnf_style()

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
        try:
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
        finally:
            self._finalize_pipeline_storage_mnf_style()

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
        try:
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
        finally:
            self._finalize_pipeline_storage_mnf_style()

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
        try:
            actual, actual_meta = self._run_stage(dataset, stage, output_ref_name)

            work = self._expected_input(dataset)
            expected = _nan_aware_linear_ndimage_filtered_output(
                work,
                ndimage.gaussian_filter,
                {"sigma": 2.0, "axes": (2,), "mode": "reflect", "truncate": 4.0},
            )

            mask = ~np.isfinite(work)
            expected[mask] = work[mask]
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
            self._check_meta(actual_meta, dataset)
        finally:
            self._finalize_pipeline_storage_mnf_style()

    def _run_nan_propagation_cases(self, dataset, cases) -> None:
        try:
            for label, stage_builder, filter_registry_key, filter_kwargs in cases:
                with self.subTest(label=label):
                    output_ref_name = f"{label}_caltech_data_ignore"
                    stage = stage_builder(
                        dataset,
                        filter_registry_key=filter_registry_key,
                        filter_kwargs=filter_kwargs,
                        output_ref_name=output_ref_name,
                    )
                    actual, _ = self._run_stage(dataset, stage, output_ref_name)
                    self._assert_output_not_all_nan(actual, label)
        finally:
            self._finalize_pipeline_storage_mnf_style()

    def test_caltech_data_ignore_mean_nan_propagation_outputs_not_all_nan(self) -> None:
        base = self.test_model.load_dataset(str(_CALTECH_DATA_IGNORE_HDR))
        dataset = self._dataset_with_data_ignore_written_as_nan(base)
        cases = [
            ("spatial_mean", self._make_spatial_stage, "uniform_filter", {"size": 3, "mode": "reflect"}),
            ("spectral_mean", self._make_spectral_stage, "uniform_filter", {"size": 5, "mode": "reflect"}),
        ]
        self._run_nan_propagation_cases(dataset, cases)

    def test_caltech_data_ignore_median_nan_propagation_outputs_not_all_nan(self) -> None:
        base = self.test_model.load_dataset(str(_CALTECH_DATA_IGNORE_HDR))
        dataset = self._dataset_with_data_ignore_written_as_nan(base)
        cases = [
            ("spatial_median", self._make_spatial_stage, "median_filter", {"size": 3, "mode": "reflect"}),
            ("spectral_median", self._make_spectral_stage, "median_filter", {"size": 5, "mode": "reflect"}),
        ]
        self._run_nan_propagation_cases(dataset, cases)

    def test_caltech_data_ignore_gaussian_nan_propagation_outputs_not_all_nan(self) -> None:
        base = self.test_model.load_dataset(str(_CALTECH_DATA_IGNORE_HDR))
        dataset = self._dataset_with_data_ignore_written_as_nan(base)
        cases = [
            (
                "spatial_gaussian",
                self._make_spatial_stage,
                "gaussian_filter",
                {"sigma": 1.0, "mode": "reflect"},
            ),
            (
                "spectral_gaussian",
                self._make_spectral_stage,
                "gaussian_filter",
                {"sigma": 2.0, "mode": "reflect"},
            ),
        ]
        self._run_nan_propagation_cases(dataset, cases)


if __name__ == "__main__":
    unittest.main()
