import unittest
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

import tests.context

from test_utils.test_model import WiserTestModel
from wiser.gui.smooth_filter import (
    SmoothingDomain,
    SmoothingFilterKind,
    SmoothingFilterSemanticTask,
)
from wiser.utils.primitives import ExternalRasterHandle
from wiser.utils.task_stage_utils import build_smoothing_exclusion_mask

pytestmark = [
    pytest.mark.integration,
]

_JPL_HDR = Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_425_7_7.hdr"


class TestSmoothingFilterSemanticTask(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _submit_task(
        self,
        dataset,
        *,
        domain: SmoothingDomain,
        filter_kind: SmoothingFilterKind,
        mode: str,
        cval: float = 0.0,
        size=None,
        sigma=None,
        radius=None,
        output_ref_name: str = "smooth_out",
    ):
        app_state = self.test_model.app_state
        app_services = self.test_model.app_services

        dataset_ref = app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=dataset)
        )
        task = SmoothingFilterSemanticTask(
            app_state=app_state,
            source_dataset=dataset,
            input_ref=dataset_ref,
            domain=domain,
            filter_kind=filter_kind,
            mode=mode,
            cval=cval,
            size=size,
            sigma=sigma,
            radius=radius,
            output_ref_name=output_ref_name,
        )
        task_plan = app_services.task_planner.plan_semantic_task(task)
        future = app_services.task_manager.register_and_submit_task_plan(app_services.scheduler, task_plan)
        future.result(timeout=180)
        self.test_model.app.processEvents()

        datasets = app_state.get_datasets()
        self.assertGreaterEqual(len(datasets), 1, "No dataset was added after task completion")
        return datasets[-1]

    def _expected_input(self, dataset) -> np.ndarray:
        """
        Float32 [y][x][b] array with nodata / bad-band positions and remaining
        non-finite values set to NaN — mirrors what the tile runner does before scipy.
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

    def _result_array(self, result_dataset) -> np.ndarray:
        """Return result dataset as float32 [y][x][b]."""
        return np.asarray(
            result_dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float32
        ).transpose(1, 2, 0)

    def _check_metadata(self, result_dataset, source_dataset) -> None:
        source_bad_bands = source_dataset.get_bad_bands()
        if source_bad_bands is None:
            self.assertIsNone(result_dataset.get_bad_bands())
        else:
            self.assertTrue(
                np.array_equal(
                    np.asarray(result_dataset.get_bad_bands()),
                    np.asarray(source_bad_bands),
                )
            )
        self.assertEqual(
            result_dataset.get_data_ignore_value(),
            source_dataset.get_data_ignore_value(),
        )
        if source_dataset.has_wavelengths():
            self.assertTrue(result_dataset.has_wavelengths())

    # ------------------------------------------------------------------
    # Spatial: mean (uniform_filter)
    # ------------------------------------------------------------------

    def test_semantic_task_spatial_mean_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        result_dataset = self._submit_task(
            dataset,
            domain=SmoothingDomain.SPATIAL,
            filter_kind=SmoothingFilterKind.MEAN,
            mode="reflect",
            size=3,
            output_ref_name="sem_spatial_mean",
        )

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.uniform_filter(work, size=(3, 3), axes=(0, 1), mode="reflect"),
            dtype=np.float32,
        )
        mask = ~np.isfinite(work)
        expected[mask] = work[mask]

        actual = self._result_array(result_dataset)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_metadata(result_dataset, dataset)

    # ------------------------------------------------------------------
    # Spatial: median (median_filter)
    # ------------------------------------------------------------------

    def test_semantic_task_spatial_median_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        result_dataset = self._submit_task(
            dataset,
            domain=SmoothingDomain.SPATIAL,
            filter_kind=SmoothingFilterKind.MEDIAN,
            mode="reflect",
            size=3,
            output_ref_name="sem_spatial_median",
        )

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.median_filter(work, size=(3, 3), axes=(0, 1), mode="reflect"),
            dtype=np.float32,
        )
        mask = ~np.isfinite(work)
        expected[mask] = work[mask]

        actual = self._result_array(result_dataset)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_metadata(result_dataset, dataset)

    # ------------------------------------------------------------------
    # Spatial: gaussian (gaussian_filter)
    # ------------------------------------------------------------------

    def test_semantic_task_spatial_gaussian_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        result_dataset = self._submit_task(
            dataset,
            domain=SmoothingDomain.SPATIAL,
            filter_kind=SmoothingFilterKind.GAUSSIAN,
            mode="reflect",
            sigma=1.0,
            output_ref_name="sem_spatial_gaussian",
        )

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.gaussian_filter(work, sigma=(1.0, 1.0), axes=(0, 1), mode="reflect", truncate=4.0),
            dtype=np.float32,
        )
        mask = ~np.isfinite(work)
        expected[mask] = work[mask]

        actual = self._result_array(result_dataset)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_metadata(result_dataset, dataset)

    # ------------------------------------------------------------------
    # Spectral: mean (uniform_filter)
    # ------------------------------------------------------------------

    def test_semantic_task_spectral_mean_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        result_dataset = self._submit_task(
            dataset,
            domain=SmoothingDomain.SPECTRAL,
            filter_kind=SmoothingFilterKind.MEAN,
            mode="reflect",
            size=5,
            output_ref_name="sem_spectral_mean",
        )

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.uniform_filter(work, size=5, axes=(2,), mode="reflect"),
            dtype=np.float32,
        )
        mask = ~np.isfinite(work)
        expected[mask] = work[mask]

        actual = self._result_array(result_dataset)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_metadata(result_dataset, dataset)

    # ------------------------------------------------------------------
    # Spectral: median (median_filter)
    # ------------------------------------------------------------------

    def test_semantic_task_spectral_median_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        result_dataset = self._submit_task(
            dataset,
            domain=SmoothingDomain.SPECTRAL,
            filter_kind=SmoothingFilterKind.MEDIAN,
            mode="reflect",
            size=5,
            output_ref_name="sem_spectral_median",
        )

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.median_filter(work, size=5, axes=(2,), mode="reflect"),
            dtype=np.float32,
        )
        mask = ~np.isfinite(work)
        expected[mask] = work[mask]

        actual = self._result_array(result_dataset)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_metadata(result_dataset, dataset)

    # ------------------------------------------------------------------
    # Spectral: gaussian (gaussian_filter)
    # ------------------------------------------------------------------

    def test_semantic_task_spectral_gaussian_matches_scipy_on_jpl(self) -> None:
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        result_dataset = self._submit_task(
            dataset,
            domain=SmoothingDomain.SPECTRAL,
            filter_kind=SmoothingFilterKind.GAUSSIAN,
            mode="reflect",
            sigma=2.0,
            output_ref_name="sem_spectral_gaussian",
        )

        work = self._expected_input(dataset)
        expected = np.asarray(
            ndimage.gaussian_filter(work, sigma=2.0, axes=(2,), mode="reflect", truncate=4.0),
            dtype=np.float32,
        )
        mask = ~np.isfinite(work)
        expected[mask] = work[mask]

        actual = self._result_array(result_dataset)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.allclose(actual, expected, atol=1e-5, equal_nan=True))
        self._check_metadata(result_dataset, dataset)


if __name__ == "__main__":
    unittest.main()
