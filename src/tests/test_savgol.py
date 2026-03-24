import unittest
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import savgol_filter

import tests.context

from test_utils.test_model import WiserTestModel
from wiser.bandmath.types import VariableType
from wiser.gui.sav_golay import SavGolayDialog, savgol_filter_spectrum
from wiser.raster.loader import RasterDataLoader
from wiser.raster.spectrum import NumPyArraySpectrum, SpectrumAtPoint
from wiser.utils.primitives import PriorityClass
from wiser.utils.storage_client import StorageClient
from wiser.utils.storage_layer import ExternalRasterHandle
from wiser.utils.task_stage_utils import (
    get_good_band_runs,
    get_savgol_filter_pipeline,
    recombine_dataset_tile_from_good_band_runs,
    split_dataset_tile_by_good_band_runs,
)
from wiser.utils.task_system import SemanticTask

pytestmark = [
    pytest.mark.integration,
]


class TestSavGolay(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _run_pipeline(self, dataset, *, window_length: int, polyorder: int, output_ref_name: str):
        app_services = self.test_model.app_services
        dataset_ref = app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=dataset)
        )
        pipeline = get_savgol_filter_pipeline(
            dataset_ref=dataset_ref,
            window_length=window_length,
            polyorder=polyorder,
            output_ref_name=output_ref_name,
        )
        task = SemanticTask(
            priority_class=PriorityClass.BACKGROUND,
            input_ref=dataset_ref,
            algorithm_pipeline=pipeline,
        )
        task.id = 3001

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
        return np.asarray(output_data), output_meta, storage_client, app_services

    def test_savgol_pipeline_matches_scipy_on_circuit_fixture(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent
            / ".."
            / "test_utils"
            / "test_datasets"
            / "circuit_4_100_150_um.hdr"
        ).resolve()
        dataset = self.test_model.load_dataset(str(dataset_path))

        storage_client = None
        app_services = None
        try:
            actual, actual_meta, storage_client, app_services = self._run_pipeline(
                dataset,
                window_length=3,
                polyorder=1,
                output_ref_name="savgol_circuit",
            )

            expected_input = np.asarray(
                dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float32
            ).transpose(1, 2, 0)
            expected = np.asarray(
                savgol_filter(expected_input, window_length=3, polyorder=1, deriv=0, axis=2, mode="interp"),
                dtype=np.float32,
            )

            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(np.allclose(actual, expected, atol=1e-5))
            self.assertEqual(actual_meta.shape, expected.shape)
            if dataset.get_bad_bands() is None:
                self.assertIsNone(actual_meta.bad_bands)
            else:
                self.assertTrue(
                    np.array_equal(np.asarray(actual_meta.bad_bands), np.asarray(dataset.get_bad_bands()))
                )
            self.assertEqual(actual_meta.nodata, dataset.get_data_ignore_value())
        finally:
            if storage_client is not None:
                storage_client.close()
            if app_services is not None:
                app_services.scheduler.shutdown(wait=True)
                app_services.storage_service.close()

    def test_savgol_dialog_populates_dataset_choices(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_425_7_7.hdr"
        ).resolve()
        self.test_model.load_dataset(str(dataset_path))

        dialog = SavGolayDialog(
            app_state=self.test_model.app_state,
            app_services=self.test_model.app_services,
            target_type=VariableType.IMAGE_CUBE_DATASET,
        )
        try:
            self.assertEqual(dialog._ui.lbl_choose_ds_spec.text(), "Choose Dataset")
            self.assertGreaterEqual(dialog._ui.cbox_choice.count(), 1)
        finally:
            dialog.close()

    def test_savgol_dialog_populates_spectrum_choices(self) -> None:
        spectrum = NumPyArraySpectrum(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="spec")
        self.test_model.app_state.collect_spectrum(spectrum)

        dialog = SavGolayDialog(
            app_state=self.test_model.app_state,
            app_services=self.test_model.app_services,
            target_type=VariableType.SPECTRUM,
        )
        try:
            self.assertEqual(dialog._ui.lbl_choose_ds_spec.text(), "Choose Spectrum")
            self.assertGreaterEqual(dialog._ui.cbox_choice.count(), 1)
        finally:
            dialog.close()

    def test_savgol_pipeline_matches_scipy_on_jpl_fixture(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_425_7_7.hdr"
        ).resolve()
        dataset = self.test_model.load_dataset(str(dataset_path))

        storage_client = None
        app_services = None
        try:
            actual, actual_meta, storage_client, app_services = self._run_pipeline(
                dataset,
                window_length=5,
                polyorder=2,
                output_ref_name="savgol_jpl",
            )

            expected_input = np.asarray(
                dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float32
            ).transpose(1, 2, 0)
            expected = np.asarray(
                savgol_filter(expected_input, window_length=5, polyorder=2, deriv=0, axis=2, mode="interp"),
                dtype=np.float32,
            )
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(np.allclose(actual, expected, atol=1e-5))
            self.assertEqual(actual_meta.shape, expected.shape)
            if dataset.get_bad_bands() is None:
                self.assertIsNone(actual_meta.bad_bands)
            else:
                self.assertTrue(
                    np.array_equal(np.asarray(actual_meta.bad_bands), np.asarray(dataset.get_bad_bands()))
                )
            self.assertEqual(actual_meta.nodata, dataset.get_data_ignore_value())
        finally:
            if storage_client is not None:
                storage_client.close()
            if app_services is not None:
                app_services.scheduler.shutdown(wait=True)
                app_services.storage_service.close()

    def test_savgol_single_spectrum_matches_dataset_result_on_jpl_fixture(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_425_7_7.hdr"
        ).resolve()
        dataset = self.test_model.load_dataset(str(dataset_path))
        point = (3, 3)

        storage_client = None
        app_services = None
        try:
            actual, _, storage_client, app_services = self._run_pipeline(
                dataset,
                window_length=5,
                polyorder=2,
                output_ref_name="savgol_jpl_spectrum_match",
            )

            point_spectrum = SpectrumAtPoint(dataset, point)
            filtered_spectrum = savgol_filter_spectrum(
                point_spectrum,
                window_length=5,
                polyorder=2,
            )

            self.assertTrue(
                np.allclose(filtered_spectrum.get_spectrum(), actual[point[1], point[0], :], atol=1e-5)
            )
            self.assertEqual(filtered_spectrum.get_wavelengths(), point_spectrum.get_wavelengths())
            self.assertTrue(
                np.array_equal(
                    np.asarray(filtered_spectrum.get_bad_bands()),
                    np.asarray(point_spectrum.get_bad_bands()),
                )
            )
        finally:
            if storage_client is not None:
                storage_client.close()
            if app_services is not None:
                app_services.scheduler.shutdown(wait=True)
                app_services.storage_service.close()

    def test_savgol_pipeline_matches_manual_split_filter_recombine_with_bad_bands_and_nodata(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_425_7_7.hdr"
        ).resolve()
        base_dataset = self.test_model.load_dataset(str(dataset_path))

        image_by_band = np.array(base_dataset.get_image_data(filter_data_ignore_value=False), copy=True)
        band_count, _, _ = image_by_band.shape
        data_ignore_value = -9999.0
        middle_left = (band_count // 2) - 1
        middle_right = band_count // 2
        image_by_band[:, 0, :] = data_ignore_value

        mutated_dataset = RasterDataLoader().dataset_from_numpy_array(image_by_band)
        bad_bands = [1] * band_count
        bad_bands[middle_left] = 0
        bad_bands[middle_right] = 0
        mutated_dataset.set_bad_bands(bad_bands)
        mutated_dataset.set_data_ignore_value(data_ignore_value)

        storage_client = None
        app_services = None
        try:
            actual, actual_meta, storage_client, app_services = self._run_pipeline(
                mutated_dataset,
                window_length=5,
                polyorder=2,
                output_ref_name="savgol_jpl_bad_bands",
            )

            expected_input = np.asarray(
                mutated_dataset.get_image_data(filter_data_ignore_value=False),
                dtype=np.float32,
            ).transpose(1, 2, 0)
            good_band_runs = get_good_band_runs(np.asarray(bad_bands))
            chunks = split_dataset_tile_by_good_band_runs(expected_input, good_band_runs)
            filtered_chunks = [
                np.asarray(
                    savgol_filter(chunk, window_length=5, polyorder=2, deriv=0, axis=2, mode="interp"),
                    dtype=np.float32,
                )
                for chunk in chunks
            ]
            expected = recombine_dataset_tile_from_good_band_runs(
                expected_input.shape,
                good_band_runs,
                filtered_chunks,
                base_array=expected_input,
            )
            expected[0, :, :] = data_ignore_value

            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(np.allclose(actual, expected, atol=1e-5))
            self.assertEqual(actual_meta.bad_bands.tolist(), bad_bands)
            self.assertEqual(actual_meta.nodata, data_ignore_value)
        finally:
            if storage_client is not None:
                storage_client.close()
            if app_services is not None:
                app_services.scheduler.shutdown(wait=True)
                app_services.storage_service.close()


if __name__ == "__main__":
    test_savgol = TestSavGolay()
    test_savgol.setUp()
    try:
        test_savgol.test_savgol_pipeline_matches_scipy_on_circuit_fixture()
        test_savgol.test_savgol_dialog_populates_dataset_choices()
        test_savgol.test_savgol_dialog_populates_spectrum_choices()
        test_savgol.test_savgol_pipeline_matches_scipy_on_jpl_fixture()
        test_savgol.test_savgol_single_spectrum_matches_dataset_result_on_jpl_fixture()
        test_savgol.test_savgol_pipeline_matches_manual_split_filter_recombine_with_bad_bands_and_nodata()
    finally:
        test_savgol.tearDown()
