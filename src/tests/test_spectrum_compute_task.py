"""Tests for semantic-task spectrum extraction (ROI / point) vs synchronous get_spectrum().

Run from ``src/`` using the ``wiser-dev`` conda env, e.g.::

    conda activate wiser-dev
    python -m pytest tests/test_spectrum_compute_task.py -v

Or: ``conda run -n wiser-dev python -m pytest tests/test_spectrum_compute_task.py -v``
"""
import unittest

import numpy as np
import pytest

import tests.context

from PySide2.QtCore import QPoint
from PySide2.QtWidgets import QApplication

from test_utils.memory_cleanup import release_kept_refs
from test_utils.test_model import WiserTestModel

from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import RectangleSelection
from wiser.raster.spectrum import ROIAverageSpectrum, SpectrumAtPoint, SpectrumAverageMode, NumPyArraySpectrum
from wiser.raster.spectrum_compute_task import (
    build_spectrum_recompute_task,
    dataset_plan_meta_from_data_ref,
    get_raster_backed_spectrum_pipeline,
    COMPUTE_KIND_POINT,
    COMPUTE_KIND_ROI,
    DEFAULT_SPECTRUM_OUTPUT_NAME,
)
from wiser.utils.primitives import DeletePolicy, ExternalRasterHandle, PriorityClass
from wiser.utils.storage_client import StorageClient
from wiser.utils.task_system import SemanticTask

pytestmark = [
    pytest.mark.integration,
]


class TestSpectrumComputeTask(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _small_numpy_dataset(self):
        bands, h, w = 4, 8, 8
        arr = np.arange(1, bands + 1, dtype=np.float32).reshape((bands, 1, 1)) * np.ones(
            (bands, h, w), dtype=np.float32
        )
        loader = RasterDataLoader()
        return loader.dataset_from_numpy_array(arr, cache=None)

    def test_numpy_immediate_emits_same_as_get_spectrum(self):
        arr = np.linspace(0.1, 0.9, 12, dtype=np.float32)
        spec = NumPyArraySpectrum(arr, name="t", source_name="s")
        task = build_spectrum_recompute_task(spec)
        out = []
        task.result_ready.connect(lambda a: out.append(np.asarray(a)))
        task.emit_now()
        self.assertEqual(len(out), 1)
        np.testing.assert_array_almost_equal(out[0], spec.get_spectrum())

    def test_roi_average_pipeline_matches_get_spectrum(self):
        dataset = self._small_numpy_dataset()
        roi = RegionOfInterest(name="r")
        roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(3, 3)))

        ref_spec = ROIAverageSpectrum(dataset, roi, avg_mode=SpectrumAverageMode.MEAN)
        expected = ref_spec.get_spectrum()

        app_services = self.test_model.app_services
        storage_client = None
        try:
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            plan_meta = dataset_plan_meta_from_data_ref(dataset_ref)
            pipeline = get_raster_backed_spectrum_pipeline(
                dataset_ref=dataset_ref,
                dataset_plan_meta=plan_meta,
                compute_kind=COMPUTE_KIND_ROI,
                output_ref_name=DEFAULT_SPECTRUM_OUTPUT_NAME,
                roi=roi,
                avg_mode=SpectrumAverageMode.MEAN,
            )
            pipeline.stages[0].set_output_delete_policy(DEFAULT_SPECTRUM_OUTPUT_NAME, DeletePolicy.KEEP)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=pipeline,
            )
            task.id = 9001

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=60)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            out_ref = task_plan.bindings[DEFAULT_SPECTRUM_OUTPUT_NAME]
            got, _ = storage_client.read_data(out_ref, filter_data=False)
            np.testing.assert_allclose(np.asarray(got, dtype=np.float32), expected, rtol=1e-5, atol=1e-5)
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_spectrum_at_point_pipeline_matches_get_spectrum(self):
        dataset = self._small_numpy_dataset()
        point = (2, 3)
        area = (3, 3)
        ref_spec = SpectrumAtPoint(dataset, point, area=area, avg_mode=SpectrumAverageMode.MEAN)
        expected = ref_spec.get_spectrum()

        app_services = self.test_model.app_services
        storage_client = None
        try:
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            plan_meta = dataset_plan_meta_from_data_ref(dataset_ref)
            pipeline = get_raster_backed_spectrum_pipeline(
                dataset_ref=dataset_ref,
                dataset_plan_meta=plan_meta,
                compute_kind=COMPUTE_KIND_POINT,
                output_ref_name=DEFAULT_SPECTRUM_OUTPUT_NAME,
                point=point,
                area=area,
                avg_mode=SpectrumAverageMode.MEAN,
            )
            pipeline.stages[0].set_output_delete_policy(DEFAULT_SPECTRUM_OUTPUT_NAME, DeletePolicy.KEEP)
            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=pipeline,
            )
            task.id = 9002

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=60)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            out_ref = task_plan.bindings[DEFAULT_SPECTRUM_OUTPUT_NAME]
            got, _ = storage_client.read_data(out_ref, filter_data=False)
            np.testing.assert_allclose(np.asarray(got, dtype=np.float32), expected, rtol=1e-5, atol=1e-5)
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_build_spectrum_recompute_task_raster_emits_via_signal(self):
        dataset = self._small_numpy_dataset()
        roi = RegionOfInterest(name="r2")
        roi.add_selection(RectangleSelection(QPoint(1, 1), QPoint(4, 4)))
        ref_spec = ROIAverageSpectrum(dataset, roi)

        app_services = self.test_model.app_services
        try:
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            sem = build_spectrum_recompute_task(
                ref_spec,
                dataset_ref=dataset_ref,
                task_id=9003,
            )
            self.assertIsNotNone(sem.id)
            received = []

            def on_ready(arr):
                received.append(np.asarray(arr))

            sem.result_ready.connect(on_ready)
            task_plan = app_services.task_planner.plan_semantic_task(sem)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=60)
            # completion_callback emits from the scheduler thread; Qt queues result_ready to the GUI thread.
            QApplication.processEvents()
            self.assertEqual(len(received), 1)
            np.testing.assert_allclose(received[0], ref_spec.get_spectrum(), rtol=1e-5, atol=1e-5)
        finally:
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()


class TestGetSpectrumAsyncMatchSync(unittest.TestCase):
    """``get_spectrum_async`` should match ``get_spectrum()`` once work completes."""

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _small_numpy_dataset(self):
        bands, h, w = 4, 8, 8
        arr = np.arange(1, bands + 1, dtype=np.float32).reshape((bands, 1, 1)) * np.ones(
            (bands, h, w), dtype=np.float32
        )
        loader = RasterDataLoader()
        return loader.dataset_from_numpy_array(arr, cache=None)

    def test_numpy_get_spectrum_async_matches_get_spectrum(self):
        arr = np.linspace(0.2, 1.1, 16, dtype=np.float32)
        spec = NumPyArraySpectrum(arr, name="async_np", source_name="src")
        expected = spec.get_spectrum()
        received = []

        spec.get_spectrum_async(done=lambda a: received.append(np.asarray(a, dtype=np.float32)))

        self.assertEqual(len(received), 1)
        np.testing.assert_array_equal(received[0], expected)

    def test_roi_average_get_spectrum_async_matches_after_future(self):
        dataset = self._small_numpy_dataset()
        roi = RegionOfInterest(name="async_roi")
        roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(5, 5)))
        spec = ROIAverageSpectrum(dataset, roi, avg_mode=SpectrumAverageMode.MEAN)
        expected = spec.get_spectrum()

        app_services = self.test_model.app_services
        futures = []

        def submit(task):
            f = app_services.submit_semantic_task(task)
            futures.append(f)
            return f

        try:
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            received = []
            spec.get_spectrum_async(
                dataset_ref=dataset_ref,
                submit_semantic_task=submit,
                done=lambda a: received.append(np.asarray(a, dtype=np.float32)),
                task_id=9101,
            )
            self.assertEqual(len(futures), 1)
            futures[0].result(timeout=60)
            QApplication.processEvents()

            self.assertEqual(len(received), 1)
            np.testing.assert_allclose(received[0], expected, rtol=1e-5, atol=1e-5)
        finally:
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_spectrum_at_point_get_spectrum_async_matches_after_future(self):
        dataset = self._small_numpy_dataset()
        spec = SpectrumAtPoint(
            dataset,
            (2, 3),
            area=(3, 3),
            avg_mode=SpectrumAverageMode.MEAN,
        )
        expected = spec.get_spectrum()

        app_services = self.test_model.app_services
        futures = []

        def submit(task):
            f = app_services.submit_semantic_task(task)
            futures.append(f)
            return f

        try:
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            received = []
            spec.get_spectrum_async(
                dataset_ref=dataset_ref,
                submit_semantic_task=submit,
                done=lambda a: received.append(np.asarray(a, dtype=np.float32)),
                task_id=9102,
            )
            self.assertEqual(len(futures), 1)
            futures[0].result(timeout=60)
            QApplication.processEvents()

            self.assertEqual(len(received), 1)
            np.testing.assert_allclose(received[0], expected, rtol=1e-5, atol=1e-5)
        finally:
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()
