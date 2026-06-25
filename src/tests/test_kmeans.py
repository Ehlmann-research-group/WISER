import datetime
from typing import Tuple
import unittest
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKMeans

import tests.context  # noqa: F401 – adds src/ to sys.path

from test_utils.memory_cleanup import release_kept_refs
from test_utils.test_model import WiserTestModel
from wiser.gui.kmeans import (
    KMeansAlgorithm,
    KMeansCentroids,
    KMeansInitMethod,
    KMeansParameters,
    KMeansRunHistoryDialog,
    KMeansRunRecord,
    KMeansSemanticTask,
    get_kmeans_pipeline,
)
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.utils.primitives import DeletePolicy, ExternalRasterHandle, PriorityClass
from wiser.utils.storage_client import StorageClient
from wiser.utils.task_system import SemanticTask

from tests.utils import (
    NAN_INF_BAD_BANDS,
    NAN_INF_DATA_IGNORE_VALUE,
    NAN_INF_INVALID_YX,
    build_unmasked_nan_inf_cube,
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

_K = 5
_SEED = 42


class TestKMeansStage(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_kmeans_stage_labels_and_centroids_match_sklearn_on_jpl(self) -> None:
        """
        Run the KMeansStage pipeline on the jpl_425_7_7 fixture and verify
        that the per-pixel labels and the full-band centroids are exactly the
        values that sklearn KMeans produces when given identical inputs and the
        same seed.
        """
        app_services = self.test_model.app_services
        storage_client = None
        try:
            dataset = RasterDataLoader().load_from_file(str(_JPL_HDR))[0]
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )

            # ------------------------------------------------------------------
            # Reference: jpl_425_7_7 has no bad bands and no nodata, so simply
            # flatten, cluster, and reshape back.
            # ------------------------------------------------------------------
            image_yxb = np.asarray(
                dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float32
            ).transpose(1, 2, 0)  # (y, x, b)
            y, x, b = image_yxb.shape
            flat = image_yxb.reshape(y * x, b)

            ref_kmeans = SklearnKMeans(
                n_clusters=_K,
                init="k-means++",
                n_init=3,
                max_iter=100,
                tol=1e-4,
                random_state=_SEED,
                algorithm="lloyd",
            )
            ref_labels_flat = ref_kmeans.fit_predict(flat).astype(np.int32)
            ref_labels_image = ref_labels_flat.reshape(y, x, 1)
            ref_centroids = ref_kmeans.cluster_centers_.astype(np.float32)  # (k, b)

            params = KMeansParameters(
                dataset_id=0,  # dataset not registered in app_state; use sentinel
                k=_K,
                init_method=KMeansInitMethod.KMEANS_PLUS_PLUS,
                num_inits=3,
                max_iter=100,
                tol=1e-4,
                seed=_SEED,
                algorithm=KMeansAlgorithm.LLOYD,
            )

            pipeline = get_kmeans_pipeline(dataset_ref, params)
            pipeline.stages[0].set_output_delete_policy("kmeans_labels", DeletePolicy.KEEP)
            pipeline.stages[0].set_output_delete_policy("kmeans_centroids", DeletePolicy.KEEP)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=pipeline,
            )
            task.id = 5001

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            future.result(timeout=60)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )

            labels_ref = task_plan.bindings["kmeans_labels"]
            centroids_ref = task_plan.bindings["kmeans_centroids"]
            labels_out, _ = storage_client.read_data(labels_ref)
            centroids_out, _ = storage_client.read_data(centroids_ref)

            # ------------------------------------------------------------------
            # Assertions
            # ------------------------------------------------------------------
            labels_array = np.asarray(labels_out).astype(np.int32)
            centroids_array = np.asarray(centroids_out)

            self.assertEqual(labels_array.shape, ref_labels_image.shape)
            self.assertEqual(centroids_array.shape, ref_centroids.shape)

            np.testing.assert_array_equal(
                labels_array,
                ref_labels_image,
                err_msg="Per-pixel cluster labels do not match sklearn reference.",
            )
            np.testing.assert_allclose(
                centroids_array,
                ref_centroids,
                atol=1e-5,
                err_msg="Cluster centroids do not match sklearn reference.",
            )

        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()


class TestKMeansSemanticTask(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_semantic_task_adds_labels_dataset_matching_sklearn_on_jpl(self) -> None:
        """
        Run KMeansSemanticTask on jpl_425_7_7 and verify that:
          - exactly one new dataset is added to WISER after the task completes
          - the label image in that dataset matches sklearn KMeans with the same seed
          - the dataset's data_ignore_value is -1
        """
        app_state = self.test_model.app_state
        app_services = self.test_model.app_services

        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        dataset_ref = app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=dataset)
        )

        params = KMeansParameters(
            dataset_id=dataset.get_id(),
            k=_K,
            init_method=KMeansInitMethod.KMEANS_PLUS_PLUS,
            num_inits=3,
            max_iter=100,
            tol=1e-4,
            seed=_SEED,
            algorithm=KMeansAlgorithm.LLOYD,
        )

        datasets_before = len(app_state.get_datasets())

        kmeans_task = KMeansSemanticTask(
            app_state=app_state,
            source_dataset=dataset,
            input_ref=dataset_ref,
            params=params,
        )
        # The dialog wires this in production; the test drives the task directly.
        kmeans_task.run_recorded.connect(app_state.get_kmeans_history().add_record)

        task_plan = app_services.task_planner.plan_semantic_task(kmeans_task)
        future = app_services.task_manager.register_and_submit_task_plan(app_services.scheduler, task_plan)
        future.result(timeout=180)
        self.test_model.app.processEvents()

        datasets_after = app_state.get_datasets()
        self.assertEqual(
            len(datasets_after),
            datasets_before + 1,
            "Expected exactly one new dataset to be added by the semantic task",
        )

        labels_ds = datasets_after[-1]

        self.assertEqual(labels_ds.get_data_ignore_value(), -1)

        # (1, y, x) from get_image_data → (y, x, 1) after transpose
        labels_byb = np.asarray(labels_ds.get_image_data(filter_data_ignore_value=False))
        labels_array = labels_byb.transpose(1, 2, 0).astype(np.int32)

        # ------------------------------------------------------------------
        # Reference: jpl_425_7_7 has no bad bands and no nodata
        # ------------------------------------------------------------------
        image_yxb = np.asarray(
            dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float32
        ).transpose(1, 2, 0)  # (y, x, b)
        y, x, b = image_yxb.shape
        flat = image_yxb.reshape(y * x, b)

        ref_kmeans = SklearnKMeans(
            n_clusters=_K,
            init="k-means++",
            n_init=3,
            max_iter=100,
            tol=1e-4,
            random_state=_SEED,
            algorithm="lloyd",
        )
        ref_labels_flat = ref_kmeans.fit_predict(flat).astype(np.int32)
        ref_labels_image = ref_labels_flat.reshape(y, x, 1)

        self.assertEqual(labels_array.shape, ref_labels_image.shape)
        np.testing.assert_array_equal(
            labels_array,
            ref_labels_image,
            err_msg="Semantic task label image does not match sklearn reference.",
        )

        # Verify the completed run was recorded in the K-Means history.
        records = app_state.get_kmeans_history().get_records()
        self.assertEqual(len(records), 1, "Expected exactly one K-Means run record")
        record = records[-1]
        self.assertEqual(record.params, params)
        self.assertEqual(record.effective_seed, _SEED)
        stored_centroids = record.centroids
        self.assertIsNotNone(stored_centroids, "KMeansCentroids were not stored in the run record")
        self.assertEqual(stored_centroids.num_centroids(), _K)
        ref_centroids = ref_kmeans.cluster_centers_.astype(np.float32)
        np.testing.assert_allclose(
            stored_centroids._centroids,
            ref_centroids,
            atol=1e-5,
            err_msg="Stored centroids do not match sklearn reference.",
        )

    def test_semantic_task_data_ignore_pixel_reads_as_nan(self) -> None:
        # Guard against regression where labels stored as int32 caused np.nan
        # to be silently truncated to 0, making data-ignore pixels appear as
        # cluster 0 instead of NaN when queried via get_all_bands_at.
        app_state = self.test_model.app_state
        app_services = self.test_model.app_services

        dataset = self.test_model.load_dataset(str(_CALTECH_DATA_IGNORE_HDR))
        dataset_ref = app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=dataset)
        )

        params = KMeansParameters(
            dataset_id=dataset.get_id(),
            k=_K,
            init_method=KMeansInitMethod.KMEANS_PLUS_PLUS,
            num_inits=3,
            max_iter=100,
            tol=1e-4,
            seed=_SEED,
            algorithm=KMeansAlgorithm.LLOYD,
        )

        kmeans_task = KMeansSemanticTask(
            app_state=app_state,
            source_dataset=dataset,
            input_ref=dataset_ref,
            params=params,
        )

        task_plan = app_services.task_planner.plan_semantic_task(kmeans_task)
        future = app_services.task_manager.register_and_submit_task_plan(app_services.scheduler, task_plan)
        future.result(timeout=180)
        self.test_model.app.processEvents()

        datasets_after = app_state.get_datasets()
        labels_ds = datasets_after[-1]

        # Pixel (0, 0) is a data-ignore pixel in caltech_425_6_6_data_ignore.
        # With float32 labels the nodata sentinel -1 round-trips through NaN
        # correctly; with the old int32 labels it silently became 0.
        data_ignore_pixel_value = labels_ds.get_all_bands_at(0, 0)
        self.assertTrue(
            np.isnan(data_ignore_pixel_value[0]),
            f"Expected NaN for data-ignore pixel (0, 0) but got {data_ignore_pixel_value[0]}",
        )


class TestKMeansSemanticTaskParameters(unittest.TestCase):
    """
    Parameter-coverage tests for KMeansSemanticTask.

    Each test exercises one non-default parameter while leaving the others as
    None (sklearn defaults).  Every test verifies:
      - exactly one new dataset is added to WISER after the task completes
      - the label image has shape (y, x, 1) with all values in [-1, k-1]
      - data_ignore_value is -1
      - KMeansCentroids are stored in app_state with shape (k, b)
    """

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _submit_task(self, dataset, params) -> Tuple[RasterDataSet, KMeansCentroids]:
        """Register *dataset*, submit KMeansSemanticTask, process events.

        Returns ``(labels_ds, centroids)`` after asserting a new dataset was added.
        """
        app_state = self.test_model.app_state
        app_services = self.test_model.app_services

        dataset_ref = app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=dataset)
        )

        datasets_before = len(app_state.get_datasets())

        kmeans_task = KMeansSemanticTask(
            app_state=app_state,
            source_dataset=dataset,
            input_ref=dataset_ref,
            params=params,
        )
        # The dialog wires this in production; the test drives the task directly.
        kmeans_task.run_recorded.connect(app_state.get_kmeans_history().add_record)

        task_plan = app_services.task_planner.plan_semantic_task(kmeans_task)
        future = app_services.task_manager.register_and_submit_task_plan(app_services.scheduler, task_plan)
        future.result(timeout=180)
        self.test_model.app.processEvents()

        datasets_after = app_state.get_datasets()
        self.assertEqual(
            len(datasets_after),
            datasets_before + 1,
            "Expected exactly one new dataset to be added",
        )

        labels_ds = datasets_after[-1]
        centroids = app_state.get_kmeans_history().get_records()[-1].centroids
        return labels_ds, centroids

    def _assert_output(self, dataset, labels_ds, centroids, k=_K):
        """Verify basic correctness of the semantic-task output."""
        self.assertEqual(labels_ds.get_data_ignore_value(), -1)

        labels_byb = np.asarray(labels_ds.get_image_data(filter_data_ignore_value=False))
        self.assertEqual(labels_byb.shape[0], 1, "Expected 1-band label image")
        labels_arr = labels_byb[0].astype(np.int32)  # (y, x)
        self.assertTrue(np.all(labels_arr >= -1), "Label value below -1 found")
        self.assertTrue(np.all(labels_arr < k), "Label value >= k found")

        b = np.asarray(dataset.get_image_data(filter_data_ignore_value=False)).shape[0]

        self.assertIsNotNone(centroids, "KMeansCentroids were not stored in app_state")
        self.assertEqual(centroids.num_centroids(), k)
        self.assertEqual(centroids._centroids.shape, (k, b))

    def _make_params(self, dataset, **overrides):
        """Build KMeansParameters with sensible defaults, applying *overrides*."""
        kwargs = dict(
            dataset_id=dataset.get_id(),
            k=_K,
            init_method=KMeansInitMethod.KMEANS_PLUS_PLUS,
            num_inits=None,
            max_iter=None,
            tol=None,
            seed=None,
            algorithm=KMeansAlgorithm.LLOYD,
        )
        kwargs.update(overrides)
        return KMeansParameters(**kwargs)

    # ------------------------------------------------------------------
    # Baseline: all parameters explicitly specified
    # ------------------------------------------------------------------

    def test_all_parameters_set(self):
        """Run with every parameter given an explicit value and verify exact sklearn match."""
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        params = self._make_params(
            dataset,
            init_method=KMeansInitMethod.KMEANS_PLUS_PLUS,
            num_inits=3,
            max_iter=100,
            tol=1e-4,
            seed=_SEED,
            algorithm=KMeansAlgorithm.LLOYD,
        )
        labels_ds, centroids = self._submit_task(dataset, params)
        self._assert_output(dataset, labels_ds, centroids)

        image_yxb = np.asarray(
            dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float32
        ).transpose(1, 2, 0)
        y, x, b = image_yxb.shape
        flat = image_yxb.reshape(y * x, b)
        ref = SklearnKMeans(
            n_clusters=_K,
            init="k-means++",
            n_init=3,
            max_iter=100,
            tol=1e-4,
            random_state=_SEED,
            algorithm="lloyd",
        )
        ref_labels = ref.fit_predict(flat).astype(np.int32).reshape(y, x, 1)
        ref_centroids = ref.cluster_centers_.astype(np.float32)

        labels_byb = np.asarray(labels_ds.get_image_data(filter_data_ignore_value=False))
        np.testing.assert_array_equal(
            labels_byb.transpose(1, 2, 0).astype(np.int32),
            ref_labels,
            err_msg="all-parameters test: labels do not match sklearn reference.",
        )
        np.testing.assert_allclose(
            centroids._centroids,
            ref_centroids,
            atol=1e-5,
            err_msg="all-parameters test: centroids do not match sklearn reference.",
        )

    # ------------------------------------------------------------------
    # Init-method variants
    # ------------------------------------------------------------------

    def test_init_method_kmeans_plus_plus_rest_none(self):
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        params = self._make_params(dataset, init_method=KMeansInitMethod.KMEANS_PLUS_PLUS)
        labels_ds, centroids = self._submit_task(dataset, params)
        self._assert_output(dataset, labels_ds, centroids)

    def test_init_method_random_rest_none(self):
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        params = self._make_params(dataset, init_method=KMeansInitMethod.RANDOM)
        labels_ds, centroids = self._submit_task(dataset, params)
        self._assert_output(dataset, labels_ds, centroids)

    def test_init_method_manual_rest_none(self):
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        # Extract k pixel spectra from the image to use as manual initial centroids.
        image_byb = np.asarray(
            dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float32
        )  # (b, y, x)
        flat = image_byb.transpose(1, 2, 0).reshape(-1, image_byb.shape[0])  # (y*x, b)
        manual_spectra = [flat[i] for i in range(_K)]

        params = self._make_params(
            dataset,
            init_method=KMeansInitMethod.MANUAL,
            _manual_spectra=manual_spectra,
        )
        labels_ds, centroids = self._submit_task(dataset, params)
        self._assert_output(dataset, labels_ds, centroids)

    # ------------------------------------------------------------------
    # Individual optional parameters
    # ------------------------------------------------------------------

    def test_num_inits_set(self):
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        params = self._make_params(dataset, num_inits=5)
        labels_ds, centroids = self._submit_task(dataset, params)
        self._assert_output(dataset, labels_ds, centroids)

    def test_max_iter_set(self):
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        params = self._make_params(dataset, max_iter=50)
        labels_ds, centroids = self._submit_task(dataset, params)
        self._assert_output(dataset, labels_ds, centroids)

    def test_tol_set(self):
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        params = self._make_params(dataset, tol=1e-3)
        labels_ds, centroids = self._submit_task(dataset, params)
        self._assert_output(dataset, labels_ds, centroids)

    def test_seed_set(self):
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        params = self._make_params(dataset, seed=_SEED)
        labels_ds, centroids = self._submit_task(dataset, params)
        self._assert_output(dataset, labels_ds, centroids)

        # With a fixed seed we can also verify exact agreement with sklearn.
        image_yxb = np.asarray(
            dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float32
        ).transpose(1, 2, 0)
        y, x, b = image_yxb.shape
        flat = image_yxb.reshape(y * x, b)
        ref = SklearnKMeans(
            n_clusters=_K,
            init="k-means++",
            n_init=10,
            max_iter=300,
            tol=1e-4,
            random_state=_SEED,
            algorithm="lloyd",
        )
        ref_labels = ref.fit_predict(flat).astype(np.int32).reshape(y, x, 1)
        labels_byb = np.asarray(labels_ds.get_image_data(filter_data_ignore_value=False))
        np.testing.assert_array_equal(
            labels_byb.transpose(1, 2, 0).astype(np.int32),
            ref_labels,
            err_msg="seed-only test: labels do not match sklearn reference.",
        )

    def test_algorithm_elkan_set(self):
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        params = self._make_params(dataset, algorithm=KMeansAlgorithm.ELKAN)
        labels_ds, centroids = self._submit_task(dataset, params)
        self._assert_output(dataset, labels_ds, centroids)


class TestKMeansNanResistance(unittest.TestCase):
    """KMeans must drop unmasked NaN/Inf pixels (and nodata) and label them as
    data-ignore (-1), instead of erroring. Pre-fix, `_run_kmeans` raised
    ValueError on any non-finite value in a valid (non-nodata) pixel."""

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_kmeans_tolerates_unmasked_nan_and_inf(self) -> None:
        # Synthetic cube with a bad band, two all-nodata pixels, and unmasked
        # NaN/+Inf/-Inf in good bands at distinct pixels (NAN_INF_INVALID_YX).
        dataset = RasterDataLoader().dataset_from_numpy_array(build_unmasked_nan_inf_cube())
        dataset.set_bad_bands(NAN_INF_BAD_BANDS)
        dataset.set_data_ignore_value(NAN_INF_DATA_IGNORE_VALUE)

        app_services = self.test_model.app_services
        storage_client = None
        try:
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            k = 3
            params = KMeansParameters(
                dataset_id=0,  # dataset not registered in app_state; use sentinel
                k=k,
                init_method=KMeansInitMethod.KMEANS_PLUS_PLUS,
                num_inits=3,
                max_iter=100,
                tol=1e-4,
                seed=_SEED,
                algorithm=KMeansAlgorithm.LLOYD,
            )

            pipeline = get_kmeans_pipeline(dataset_ref, params)
            pipeline.stages[0].set_output_delete_policy("kmeans_labels", DeletePolicy.KEEP)
            pipeline.stages[0].set_output_delete_policy("kmeans_centroids", DeletePolicy.KEEP)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=dataset_ref,
                algorithm_pipeline=pipeline,
            )
            task.id = 5101

            task_plan = app_services.task_planner.plan_semantic_task(task)
            future = app_services.scheduler.run_task_plan(task_plan)
            # Pre-fix this raised "KMeans input contains NaN or infinite values...".
            future.result(timeout=60)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )
            labels_out, _ = storage_client.read_data(task_plan.bindings["kmeans_labels"])
            labels = np.asarray(labels_out).astype(np.int32)  # (y, x, 1)

            self.assertEqual(labels.shape, (12, 12, 1))

            # Every nodata / NaN / Inf pixel is dropped and labelled data-ignore.
            invalid_mask = np.zeros((12, 12), dtype=bool)
            for yy, xx in NAN_INF_INVALID_YX:
                invalid_mask[yy, xx] = True
                self.assertEqual(
                    labels[yy, xx, 0],
                    -1,
                    f"Invalid pixel {(yy, xx)} should be labelled data-ignore (-1)",
                )

            # Every surviving pixel got a real cluster label in [0, k-1].
            valid_labels = labels[~invalid_mask][:, 0]
            self.assertTrue(np.all(valid_labels >= 0))
            self.assertTrue(np.all(valid_labels < k))
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()


class TestKMeansRunHistoryDialog(unittest.TestCase):
    """Rendering tests for the past-runs viewer (no clustering involved).

    Records are added to the manager directly so the dialog logic — the
    active/closed split, the auto-seed flag, status text, and Delete — can be
    exercised without paying for a real K-Means run.
    """

    # Column indices mirror the private constants in kmeans.py.
    _COL_INIT = 4
    _COL_SEED = 5
    _COL_STATUS = 6

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _make_record(self, run_id, dataset_id, name, *, seed, effective_seed, k=3):
        params = KMeansParameters(
            dataset_id=dataset_id,
            k=k,
            init_method=KMeansInitMethod.RANDOM,
            num_inits=None,
            max_iter=None,
            tol=None,
            seed=seed,
            algorithm=KMeansAlgorithm.LLOYD,
        )
        return KMeansRunRecord(
            run_id=run_id,
            timestamp=datetime.datetime.now(),
            input_dataset_id=dataset_id,
            input_dataset_name_snapshot=name,
            params=params,
            centroids=KMeansCentroids(np.random.rand(k, 5).astype(np.float32)),
            effective_seed=effective_seed,
        )

    def test_empty_history_renders_no_rows(self):
        dialog = KMeansRunHistoryDialog(self.test_model.app_state)
        self.assertEqual(dialog._tbl_active.rowCount(), 0)
        self.assertEqual(dialog._tbl_closed.rowCount(), 0)

    def test_alive_vs_closed_split(self):
        app_state = self.test_model.app_state
        dataset = self.test_model.load_dataset(str(_JPL_HDR))
        history = app_state.get_kmeans_history()

        dialog = KMeansRunHistoryDialog(app_state)

        # Alive: references the loaded dataset.  Closed: references a
        # never-registered id, so its input dataset can't be resolved.
        history.add_record(
            self._make_record(1, dataset.get_id(), dataset.get_name(), seed=7, effective_seed=7)
        )
        history.add_record(self._make_record(2, 999_999, "ghost", seed=None, effective_seed=4242))
        self.test_model.app.processEvents()

        self.assertEqual(dialog._tbl_active.rowCount(), 1)
        self.assertEqual(dialog._tbl_closed.rowCount(), 1)

        # The seeded run shows its seed verbatim; the unseeded run flags the
        # auto-drawn seed but still shows the exact reproducible value.
        self.assertEqual(dialog._tbl_active.item(0, self._COL_SEED).text(), "7")
        closed_seed = dialog._tbl_closed.item(0, self._COL_SEED).text()
        self.assertIn("4242", closed_seed)
        self.assertIn("auto", closed_seed)

        self.assertIn("closed", dialog._tbl_closed.item(0, self._COL_STATUS).text().lower())
        self.assertEqual(dialog._tbl_active.item(0, self._COL_INIT).text(), "random")

    def test_delete_removes_row(self):
        app_state = self.test_model.app_state
        history = app_state.get_kmeans_history()
        dialog = KMeansRunHistoryDialog(app_state)

        history.add_record(self._make_record(1, 999_999, "ghost", seed=None, effective_seed=1))
        history.add_record(self._make_record(2, 999_998, "ghost2", seed=None, effective_seed=2))
        self.test_model.app.processEvents()
        self.assertEqual(dialog._tbl_closed.rowCount(), 2)

        history.remove_record(1)
        self.test_model.app.processEvents()
        self.assertEqual(dialog._tbl_closed.rowCount(), 1)


if __name__ == "__main__":
    unittest.main()
