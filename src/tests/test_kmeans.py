import unittest
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKMeans

import tests.context  # noqa: F401 – adds src/ to sys.path

from test_utils.memory_cleanup import release_kept_refs
from test_utils.test_model import WiserTestModel
from wiser.gui.app_services import AppServices
from wiser.gui.kmeans import (
    KMeansAlgorithm,
    KMeansInitMethod,
    KMeansParameters,
    KMeansSemanticTask,
    get_kmeans_pipeline,
)
from wiser.raster.loader import RasterDataLoader
from wiser.utils.primitives import DeletePolicy, ExternalRasterHandle, PriorityClass
from wiser.utils.storage_client import StorageClient
from wiser.utils.task_system import SemanticTask

pytestmark = [
    pytest.mark.integration,
]

_JPL_HDR = Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_425_7_7.hdr"

_K = 5
_SEED = 42


def _print_test_start(case: unittest.TestCase) -> None:
    print(f"\nRunning test: {case.id()}", flush=True)


class TestKMeansStage(unittest.TestCase):
    def setUp(self):
        _print_test_start(self)
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

            # ------------------------------------------------------------------
            # Assertions
            # ------------------------------------------------------------------
            labels_array = np.asarray(labels_out)
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
        _print_test_start(self)
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

        # Verify centroids were stored in app_state under the correct key
        stored_centroids = app_state.get_kmeans_centroids(params)
        self.assertIsNotNone(stored_centroids, "KMeansCentroids were not stored in app_state")
        self.assertEqual(stored_centroids.num_centroids(), _K)
        ref_centroids = ref_kmeans.cluster_centers_.astype(np.float32)
        np.testing.assert_allclose(
            stored_centroids._centroids,
            ref_centroids,
            atol=1e-5,
            err_msg="Stored centroids do not match sklearn reference.",
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
        _print_test_start(self)
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _submit_task(self, dataset, params):
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
        centroids = app_state.get_kmeans_centroids(params)
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


if __name__ == "__main__":
    unittest.main()
