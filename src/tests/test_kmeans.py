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

            params = KMeansParameters(
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


if __name__ == "__main__":
    unittest.main()
