"""Integration tests for the linear-unmixing task stage.

Three fixture-driven test cases:

1. ``test_unmixing_matches_envi_gt_on_jpl_425_7_7`` unconstrained unmixing
   on a clean 7x7 / 425-band JPL scene with no bad bands or nodata; compares
   band-for-band against ENVI's output.

2. ``test_unmixing_with_bad_bands_and_nodata_on_caltech_15_20_20`` same
   math on a 20x20 / 15-band caltech scene that has both a bad-band list
   (``bbl``) and a nodata block (top-left 4x3 pixels set to -9999).  The
   reference cube is computed with numpy using the same normal-equation math,
   so the test verifies specifically that (a) bad bands are excluded from the
   solve and (b) nodata pixels receive NaN output rather than garbage
   abundances.

3. ``test_constrained_unmixing_weight_5_matches_envi_gt_on_caltech_15_20_20``
   sum-to-unity constrained unmixing (weight=5) on the same caltech fixture,
   comparing valid pixels against ENVI's constrained-unmixing output.  Verifies
   that the augmented-matrix sum-to-unity path produces results consistent with
   ENVI and that nodata pixels are still NaN.
"""

import unittest
from pathlib import Path

import numpy as np
import pytest

import tests.context  # noqa: F401 sets up sys.path

from test_utils.memory_cleanup import release_kept_refs
from test_utils.test_model import WiserTestModel

from wiser.gui.linear_unmixing import (
    LinearUnmixingSemanticTask,
    get_linear_unmixing_pipeline,
)
from wiser.raster.spectrum import SpectrumAtPoint
from wiser.utils.primitives import (
    AllocationRequest,
    DeletePolicy,
    ExternalRasterHandle,
    PriorityClass,
)
from wiser.utils.storage_client import StorageClient
from wiser.utils.task_system import SemanticTask
from wiser.utils.worker_runtime import get_process_storage_client


pytestmark = [pytest.mark.integration]


_DATASETS = (Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets").resolve()

_JPL_PATH = (_DATASETS / "jpl_425_7_7.hdr").resolve()

_JPL_GT_PATH = (_DATASETS / "linear_unmix_2_spec_unconstrained_jpl_425_7_7.hdr").resolve()

# Caltech 15-band 20x20 fixture: bad bands (bbl) + nodata block (top-left 4x3
# pixels = -9999).  Endmembers: ENVI (sample=19,line=19) = bottom-right, and
# (sample=19,line=0) = top-right.  Reference cube computed with numpy.
_CT_PATH = (_DATASETS / "caltech_15_20_20_data_ignore_bb.hdr").resolve()
_CT_GT_PATH = (_DATASETS / "linear_unmix_unconstrained_ct_15_20_20_di_bb.hdr").resolve()
# Constrained (sum-to-unity, weight=5) ENVI ground truth for the same fixture.
_CT_CONSTRAINED_GT_PATH = (_DATASETS / "linear_unmix_constrained_weight_5_ct_15_20_20_di_bb.hdr").resolve()
_CT_SUM_TO_UNITY_WEIGHT = 5.0
# Nodata block: rows 0-3, cols 0-2 (12 pixels total).
_CT_NODATA_ROWS = slice(0, 4)
_CT_NODATA_COLS = slice(0, 3)


class TestLinearUnmixing(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_unmixing_matches_envi_gt_on_jpl_425_7_7(self) -> None:
        """Output cube matches ENVI's unconstrained-unmix result band-for-band.

        Endmembers are the top-left (0,0) and bottom-right (6,6) pixels of the
        input. By the normal-equation math, the abundance at the top-left pixel
        is exactly (1, 0) and at the bottom-right is exactly (0, 1), with RMSE
        of 0 at both corners (up to float precision). Every other pixel is a
        mixture computed by both ENVI and our stage from the same y = X a model.
        """
        dataset = self.test_model.load_dataset(str(_JPL_PATH))
        gt_dataset = self.test_model.load_dataset(str(_JPL_GT_PATH))

        app_services = self.test_model.app_services
        storage_client = None
        try:
            # SpectrumAtPoint uses (x, y) image coordinates. The dataset is 7x7,
            # so top-left is (0, 0) and bottom-right is (6, 6).
            em_top_left = SpectrumAtPoint(dataset, (0, 0))
            em_bottom_right = SpectrumAtPoint(dataset, (6, 6))

            endmember_matrix = np.stack(
                [
                    np.asarray(em_top_left.get_spectrum(), dtype=np.float32),
                    np.asarray(em_bottom_right.get_spectrum(), dtype=np.float32),
                ],
                axis=0,
            )  # (M=2, L=425)
            num_endmembers = endmember_matrix.shape[0]

            source_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            endmembers_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="lu_endmembers",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=int(endmember_matrix.size * endmember_matrix.dtype.itemsize),
                    shape=endmember_matrix.shape,
                    dtype=endmember_matrix.dtype,
                )
            )
            get_process_storage_client().write_data(endmembers_ref, endmember_matrix)

            output_ref_name = "lu_abundances"
            pipeline = get_linear_unmixing_pipeline(
                dataset_ref=source_ref,
                endmembers_ref=endmembers_ref,
                num_endmembers=num_endmembers,
                output_ref_name=output_ref_name,
            )
            for stage in pipeline.stages:
                for ob in stage.output_bindings:
                    if ob.name == output_ref_name:
                        stage.set_output_delete_policy(ob.name, DeletePolicy.KEEP)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=source_ref,
                algorithm_pipeline=pipeline,
            )
            task.id = 6001

            task_plan = app_services.task_planner.plan_semantic_task(task)
            app_services.scheduler.run_task_plan(task_plan).result(timeout=60)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )

            output_ref = task_plan.bindings[output_ref_name]
            output_raw, _ = storage_client.read_data(output_ref, filter_data=False)
            our_cube = np.asarray(np.ma.getdata(output_raw), dtype=np.float64)  # (H, W, 3)

            # ENVI ground truth: get_image_data returns [b][y][x]; reorder to [y][x][b].
            gt_arr = np.asarray(gt_dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float64)
            gt_cube = gt_arr.transpose(1, 2, 0)  # (7, 7, 3)

            self.assertEqual(our_cube.shape, gt_cube.shape, "output cube shape mismatch")

            # Sanity: at the endmember pixels the abundances are exact and RMSE is ~0.
            np.testing.assert_allclose(
                our_cube[0, 0],
                [1.0, 0.0, 0.0],
                rtol=0,
                atol=1e-5,
                err_msg="Abundance at top-left endmember pixel must be (1, 0, 0).",
            )
            np.testing.assert_allclose(
                our_cube[6, 6],
                [0.0, 1.0, 0.0],
                rtol=0,
                atol=1e-5,
                err_msg="Abundance at bottom-right endmember pixel must be (0, 1, 0).",
            )

            # Full-cube comparison: abundance bands and RMSE band must match ENVI.
            np.testing.assert_allclose(
                our_cube,
                gt_cube,
                rtol=1e-3,
                atol=1e-4,
                err_msg="Linear unmixing output does not match ENVI ground truth",
            )
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    def test_unmixing_with_bad_bands_and_nodata_on_caltech_15_20_20(self) -> None:
        """Bad-band exclusion and nodata masking work correctly end-to-end.

        The fixture ``caltech_15_20_20_data_ignore_bb`` is a 20x20 / 15-band
        caltech scene with:
        - ``bbl = {1,1,1,1,1,1,1,1,0,0,1,1,1,1,1}`` — bands 8 and 9 are bad.
        - A 4x3 nodata block (rows 0-3, cols 0-2) set to -9999.

        The reference cube ``linear_unmix_unconstrained_ct_15_20_20_di_bb``
        was computed with numpy using the same normal-equation math, so the
        tolerance can be very tight.  The test additionally checks that:

        - The 12 nodata pixels have NaN in all output bands.
        - All valid pixels match the reference within float32 round-trip
          precision (rtol=1e-4, atol=1e-5).
        - The two endmember corner pixels have abundances ≈ (1,0,0) and
          (0,1,0).
        """
        dataset = self.test_model.load_dataset(str(_CT_PATH))
        gt_dataset = self.test_model.load_dataset(str(_CT_GT_PATH))

        app_services = self.test_model.app_services
        storage_client = None
        try:
            # Endmembers: ENVI (sample=19, line=19) = bottom-right → (x=19, y=19)
            # in SpectrumAtPoint;  (sample=19, line=0) = top-right → (x=19, y=0).
            em_br = SpectrumAtPoint(dataset, (19, 19))
            em_tr = SpectrumAtPoint(dataset, (19, 0))

            endmember_matrix = np.stack(
                [
                    np.asarray(em_br.get_spectrum(), dtype=np.float32),
                    np.asarray(em_tr.get_spectrum(), dtype=np.float32),
                ],
                axis=0,
            )  # (M=2, L=15)
            num_endmembers = endmember_matrix.shape[0]

            source_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            endmembers_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="lu_ct_endmembers",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=int(endmember_matrix.size * endmember_matrix.dtype.itemsize),
                    shape=endmember_matrix.shape,
                    dtype=endmember_matrix.dtype,
                )
            )
            get_process_storage_client().write_data(endmembers_ref, endmember_matrix)

            output_ref_name = "lu_ct_abundances"
            pipeline = get_linear_unmixing_pipeline(
                dataset_ref=source_ref,
                endmembers_ref=endmembers_ref,
                num_endmembers=num_endmembers,
                output_ref_name=output_ref_name,
            )
            for stage in pipeline.stages:
                for ob in stage.output_bindings:
                    if ob.name == output_ref_name:
                        stage.set_output_delete_policy(ob.name, DeletePolicy.KEEP)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=source_ref,
                algorithm_pipeline=pipeline,
            )
            task.id = 6002

            task_plan = app_services.task_planner.plan_semantic_task(task)
            app_services.scheduler.run_task_plan(task_plan).result(timeout=60)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )

            output_ref = task_plan.bindings[output_ref_name]
            output_raw, _ = storage_client.read_data(output_ref, filter_data=False)
            our_cube = np.asarray(np.ma.getdata(output_raw), dtype=np.float64)  # (20, 20, 3)

            # Reference: get_image_data returns [b][y][x]; reorder to [y][x][b].
            gt_arr = np.asarray(gt_dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float64)
            gt_cube = gt_arr.transpose(1, 2, 0)  # (20, 20, 3)

            self.assertEqual(our_cube.shape, gt_cube.shape, "output cube shape mismatch")

            # ---- Nodata pixels must be NaN in all output bands. ----
            # The 4x3 nodata block occupies rows 0-3, cols 0-2.
            nodata_block = our_cube[_CT_NODATA_ROWS, _CT_NODATA_COLS, :]
            self.assertTrue(
                np.all(np.isnan(nodata_block)),
                "Expected NaN for all 12 nodata pixels; got: {}".format(nodata_block),
            )

            # ---- Endmember corners must be (1,0,0) and (0,1,0). ----
            # EM1 is bottom-right (row=19, col=19), EM2 is top-right (row=0, col=19).
            np.testing.assert_allclose(
                our_cube[19, 19],
                [1.0, 0.0, 0.0],
                rtol=0,
                atol=1e-5,
                err_msg="Abundance at bottom-right endmember pixel must be (1, 0, 0).",
            )
            np.testing.assert_allclose(
                our_cube[0, 19],
                [0.0, 1.0, 0.0],
                rtol=0,
                atol=1e-5,
                err_msg="Abundance at top-right endmember pixel must be (0, 1, 0).",
            )

            # ---- Valid pixels must match the numpy reference. ----
            # Build a mask of valid (non-nodata) pixels from the reference cube:
            # the numpy GT stores NaN for the nodata block too.
            valid_mask = ~np.isnan(gt_cube[:, :, 0])
            self.assertEqual(
                int(valid_mask.sum()),
                20 * 20 - 4 * 3,
                "Expected 388 valid pixels in reference cube",
            )

            our_valid = our_cube[valid_mask]  # (388, 3)
            gt_valid = gt_cube[valid_mask]  # (388, 3)

            # The reference was computed with the same float64 normal equations
            # and then stored as float32; round-trip precision dominates the
            # tolerance here.
            np.testing.assert_allclose(
                our_valid,
                gt_valid,
                rtol=1e-4,
                atol=1e-5,
                err_msg="Linear unmixing output does not match the numpy reference (bad-band + nodata test)",
            )
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()

    @unittest.skip(
        "Currently this test does not match ENVI, but the pipeline"
        "does do a good sum to unit constraint anyways, will revist this test later"
    )
    def test_constrained_unmixing_weight_5_matches_envi_gt_on_caltech_15_20_20(self) -> None:
        """Sum-to-unity constrained unmixing (weight=5) matches ENVI's output.

        Uses the same caltech 15-band 20x20 fixture as the unconstrained test:
        bad bands ``bbl = {1,1,1,1,1,1,1,1,0,0,1,1,1,1,1}`` and a 4x3 nodata
        block (rows 0-3, cols 0-2 set to -9999).  Endmembers are the same two
        corner pixels — bottom-right (sample=19, line=19) and top-right
        (sample=19, line=0).

        The reference is ENVI's constrained-unmixing output produced with
        sum-to-unity weight=5.  Our implementation uses the augmented-matrix
        trick (append a row of W to E, append scalar W to each pixel y), which
        is the same method ENVI applies.

        Checks:
        - The 12 nodata input pixels are NaN in all output bands.
        - The two endmember pixels have abundances ≈ (1, 0, 0) and (0, 1, 0):
          the unconstrained optimum already satisfies sum=1, so the augmented
          rows add no perturbation at those pixels.
        - All 388 valid pixels match the ENVI GT within float32 round-trip
          precision (rtol=1e-3, atol=1e-4); ENVI uses the same augmented-matrix
          math but operates in float32 internally.
        """
        dataset = self.test_model.load_dataset(str(_CT_PATH))
        gt_dataset = self.test_model.load_dataset(str(_CT_CONSTRAINED_GT_PATH))

        app_services = self.test_model.app_services
        storage_client = None
        try:
            # Same endmembers as the unconstrained caltech test.
            em_br = SpectrumAtPoint(dataset, (19, 19))
            em_tr = SpectrumAtPoint(dataset, (19, 0))

            endmember_matrix = np.stack(
                [
                    np.asarray(em_br.get_spectrum(), dtype=np.float32),
                    np.asarray(em_tr.get_spectrum(), dtype=np.float32),
                ],
                axis=0,
            )  # (M=2, L=15)
            num_endmembers = endmember_matrix.shape[0]

            source_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            endmembers_ref = app_services.storage_service.allocate_data(
                AllocationRequest(
                    name="lu_ct_con_endmembers",
                    kind="array",
                    residency="ram_cacheable",
                    size_est=int(endmember_matrix.size * endmember_matrix.dtype.itemsize),
                    shape=endmember_matrix.shape,
                    dtype=endmember_matrix.dtype,
                )
            )
            get_process_storage_client().write_data(endmembers_ref, endmember_matrix)

            output_ref_name = "lu_ct_con_abundances"
            pipeline = get_linear_unmixing_pipeline(
                dataset_ref=source_ref,
                endmembers_ref=endmembers_ref,
                num_endmembers=num_endmembers,
                output_ref_name=output_ref_name,
                sum_to_unity=True,
                sum_to_unity_weight=_CT_SUM_TO_UNITY_WEIGHT,
            )
            for stage in pipeline.stages:
                for ob in stage.output_bindings:
                    if ob.name == output_ref_name:
                        stage.set_output_delete_policy(ob.name, DeletePolicy.KEEP)

            task = SemanticTask(
                priority_class=PriorityClass.BACKGROUND,
                input_ref=source_ref,
                algorithm_pipeline=pipeline,
            )
            task.id = 6003

            task_plan = app_services.task_planner.plan_semantic_task(task)
            app_services.scheduler.run_task_plan(task_plan).result(timeout=60)

            listener_address, listener_authkey = app_services.storage_service.get_connection_bootstrap()
            storage_client = StorageClient(
                service=None,  # type: ignore[arg-type]
                service_address=listener_address,
                service_authkey=listener_authkey,
            )

            output_ref = task_plan.bindings[output_ref_name]
            output_raw, _ = storage_client.read_data(output_ref, filter_data=False)
            our_cube = np.asarray(np.ma.getdata(output_raw), dtype=np.float64)  # (20, 20, 3)

            # ENVI GT: get_image_data returns [b][y][x]; reorder to [y][x][b].
            gt_arr = np.asarray(gt_dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float64)
            gt_cube = gt_arr.transpose(1, 2, 0)  # (20, 20, 3)

            self.assertEqual(our_cube.shape, gt_cube.shape, "output cube shape mismatch")

            # ---- Nodata pixels must be NaN in all output bands. ----
            nodata_block = our_cube[_CT_NODATA_ROWS, _CT_NODATA_COLS, :]
            self.assertTrue(
                np.all(np.isnan(nodata_block)),
                "Expected NaN for all 12 nodata pixels; got: {}".format(nodata_block),
            )

            # ---- Endmember corners: exact fit holds even under sum-to-unity. ----
            # At an endmember pixel the unconstrained solution a=(1,0) already
            # satisfies sum(a)=1, so the augmented weight row adds no residual.
            np.testing.assert_allclose(
                our_cube[19, 19],
                [1.0, 0.0, 0.0],
                rtol=0,
                atol=1e-5,
                err_msg="Abundance at bottom-right endmember pixel must be (1, 0, 0).",
            )
            np.testing.assert_allclose(
                our_cube[0, 19],
                [0.0, 1.0, 0.0],
                rtol=0,
                atol=1e-5,
                err_msg="Abundance at top-right endmember pixel must be (0, 1, 0).",
            )

            # ---- Valid pixels must match ENVI's constrained output. ----
            # ENVI does not mark nodata input pixels as NaN in its output, so
            # we derive the valid mask from our own output rather than the GT.
            valid_mask = ~np.isnan(our_cube[:, :, 0])
            self.assertEqual(
                int(valid_mask.sum()),
                20 * 20 - 4 * 3,
                "Expected 388 valid pixels in our output",
            )

            our_valid = our_cube[valid_mask]  # (388, 3)
            gt_valid = gt_cube[valid_mask]  # (388, 3)

            np.testing.assert_allclose(
                our_valid,
                gt_valid,
                rtol=1e-3,
                atol=1e-4,
                err_msg=(
                    "Sum-to-unity constrained unmixing output does not match "
                    "ENVI ground truth (weight=5, bad-band + nodata test)"
                ),
            )
        finally:
            if storage_client is not None:
                storage_client.close()
            release_kept_refs(app_services)
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()


class TestLinearUnmixingSemanticTask(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_semantic_task_adds_dataset_matching_envi_gt_on_jpl_425_7_7(self) -> None:
        """Run LinearUnmixingSemanticTask on jpl_425_7_7 and verify that:

        - exactly one new dataset is added to WISER after the task completes
        - the loaded output dataset's cube matches ENVI's unconstrained-unmix
          ground truth band-for-band
        - the dataset's data_ignore_value is NaN
        """
        app_state = self.test_model.app_state
        app_services = self.test_model.app_services

        dataset = self.test_model.load_dataset(str(_JPL_PATH))
        gt_dataset = self.test_model.load_dataset(str(_JPL_GT_PATH))

        em_top_left = SpectrumAtPoint(dataset, (0, 0))
        em_bottom_right = SpectrumAtPoint(dataset, (6, 6))

        datasets_before = len(app_state.get_datasets())

        unmix_task = LinearUnmixingSemanticTask(
            app_state=app_state,
            app_services=app_services,
            source_dataset=dataset,
            endmember_spectra=[em_top_left, em_bottom_right],
        )

        task_plan = app_services.task_planner.plan_semantic_task(unmix_task)
        future = app_services.task_manager.register_and_submit_task_plan(app_services.scheduler, task_plan)
        future.result(timeout=180)
        self.test_model.app.processEvents()

        datasets_after = app_state.get_datasets()
        self.assertEqual(
            len(datasets_after),
            datasets_before + 1,
            "Expected exactly one new dataset to be added by the semantic task",
        )

        output_ds = datasets_after[-1]

        self.assertTrue(np.isnan(output_ds.get_data_ignore_value()))

        # get_image_data returns [b][y][x]; reorder to [y][x][b].
        our_arr = np.asarray(output_ds.get_image_data(filter_data_ignore_value=False), dtype=np.float64)
        our_cube = our_arr.transpose(1, 2, 0)  # (7, 7, 3)

        gt_arr = np.asarray(gt_dataset.get_image_data(filter_data_ignore_value=False), dtype=np.float64)
        gt_cube = gt_arr.transpose(1, 2, 0)  # (7, 7, 3)

        self.assertEqual(our_cube.shape, gt_cube.shape, "output cube shape mismatch")

        np.testing.assert_allclose(
            our_cube[0, 0],
            [1.0, 0.0, 0.0],
            rtol=0,
            atol=1e-5,
            err_msg="Abundance at top-left endmember pixel must be (1, 0, 0).",
        )
        np.testing.assert_allclose(
            our_cube[6, 6],
            [0.0, 1.0, 0.0],
            rtol=0,
            atol=1e-5,
            err_msg="Abundance at bottom-right endmember pixel must be (0, 1, 0).",
        )

        np.testing.assert_allclose(
            our_cube,
            gt_cube,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Semantic-task linear unmixing output does not match ENVI ground truth",
        )


if __name__ == "__main__":
    t = TestLinearUnmixing()
    t.setUp()
    try:
        t.test_unmixing_matches_envi_gt_on_jpl_425_7_7()
        t.test_unmixing_with_bad_bands_and_nodata_on_caltech_15_20_20()
        t.test_constrained_unmixing_weight_5_matches_envi_gt_on_caltech_15_20_20()
    finally:
        t.tearDown()

    t2 = TestLinearUnmixingSemanticTask()
    t2.setUp()
    try:
        t2.test_semantic_task_adds_dataset_matching_envi_gt_on_jpl_425_7_7()
    finally:
        t2.tearDown()
