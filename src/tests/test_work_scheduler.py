import tempfile
import unittest

import numpy as np
import tests.context
# import context

from wiser.utils.primitives import DataRef, DatasetRegionRef, PriorityClass
from wiser.utils.storage_service import StorageService
from wiser.utils.task_system import TaskPlan, WorkUnit
from wiser.utils.work_scheduler import RecordingWorkScheduler, SchedulerConfig, WorkScheduler


def _ok_process_a() -> str:
    return "ok-process-a"


def _ok_process_b() -> str:
    return "ok-process-b"


def _ok_process_c() -> str:
    return "ok-process-c"


def _ok_thread_a() -> str:
    return "ok-thread-a"


def _boom_process() -> None:
    raise RuntimeError("boom")


def _make_input_ref(ref_id: str) -> DataRef:
    return DataRef(
        kind="dataset",
        ref_id=ref_id,
        uri=f"mem://{ref_id}",
        disk_format=None,
        shape=(1, 1, 1),
        dtype=np.dtype(np.float32),
        chunks=None,
        residency="ram_cacheable",
        materialization_loc="ram",
        source="internal",
        readonly=False,
    )


def _make_work_unit(
    *,
    unit_id: str,
    stage_id: str,
    priority: PriorityClass,
    executor_kind: str,
    fn,
) -> WorkUnit:
    return WorkUnit(
        unit_id=unit_id,
        stage_id=stage_id,
        priority_class=priority,
        executor_kind=executor_kind,  # type: ignore[arg-type]
        input_ref=_make_input_ref(f"in-{unit_id}"),
        input_region=DatasetRegionRef(y0=0, y1=1, x0=0, x1=1, b0=0, b1=1),
        writes=(),
        fn=fn,
        broadcast={},
        ram_peak_est_bytes=1,
    )


class TestWorkScheduler(unittest.TestCase):
    def test_run_task_plan_two_stages_records_expected_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            recorder = RecordingWorkScheduler()
            scheduler = WorkScheduler(
                SchedulerConfig(_process_budget=3, _thread_budget=3),
                service,
                recorder=recorder,
            )
            try:
                s1_u1 = _make_work_unit(
                    unit_id="s1_u1",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=_ok_process_a,
                )
                s1_u2 = _make_work_unit(
                    unit_id="s1_u2",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=_ok_process_b,
                )
                s2_u1 = _make_work_unit(
                    unit_id="s2_u1",
                    stage_id="s01",
                    priority=PriorityClass.RENDER,
                    executor_kind="process",
                    fn=_ok_process_c,
                )
                s2_u2 = _make_work_unit(
                    unit_id="s2_u2",
                    stage_id="s01",
                    priority=PriorityClass.BACKGROUND,
                    executor_kind="thread",
                    fn=_ok_thread_a,
                )

                plan = TaskPlan(
                    plan_id="plan-success",
                    semantic_task_id="semantic-success",
                    work_units={
                        s1_u1.unit_id: s1_u1,
                        s1_u2.unit_id: s1_u2,
                        s2_u1.unit_id: s2_u1,
                        s2_u2.unit_id: s2_u2,
                    },
                    stage_work_units={
                        "s00": [s1_u1.unit_id, s1_u2.unit_id],
                        "s01": [s2_u1.unit_id, s2_u2.unit_id],
                    },
                    fail_fast=True,
                )

                completion = scheduler.run_task_plan(plan)
                completion.result(timeout=5)

                events = recorder.events
                self.assertTrue(events, "Expected scheduler to emit recorder events")

                self.assertEqual(events[0].kind, "plan_submitted")
                self.assertEqual(events[0].plan_id, plan.plan_id)

                s00_enqueue_idxs = [
                    idx
                    for idx, e in enumerate(events)
                    if e.kind == "stage_enqueued" and e.plan_id == plan.plan_id and e.stage_id == "s00"
                ]
                s01_enqueue_idxs = [
                    idx
                    for idx, e in enumerate(events)
                    if e.kind == "stage_enqueued" and e.plan_id == plan.plan_id and e.stage_id == "s01"
                ]
                self.assertTrue(s00_enqueue_idxs, "Expected at least one s00 enqueue event")
                self.assertTrue(s01_enqueue_idxs, "Expected at least one s01 enqueue event")
                self.assertLess(max(s00_enqueue_idxs), min(s01_enqueue_idxs))

                s1_done_idxs = [
                    idx
                    for idx, e in enumerate(events)
                    if e.kind == "unit_done" and e.stage_id == "s00" and e.plan_id == plan.plan_id
                ]
                self.assertEqual(len(s1_done_idxs), 2)
                first_s2_submit_idx = next(
                    idx
                    for idx, e in enumerate(events)
                    if e.kind == "unit_submitted" and e.stage_id == "s01" and e.plan_id == plan.plan_id
                )
                self.assertLess(max(s1_done_idxs), first_s2_submit_idx)

                submitted_units = [
                    e.unit_id for e in events if e.kind == "unit_submitted" and e.plan_id == plan.plan_id
                ]
                done_units = [
                    e.unit_id for e in events if e.kind == "unit_done" and e.plan_id == plan.plan_id
                ]
                expected_ids = {"s1_u1", "s1_u2", "s2_u1", "s2_u2"}
                self.assertEqual(set(submitted_units), expected_ids)
                self.assertEqual(set(done_units), expected_ids)
                self.assertEqual(len(submitted_units), 4)
                self.assertEqual(len(done_units), 4)

                plan_completed_events = [
                    e for e in events if e.kind == "plan_completed" and e.plan_id == plan.plan_id
                ]
                self.assertEqual(len(plan_completed_events), 1)
                self.assertTrue(plan_completed_events[0].success)
            finally:
                scheduler.shutdown(wait=True)
                service.close()

    def test_run_task_plan_fail_fast_stops_before_stage_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            recorder = RecordingWorkScheduler()
            scheduler = WorkScheduler(
                SchedulerConfig(_process_budget=3, _thread_budget=3),
                service,
                recorder=recorder,
            )
            try:
                failing = _make_work_unit(
                    unit_id="s1_fail",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=_boom_process,
                )
                queued = _make_work_unit(
                    unit_id="s1_queued",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=_ok_process_a,
                )
                s2_process = _make_work_unit(
                    unit_id="s2_proc",
                    stage_id="s01",
                    priority=PriorityClass.RENDER,
                    executor_kind="process",
                    fn=_ok_process_b,
                )
                s2_thread = _make_work_unit(
                    unit_id="s2_thread",
                    stage_id="s01",
                    priority=PriorityClass.BACKGROUND,
                    executor_kind="thread",
                    fn=_ok_thread_a,
                )

                plan = TaskPlan(
                    plan_id="plan-fail-fast",
                    semantic_task_id="semantic-fail-fast",
                    work_units={
                        failing.unit_id: failing,
                        queued.unit_id: queued,
                        s2_process.unit_id: s2_process,
                        s2_thread.unit_id: s2_thread,
                    },
                    stage_work_units={
                        "s00": [failing.unit_id, queued.unit_id],
                        "s01": [s2_process.unit_id, s2_thread.unit_id],
                    },
                    fail_fast=True,
                )

                completion = scheduler.run_task_plan(plan)
                with self.assertRaises(RuntimeError):
                    completion.result(timeout=15)

                events = recorder.events
                self.assertTrue(events, "Expected scheduler to emit recorder events")

                stage_2_enqueued = any(
                    e.kind == "stage_enqueued" and e.plan_id == plan.plan_id and e.stage_id == "s01"
                    for e in events
                )
                self.assertFalse(stage_2_enqueued)

                stage_2_submitted = any(
                    e.kind == "unit_submitted" and e.plan_id == plan.plan_id and e.stage_id == "s01"
                    for e in events
                )
                self.assertFalse(stage_2_submitted)

                plan_completed_events = [
                    e for e in events if e.kind == "plan_completed" and e.plan_id == plan.plan_id
                ]
                self.assertEqual(len(plan_completed_events), 1)
                self.assertFalse(plan_completed_events[0].success)
            finally:
                scheduler.shutdown(wait=True)
                service.close()


if __name__ == "__main__":
    test_work_scheduler = TestWorkScheduler()
    test_work_scheduler.test_run_task_plan_two_stages_records_expected_events()
    # test_work_scheduler.test_run_task_plan_fail_fast_stops_before_stage_2()
