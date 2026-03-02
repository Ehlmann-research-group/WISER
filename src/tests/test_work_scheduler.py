import tempfile
import time
import unittest
from functools import partial

import numpy as np
import tests.context
# import context

from wiser.utils.primitives import DataRef, DatasetRegionRef, PriorityClass
from wiser.utils.storage_service import StorageService
from wiser.utils.task_system import TaskPlan, WorkUnit
from wiser.utils.work_scheduler import RecordingWorkScheduler, SchedulerConfig, WorkScheduler

import pytest

pytestmark = [
    pytest.mark.scheduler,
]


def _ok_process_a() -> str:
    return "ok-process-a"


def _ok_process_b() -> str:
    return "ok-process-b"


def _ok_process_c() -> str:
    return "ok-process-c"


def _ok_thread_a() -> str:
    return "ok-thread-a"


def _sleep_then_return(label: str, sleep_seconds: float) -> str:
    time.sleep(sleep_seconds)
    return label


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
    ram_peak_est_bytes: int = 1,
) -> WorkUnit:
    return WorkUnit(
        unit_id=unit_id,
        stage_id=stage_id,
        priority_class=priority,
        executor_kind=executor_kind,  # type: ignore[arg-type]
        fn=fn,
        ram_peak_est_bytes=ram_peak_est_bytes,
    )


class TestWorkScheduler(unittest.TestCase):
    def test_queue_transition_log_shows_main_blocked_reserved_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            scheduler = WorkScheduler(
                SchedulerConfig(
                    _process_budget=7,
                    _thread_budget=3,
                    _ram_budget=5_000,
                    _defer_to_reserved_threshold=2,
                    _process_priority_tokens={
                        PriorityClass.INTERACTIVE: 5,
                        PriorityClass.RENDER: 1,
                        PriorityClass.BACKGROUND: 1,
                    },
                    _thread_priority_tokens={
                        PriorityClass.INTERACTIVE: 1,
                        PriorityClass.RENDER: 1,
                        PriorityClass.BACKGROUND: 1,
                    },
                ),
                service,
            )
            try:
                # Required stage order by RAM size: 1000, 5000, 1000, 1000.
                u1 = _make_work_unit(
                    unit_id="u1_1000",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=partial(_sleep_then_return, "u1", 1),
                    ram_peak_est_bytes=1_000,
                )
                u2 = _make_work_unit(
                    unit_id="u2_5000",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=partial(_sleep_then_return, "u2", 0.1),
                    ram_peak_est_bytes=5_000,
                )
                u3 = _make_work_unit(
                    unit_id="u3_1000",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=partial(_sleep_then_return, "u3", 1),
                    ram_peak_est_bytes=1_000,
                )
                u4 = _make_work_unit(
                    unit_id="u4_1000",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=partial(_sleep_then_return, "u4", 1),
                    ram_peak_est_bytes=1_000,
                )

                u5 = _make_work_unit(
                    unit_id="u5_1000",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=partial(_sleep_then_return, "u5", 1),
                    ram_peak_est_bytes=1_000,
                )

                plan = TaskPlan(
                    plan_id="plan-blocked-reserved",
                    semantic_task_id="semantic-blocked-reserved",
                    work_units={
                        u1.unit_id: u1,
                        u2.unit_id: u2,
                        u3.unit_id: u3,
                        u4.unit_id: u4,
                        u5.unit_id: u5,
                    },
                    stage_work_units={"s00": [u1.unit_id, u2.unit_id, u3.unit_id, u4.unit_id, u5.unit_id]},
                    fail_fast=True,
                )

                completion = scheduler.run_task_plan(plan)
                completion.result(timeout=30)

                u2_events = scheduler.get_queue_transition_log_for_unit("u2_5000")

                u2_to_queues = [event.to_queue for event in u2_events]
                self.assertEqual(
                    u2_to_queues,
                    [
                        "main:process:interactive",
                        "blocked:process:interactive",
                        "reserved:interactive",
                        "in_flight:process",
                        "done",
                    ],
                )

                u2_reasons = [event.reason for event in u2_events]
                self.assertEqual(
                    u2_reasons,
                    [
                        "stage_enqueued",
                        "ram_gate_failed",
                        "defer_threshold_exceeded",
                        "reserved_admitted",
                        "unit_succeeded",
                    ],
                )

                u2_from_queues = [event.from_queue for event in u2_events]
                self.assertEqual(
                    u2_from_queues,
                    [
                        None,
                        "main:process:interactive",
                        "blocked:process:interactive",
                        "reserved:interactive",
                        "in_flight:process",
                    ],
                )

                u2_defer_counts = [event.defer_count for event in u2_events]
                self.assertEqual(u2_defer_counts, [0, 1, 3, 3, 0])
            finally:
                scheduler.shutdown(wait=True)
                service.close()

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
    test_work_scheduler.test_queue_transition_log_shows_main_blocked_reserved_flow()
    # test_work_scheduler.test_run_task_plan_fail_fast_stops_before_stage_2()
