import os
import tempfile
import time
import unittest
from concurrent.futures import BrokenExecutor, Future
from io import StringIO
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import patch

import numpy as np
import tests.context
# import context

from wiser.utils.primitives import (
    AllocationRequest,
    DataRef,
    DatasetRegionRef,
    DeletePolicy,
    DeletionState,
    PriorityClass,
    ProducerState,
)
from wiser.utils.storage_service import StorageService, shared_mem_exists
from wiser.utils.task_system import TaskPlan, WorkUnit
from wiser.utils.work_scheduler import (
    RecordingWorkScheduler,
    SchedulerConfig,
    WorkScheduler,
    _RestartableExecutor,
)

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


def _kill_worker_process() -> None:
    """Abruptly terminate the worker process to break the ProcessPoolExecutor.

    This simulates an OOM kill / segfault / native crash in a compute worker,
    which leaves the pool in a permanent `BrokenProcessPool` state.
    """
    os._exit(1)


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
    def test_recording_work_scheduler_prints_timing_summary(self) -> None:
        clock_times = iter([0.0, 1.0, 2.0, 5.5, 6.0, 7.5, 8.0])
        recorder = RecordingWorkScheduler(clock=lambda: next(clock_times))

        recorder.on_plan_submitted("plan-1")
        recorder.on_stage_enqueued("plan-1", "s00")
        recorder.on_unit_submitted("plan-1", "s00", "u1", "process", PriorityClass.INTERACTIVE)
        recorder.on_unit_done("plan-1", "s00", "u1", success=True)
        recorder.on_unit_submitted("plan-1", "s00", "u2", "thread", PriorityClass.BACKGROUND)
        recorder.on_unit_done("plan-1", "s00", "u2", success=True)
        recorder.on_plan_completed("plan-1", success=True)

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            recorder.print_timing_summary()

        self.assertEqual(
            stdout.getvalue(),
            "Plan plan-1 (8.000000s)\n"
            "  Stage s00 (6.500000s)\n"
            "    Unit u1: 3.500000s\n"
            "    Unit u2: 1.500000s\n",
        )

    def test_run_task_plan_reclaims_delete_when_releasable_outputs_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            scheduler = WorkScheduler(SchedulerConfig(_process_budget=3, _thread_budget=3), service)
            try:
                produced_ref = service.allocate_data(
                    AllocationRequest(
                        name="temporary_output",
                        kind="dataset",
                        residency="ram_cacheable",
                        size_est=np.dtype(np.float32).itemsize,
                        shape=(1, 1, 1),
                        dtype=np.dtype(np.float32),
                        delete_policy=DeletePolicy.DELETE_WHEN_RELEASABLE,
                    ),
                    owner_plan_id="plan-reclaim-success",
                    planned_consumer_plan_ids={"plan-reclaim-success"},
                )
                shared_mem_name = service._shared_mem_handles_names.get(produced_ref.uri)

                unit = _make_work_unit(
                    unit_id="s1_u1",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="thread",
                    fn=_ok_thread_a,
                )
                plan = TaskPlan(
                    plan_id="plan-reclaim-success",
                    semantic_task_id="semantic-reclaim-success",
                    work_units={unit.unit_id: unit},
                    stage_work_units={"s00": [unit.unit_id]},
                    produced_ref_ids={produced_ref.ref_id},
                )

                scheduler.run_task_plan(plan).result(timeout=5)

                record = service.get_lease_record(produced_ref.ref_id)
                self.assertEqual(record.producer_state, ProducerState.COMPLETED)
                self.assertEqual(record.deletion_state, DeletionState.DELETED)
                self.assertEqual(record.planned_consumer_plan_ids, set())
                self.assertNotIn(produced_ref.ref_id, service.data_refs)
                self.assertNotIn(produced_ref.ref_id, service.meta_by_ref)
                self.assertNotIn(produced_ref.ref_id, service.external_handles)
                self.assertNotIn(produced_ref.uri, service.ram_objects)
                self.assertNotIn(produced_ref.uri, service.ram_est_bytes)
                self.assertIsNotNone(shared_mem_name)
                self.assertFalse(shared_mem_exists(shared_mem_name))
            finally:
                scheduler.shutdown(wait=True)
                service.close()

    def test_stage_steps_enforce_barrier_between_stage_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            recorder = RecordingWorkScheduler()
            scheduler = WorkScheduler(
                SchedulerConfig(
                    _process_budget=3,
                    _thread_budget=3,
                    _process_priority_tokens={
                        PriorityClass.INTERACTIVE: 3,
                        PriorityClass.RENDER: 0,
                        PriorityClass.BACKGROUND: 0,
                    },
                ),
                service,
                recorder=recorder,
            )
            try:
                step1_u1 = _make_work_unit(
                    unit_id="step1_u1",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=partial(_sleep_then_return, "step1_u1", 0.3),
                )
                step1_u2 = _make_work_unit(
                    unit_id="step1_u2",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=partial(_sleep_then_return, "step1_u2", 0.3),
                )
                step2_u1 = _make_work_unit(
                    unit_id="step2_u1",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=partial(_sleep_then_return, "step2_u1", 0.01),
                )

                plan = TaskPlan(
                    plan_id="plan-stage-steps",
                    semantic_task_id="semantic-stage-steps",
                    work_units={
                        step1_u1.unit_id: step1_u1,
                        step1_u2.unit_id: step1_u2,
                        step2_u1.unit_id: step2_u1,
                    },
                    stage_work_units={
                        "s00": [step1_u1.unit_id, step1_u2.unit_id, step2_u1.unit_id],
                    },
                    stage_steps={
                        "s00": [[step1_u1.unit_id, step1_u2.unit_id], [step2_u1.unit_id]],
                    },
                    fail_fast=True,
                )

                completion = scheduler.run_task_plan(plan)
                completion.result(timeout=15)

                events = recorder.events
                done_index_by_unit = {
                    e.unit_id: idx
                    for idx, e in enumerate(events)
                    if e.kind == "unit_done" and e.plan_id == plan.plan_id
                }
                submit_index_by_unit = {
                    e.unit_id: idx
                    for idx, e in enumerate(events)
                    if e.kind == "unit_submitted" and e.plan_id == plan.plan_id
                }
                self.assertIn("step1_u1", done_index_by_unit)
                self.assertIn("step1_u2", done_index_by_unit)
                self.assertIn("step2_u1", submit_index_by_unit)
                self.assertGreater(
                    submit_index_by_unit["step2_u1"],
                    max(done_index_by_unit["step1_u1"], done_index_by_unit["step1_u2"]),
                )
            finally:
                scheduler.shutdown(wait=True)
                service.close()

    def test_queue_transition_log_shows_main_blocked_reserved_flow(self) -> None:
        """Verifies the full queue transition log for a unit that exhausts the RAM budget.

        Schedules 5 process units (u2 requires all 5000 bytes of RAM budget) so u2 initially
        fails the RAM gate and lands in the blocked queue. After exceeding the defer threshold,
        it is promoted to the reserved queue where it waits for exclusive RAM admission.
        Asserts exact to_queue, from_queue, and reason sequences across all five transitions.
        """
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
                self.assertEqual(u2_defer_counts, [0, 1, 3, 5, 0])
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
                completion.result(timeout=10)

                events = recorder.events
                self.assertTrue(events, "Expected scheduler to emit recorder events")

                self.assertEqual(events[0].kind, "plan_submitted")
                self.assertEqual(events[0].plan_id, plan.plan_id)
                self.assertIsInstance(events[0].time, float)

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
                produced_ref = service.allocate_data(
                    AllocationRequest(
                        name="failed_temporary_output",
                        kind="dataset",
                        residency="ram_cacheable",
                        size_est=np.dtype(np.float32).itemsize,
                        shape=(1, 1, 1),
                        dtype=np.dtype(np.float32),
                        delete_policy=DeletePolicy.DELETE_WHEN_RELEASABLE,
                    ),
                    owner_plan_id="plan-fail-fast",
                    planned_consumer_plan_ids={"plan-fail-fast"},
                )
                shared_mem_name = service._shared_mem_handles_names.get(produced_ref.uri)
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
                    produced_ref_ids={produced_ref.ref_id},
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

                record = service.get_lease_record(produced_ref.ref_id)
                self.assertEqual(record.producer_state, ProducerState.FAILED)
                self.assertEqual(record.deletion_state, DeletionState.DELETED)
                self.assertEqual(record.planned_consumer_plan_ids, set())
                self.assertNotIn(produced_ref.ref_id, service.data_refs)
                self.assertNotIn(produced_ref.ref_id, service.meta_by_ref)
                self.assertNotIn(produced_ref.ref_id, service.external_handles)
                self.assertNotIn(produced_ref.uri, service.ram_objects)
                self.assertNotIn(produced_ref.uri, service.ram_est_bytes)
                self.assertIsNotNone(shared_mem_name)
                self.assertFalse(shared_mem_exists(shared_mem_name))
            finally:
                scheduler.shutdown(wait=True)
                service.close()

    def test_run_task_plan_reclaims_delete_when_releasable_memmap_output_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            scheduler = WorkScheduler(SchedulerConfig(_process_budget=3, _thread_budget=3), service)
            try:
                produced_ref = service.allocate_data(
                    AllocationRequest(
                        name="temporary_memmap_output",
                        kind="dataset",
                        residency="spill_required",
                        size_est=np.dtype(np.float32).itemsize,
                        shape=(1, 1, 1),
                        dtype=np.dtype(np.float32),
                        delete_policy=DeletePolicy.DELETE_WHEN_RELEASABLE,
                    ),
                    preferred_storage="memmap",
                    owner_plan_id="plan-reclaim-memmap",
                    planned_consumer_plan_ids={"plan-reclaim-memmap"},
                )
                output_path = service._file_uri_to_path(produced_ref.uri)

                unit = _make_work_unit(
                    unit_id="s1_u1",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="thread",
                    fn=_ok_thread_a,
                )
                plan = TaskPlan(
                    plan_id="plan-reclaim-memmap",
                    semantic_task_id="semantic-reclaim-memmap",
                    work_units={unit.unit_id: unit},
                    stage_work_units={"s00": [unit.unit_id]},
                    produced_ref_ids={produced_ref.ref_id},
                )

                scheduler.run_task_plan(plan).result(timeout=5)

                record = service.get_lease_record(produced_ref.ref_id)
                self.assertEqual(record.producer_state, ProducerState.COMPLETED)
                self.assertEqual(record.deletion_state, DeletionState.DELETED)
                self.assertFalse(output_path.exists())
            finally:
                scheduler.shutdown(wait=True)
                service.close()

    def test_run_task_plan_reclaims_delete_when_releasable_zarr_output_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            scheduler = WorkScheduler(SchedulerConfig(_process_budget=3, _thread_budget=3), service)
            try:
                produced_ref = service.allocate_data(
                    AllocationRequest(
                        name="temporary_zarr_output",
                        kind="dataset",
                        residency="spill_required",
                        size_est=np.dtype(np.float32).itemsize,
                        shape=(1, 1, 1),
                        dtype=np.dtype(np.float32),
                        delete_policy=DeletePolicy.DELETE_WHEN_RELEASABLE,
                    ),
                    preferred_storage="zarr",
                    owner_plan_id="plan-reclaim-zarr",
                    planned_consumer_plan_ids={"plan-reclaim-zarr"},
                )
                store_path = service._zarr_uri_to_path(produced_ref.uri)

                unit = _make_work_unit(
                    unit_id="s1_u1",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="thread",
                    fn=_ok_thread_a,
                )
                plan = TaskPlan(
                    plan_id="plan-reclaim-zarr",
                    semantic_task_id="semantic-reclaim-zarr",
                    work_units={unit.unit_id: unit},
                    stage_work_units={"s00": [unit.unit_id]},
                    produced_ref_ids={produced_ref.ref_id},
                )

                scheduler.run_task_plan(plan).result(timeout=5)

                record = service.get_lease_record(produced_ref.ref_id)
                self.assertEqual(record.producer_state, ProducerState.COMPLETED)
                self.assertEqual(record.deletion_state, DeletionState.DELETED)
                self.assertFalse(store_path.exists())
            finally:
                scheduler.shutdown(wait=True)
                service.close()

    def test_run_task_plan_reclaims_delete_when_releasable_disk_json_output_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            scheduler = WorkScheduler(SchedulerConfig(_process_budget=3, _thread_budget=3), service)
            try:
                produced_ref = service.allocate_data(
                    AllocationRequest(
                        name="temporary_disk_json_output",
                        kind="json",
                        residency="spill_required",
                        size_est=1024,
                        delete_policy=DeletePolicy.DELETE_WHEN_RELEASABLE,
                    ),
                    preferred_storage="json",
                    owner_plan_id="plan-reclaim-disk-json",
                    planned_consumer_plan_ids={"plan-reclaim-disk-json"},
                )
                output_path = service._file_uri_to_path(produced_ref.uri)

                unit = _make_work_unit(
                    unit_id="s1_u1",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="thread",
                    fn=_ok_thread_a,
                )
                plan = TaskPlan(
                    plan_id="plan-reclaim-disk-json",
                    semantic_task_id="semantic-reclaim-disk-json",
                    work_units={unit.unit_id: unit},
                    stage_work_units={"s00": [unit.unit_id]},
                    produced_ref_ids={produced_ref.ref_id},
                )

                scheduler.run_task_plan(plan).result(timeout=5)

                record = service.get_lease_record(produced_ref.ref_id)
                self.assertEqual(record.producer_state, ProducerState.COMPLETED)
                self.assertEqual(record.deletion_state, DeletionState.DELETED)
                self.assertFalse(output_path.exists())
            finally:
                scheduler.shutdown(wait=True)
                service.close()

    def test_run_task_plan_reclaims_delete_when_releasable_ram_json_output_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            scheduler = WorkScheduler(SchedulerConfig(_process_budget=3, _thread_budget=3), service)
            try:
                produced_ref = service.allocate_data(
                    AllocationRequest(
                        name="temporary_ram_json_output",
                        kind="json",
                        residency="ram_cacheable",
                        size_est=1024,
                        delete_policy=DeletePolicy.DELETE_WHEN_RELEASABLE,
                    ),
                    owner_plan_id="plan-reclaim-ram-json",
                    planned_consumer_plan_ids={"plan-reclaim-ram-json"},
                )

                unit = _make_work_unit(
                    unit_id="s1_u1",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="thread",
                    fn=_ok_thread_a,
                )
                plan = TaskPlan(
                    plan_id="plan-reclaim-ram-json",
                    semantic_task_id="semantic-reclaim-ram-json",
                    work_units={unit.unit_id: unit},
                    stage_work_units={"s00": [unit.unit_id]},
                    produced_ref_ids={produced_ref.ref_id},
                )

                scheduler.run_task_plan(plan).result(timeout=5)

                record = service.get_lease_record(produced_ref.ref_id)
                self.assertEqual(record.producer_state, ProducerState.COMPLETED)
                self.assertEqual(record.deletion_state, DeletionState.DELETED)
                self.assertNotIn(produced_ref.ref_id, service.data_refs)
                self.assertNotIn(produced_ref.ref_id, service.meta_by_ref)
                self.assertNotIn(produced_ref.uri, service.ram_objects)
                self.assertNotIn(produced_ref.uri, service.ram_est_bytes)
            finally:
                scheduler.shutdown(wait=True)
                service.close()


class _FakeBreakingExecutor:
    """Test double that raises BrokenExecutor on its first submit, then works.

    Used to exercise `_RestartableExecutor`'s recovery path deterministically,
    without killing real worker processes/threads.
    """

    def __init__(self, *, breaks_on_first_submit: bool) -> None:
        self.breaks_on_first_submit = breaks_on_first_submit
        self.submit_count = 0
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def submit(self, fn, *args, **kwargs) -> Future:
        self.submit_count += 1
        if self.breaks_on_first_submit and self.submit_count == 1:
            raise BrokenExecutor("simulated broken pool")
        future: Future = Future()
        future.set_result(fn(*args, **kwargs))
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class TestRestartableExecutor(unittest.TestCase):
    """Unit tests for the generic self-healing executor wrapper (process + thread)."""

    def test_submit_restarts_once_on_broken_executor_and_retries(self) -> None:
        created: list[_FakeBreakingExecutor] = []

        def factory() -> _FakeBreakingExecutor:
            # Only the very first executor simulates a break.
            executor = _FakeBreakingExecutor(breaks_on_first_submit=len(created) == 0)
            created.append(executor)
            return executor

        pool = _RestartableExecutor("FakePool", factory)
        result = pool.submit(lambda value: value, "payload").result(timeout=5)

        self.assertEqual(result, "payload")
        self.assertEqual(len(created), 2, "the broken executor should have been replaced exactly once")
        self.assertIs(pool.executor, created[1])
        # The broken executor was torn down without blocking and cancelling backlog.
        self.assertEqual(created[0].shutdown_calls, [(False, True)])

    def test_submit_propagates_when_fresh_executor_also_breaks(self) -> None:
        # Every executor this factory produces breaks: recovery must give up
        # rather than loop forever.
        pool = _RestartableExecutor(
            "AlwaysBroken", lambda: _FakeBreakingExecutor(breaks_on_first_submit=True)
        )
        with self.assertRaises(BrokenExecutor):
            pool.submit(lambda: "never runs")


class TestWorkSchedulerBrokenPoolRecovery(unittest.TestCase):
    """End-to-end recovery when the scheduler's ProcessPoolExecutor breaks."""

    def test_recovers_after_worker_kills_process_pool(self) -> None:
        """A killed worker breaks the pool; the next plan must transparently restart it."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            scheduler = WorkScheduler(SchedulerConfig(_process_budget=3, _thread_budget=3), service)
            try:
                # Plan 1: a process unit that kills its worker, breaking the pool.
                kill_unit = _make_work_unit(
                    unit_id="kill_u1",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=_kill_worker_process,
                )
                kill_plan = TaskPlan(
                    plan_id="plan-kill",
                    semantic_task_id="semantic-kill",
                    work_units={kill_unit.unit_id: kill_unit},
                    stage_work_units={"s00": [kill_unit.unit_id]},
                )
                with self.assertRaises(Exception):
                    scheduler.run_task_plan(kill_plan).result(timeout=30)

                broken_executor = scheduler._process_pool.executor

                # Plan 2: ordinary process work that must succeed on a restarted pool.
                recover_unit = _make_work_unit(
                    unit_id="recover_u1",
                    stage_id="s00",
                    priority=PriorityClass.INTERACTIVE,
                    executor_kind="process",
                    fn=_ok_process_a,
                )
                recover_plan = TaskPlan(
                    plan_id="plan-recover",
                    semantic_task_id="semantic-recover",
                    work_units={recover_unit.unit_id: recover_unit},
                    stage_work_units={"s00": [recover_unit.unit_id]},
                )
                with self.assertLogs("wiser.utils.work_scheduler", level="WARNING") as log_ctx:
                    scheduler.run_task_plan(recover_plan).result(timeout=30)

                # The underlying pool was replaced and the restart was logged.
                self.assertIsNot(scheduler._process_pool.executor, broken_executor)
                self.assertTrue(
                    any("restarted with a fresh executor" in message for message in log_ctx.output)
                )
            finally:
                scheduler.shutdown(wait=True)
                service.close()


if __name__ == "__main__":
    unittest.main()
