from __future__ import annotations

import logging
import multiprocessing
import os
import sys

from concurrent.futures import (
    BrokenExecutor,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from dataclasses import dataclass, field
from collections import Counter, deque
from threading import Lock, Semaphore
from time import perf_counter
from typing import Any, Callable, Deque, Dict, Optional, TYPE_CHECKING
from multiprocessing.managers import dispatch

from matplotlib.pylab import Enum

from .primitives import PriorityClass
from .task_system import TaskPlan, WorkUnit
from .worker_runtime import initialize_process_storage_client, initialize_thread_worker

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from wiser.utils.storage_service import StorageService
    from wiser.utils.task_system import TaskManager

available_cpus = os.cpu_count() or 1
SCHEDULER_PROCESS_BUDGET = min(12, available_cpus)
SCHEDULER_RAM_BUDGET = 2_000_000_000
SCHEDULER_THREAD_BUDGET = 32
SCHEDULER_DEFER_TO_RESERVED_THRESHOLD = 4
PRIORITY_LANE_COUNT = 3
# Max times a reserved unit may fail RAM admission before its plan is aborted.
# Prevents indefinite hangs when a unit's `ram_peak_est_bytes` exceeds the
# scheduler's RAM budget (or otherwise cannot ever be admitted).
MAX_DEFER_COUNT = 42

# The below should add up to 1.0. Currently, we don't have any interactive
# or render budgets so we put most into background.
INTERACTIVE_EXECUTOR_BUDGET_WEIGHT = 0.125
RENDER_EXECUTOR_BUDGET_WEIGHT = 0.125
BACKGROUND_EXECUTOR_BUDGET_WEIGHT = 0.75


@dataclass
class SchedulerConfig:
    """
    Specifies the computer resources that we can use.
    """

    _process_budget: int = SCHEDULER_PROCESS_BUDGET
    _thread_budget: int = SCHEDULER_THREAD_BUDGET
    _ram_budget: int = SCHEDULER_RAM_BUDGET
    _defer_to_reserved_threshold: int = SCHEDULER_DEFER_TO_RESERVED_THRESHOLD
    _process_priority_tokens: Optional[Dict[PriorityClass, int]] = None
    _thread_priority_tokens: Optional[Dict[PriorityClass, int]] = None


@dataclass
class QueuedWorkUnit:
    plan_id: str
    stage_id: str
    work_unit: WorkUnit
    # Number of times this unit hit queue-head admission and failed RAM gating.
    defer_count: int = 0


class SchedulerConcurrencyMode(Enum):
    LOW = "1"
    HIGH = "0"


class ReservedTracker:
    """
    Tracks reserved/starved units per priority class in strict FIFO order.

    For v1, this tracker uses a fixed fairness window:
      - first 3 INTERACTIVE units
      - first 2 RENDER units
      - first 1 BACKGROUND unit

    The hold budget is the sum of `ram_peak_est_bytes` across those windows.
    """

    def __init__(
        self,
        interactive_reservation_window_size: int = 3,
        render_reservation_window_size: int = 2,
        background_reservation_window_size: int = 1,
        on_defer_exceeded: Optional[Callable[["QueuedWorkUnit"], None]] = None,
        max_defer_count: int = MAX_DEFER_COUNT,
    ) -> None:
        self._window_size_by_priority: Dict[PriorityClass, int] = {
            PriorityClass.INTERACTIVE: int(interactive_reservation_window_size),
            PriorityClass.RENDER: int(render_reservation_window_size),
            PriorityClass.BACKGROUND: int(background_reservation_window_size),
        }
        # Optional hook invoked when a reserved unit's defer_count exceeds
        # `_max_defer_count`. The tracker only signals; it does not act on the
        # unit. May be None (no-op).
        self._on_defer_exceeded: Optional[Callable[[QueuedWorkUnit], None]] = on_defer_exceeded
        self._max_defer_count: int = int(max_defer_count)
        self._reserved_queue_by_priority: Dict[PriorityClass, Deque[QueuedWorkUnit]] = {
            PriorityClass.INTERACTIVE: deque(),
            PriorityClass.RENDER: deque(),
            PriorityClass.BACKGROUND: deque(),
        }
        self._priority_iteration_order = (
            PriorityClass.INTERACTIVE,
            PriorityClass.RENDER,
            PriorityClass.BACKGROUND,
        )
        # Fixed slot template used for fair round-robin selection across windows.
        # Example for (3,2,1): I0,I1,I2,R0,R1,B0.
        self._reservation_slots_in_order: list[tuple[PriorityClass, int]] = []
        for priority_class in self._priority_iteration_order:
            reservation_window_size = self._window_size_by_priority[priority_class]
            for slot_offset in range(max(0, reservation_window_size)):
                self._reservation_slots_in_order.append((priority_class, slot_offset))
        self._next_reservation_slot_index = 0
        self._hold_bytes_by_priority: Dict[PriorityClass, int] = {
            PriorityClass.INTERACTIVE: 0,
            PriorityClass.RENDER: 0,
            PriorityClass.BACKGROUND: 0,
        }
        self._total_hold_bytes = 0
        self._recompute_hold_bytes()

    def enqueue_reserved_unit(self, queued_work_unit: QueuedWorkUnit) -> None:
        """Append to class FIFO and refresh hold accounting."""
        priority_class = queued_work_unit.work_unit.priority_class
        self._reserved_queue_by_priority[priority_class].append(queued_work_unit)
        self._recompute_hold_bytes()

    def pop_reserved_head(self, priority_class: PriorityClass) -> Optional[QueuedWorkUnit]:
        reserved_queue = self._reserved_queue_by_priority[priority_class]
        if not reserved_queue:
            return None
        queued_work_unit = reserved_queue.popleft()
        self._recompute_hold_bytes()
        return queued_work_unit

    def peek_reserved_head(self, priority_class: PriorityClass) -> Optional[QueuedWorkUnit]:
        reserved_queue = self._reserved_queue_by_priority[priority_class]
        if not reserved_queue:
            return None
        return reserved_queue[0]

    def hold_bytes(self) -> int:
        return self._total_hold_bytes

    def hold_bytes_for_interactive(self) -> int:
        return self._hold_bytes_by_priority[PriorityClass.INTERACTIVE]

    def hold_bytes_for_render(self) -> int:
        return self._hold_bytes_by_priority[PriorityClass.RENDER]

    def hold_bytes_for_background(self) -> int:
        return self._hold_bytes_by_priority[PriorityClass.BACKGROUND]

    def reserved_queue_size(self, priority_class: PriorityClass) -> int:
        return len(self._reserved_queue_by_priority[priority_class])

    def remove_units_for_plan(self, plan_id: str) -> int:
        """Best-effort purge hook used when a plan is cancelled/failed."""
        removed_count = 0
        for priority_class in self._priority_iteration_order:
            reserved_queue = self._reserved_queue_by_priority[priority_class]
            filtered_items = deque(item for item in reserved_queue if item.plan_id != plan_id)
            removed_count += len(reserved_queue) - len(filtered_items)
            self._reserved_queue_by_priority[priority_class] = filtered_items
        if removed_count > 0:
            self._recompute_hold_bytes()
        return removed_count

    def pop_next_admissible_reserved_unit(
        self,
        in_flight_ram_bytes: int,
        scheduler_ram_cap_bytes: int,
    ) -> Optional[QueuedWorkUnit]:
        candidate_with_slot = self._next_candidate_with_slot()
        if candidate_with_slot is None:
            return None
        reservation_slot_index, candidate = candidate_with_slot
        candidate_ram_bytes = candidate.work_unit.ram_peak_est_bytes
        if candidate_ram_bytes + in_flight_ram_bytes > scheduler_ram_cap_bytes:
            candidate.defer_count += 1
            self._maybe_signal_defer_exceeded(candidate)
            return None

        removed = self.remove_reserved_unit(candidate)
        if not removed:
            return None

        if self._reservation_slots_in_order:
            self._next_reservation_slot_index = (reservation_slot_index + 1) % len(
                self._reservation_slots_in_order
            )
        return candidate

    def remove_reserved_unit(self, queued_work_unit: QueuedWorkUnit) -> bool:
        """Remove a specific reserved unit instance without reordering survivors."""
        priority_class = queued_work_unit.work_unit.priority_class
        reserved_queue = self._reserved_queue_by_priority[priority_class]
        try:
            reserved_queue.remove(queued_work_unit)
        except ValueError:
            return False
        self._recompute_hold_bytes()
        return True

    def _next_candidate_with_slot(self) -> Optional[tuple[int, QueuedWorkUnit]]:
        """
        Return current round-robin candidate.

        We scan at most one full slot cycle and pick the first slot that is
        currently populated in its class queue.
        """
        if not self._reservation_slots_in_order:
            return None
        total_slots = len(self._reservation_slots_in_order)
        for scanned_slots in range(total_slots):
            slot_index = (self._next_reservation_slot_index + scanned_slots) % total_slots
            priority_class, slot_offset = self._reservation_slots_in_order[slot_index]
            reserved_queue = self._reserved_queue_by_priority[priority_class]
            if slot_offset < len(reserved_queue):
                # We use the slot_index to know what priority class to use. Then
                # we just get the first item in the queue.
                return slot_index, reserved_queue[0]
        return None

    def next_admissible_reserved_unit(
        self,
        in_flight_ram_bytes: int,
        scheduler_ram_cap_bytes: int,
    ) -> Optional[QueuedWorkUnit]:
        """
        Return the next reserved unit that can run under Variant B reserved admission.

        Candidate order is deterministic and fairness-windowed via round-robin slots:
          1) INTERACTIVE slots 0..m-1,
          2) then RENDER slots 0..n-1,
          3) then BACKGROUND slots 0..k-1,
        and the tracker remembers the last served slot to continue fairly.

        Strict fairness rule:
          - only the current round-robin candidate slot is eligible now.
          - if that candidate does not fit, this method returns None (no fallback scanning).

        The candidate is admissible when:
            candidate.ram_peak_est_bytes + in_flight_ram_bytes < scheduler_ram_cap_bytes
        """
        candidate_with_slot = self._next_candidate_with_slot()
        if candidate_with_slot is None:
            return None
        _, candidate = candidate_with_slot
        candidate_ram_bytes = candidate.work_unit.ram_peak_est_bytes
        if candidate_ram_bytes + in_flight_ram_bytes <= scheduler_ram_cap_bytes:
            return candidate
        candidate.defer_count += 1
        self._maybe_signal_defer_exceeded(candidate)
        return None

    def _maybe_signal_defer_exceeded(self, candidate: "QueuedWorkUnit") -> None:
        """Invoke the defer-exceeded hook once, when the candidate's count crosses the cap."""
        if self._on_defer_exceeded is None:
            return
        if candidate.defer_count != self._max_defer_count + 1:
            return
        self._on_defer_exceeded(candidate)

    def _recompute_hold_bytes(self) -> None:
        """Recompute per-class and total hold over configured window sizes."""
        total_hold_bytes = 0
        for priority_class in self._priority_iteration_order:
            hold_window_size = self._window_size_by_priority[priority_class]
            hold_bytes_for_priority = 0
            if hold_window_size > 0:
                reserved_queue = self._reserved_queue_by_priority[priority_class]
                for queue_index, queued_work_unit in enumerate(reserved_queue):
                    if queue_index >= hold_window_size:
                        break
                    hold_bytes_for_priority += queued_work_unit.work_unit.ram_peak_est_bytes
            self._hold_bytes_by_priority[priority_class] = hold_bytes_for_priority
            total_hold_bytes += hold_bytes_for_priority
        self._total_hold_bytes = total_hold_bytes


@dataclass(frozen=True)
class PendingDoneCallback:
    future: Future[Any]
    plan_id: str
    stage_id: str
    unit_id: str
    sem: Semaphore
    ram_peak_est_bytes: int
    executor_kind: str


@dataclass
class StageExecutionState:
    """
    Mutable execution bookkeeping for a single stage in a task plan.

    This state tracks both:
    - stage-level completion (`expected_unit_ids`), and
    - step-level barrier progress (`step_unit_ids` + `step_index`).

    `step_unit_ids` is an ordered list of barrier steps for the stage. Units in the
    same inner list are allowed to run in parallel; the next step cannot begin until
    the current step becomes terminal.

    The scheduler updates this object as units are submitted and completed:
    - `submitted_unit_ids` records units dispatched to an executor,
    - `succeeded_unit_ids` records units that finished successfully,
    - `failed_unit_ids` records units that finished with exceptions.
    """

    stage_id: str
    expected_unit_ids: set[str]
    step_unit_ids: list[list[str]]
    step_index: int = 0
    submitted_unit_ids: set[str] = field(default_factory=set)
    succeeded_unit_ids: set[str] = field(default_factory=set)
    failed_unit_ids: dict[str, BaseException] = field(default_factory=dict)

    def _completed_unit_ids(self) -> set[str]:
        return self.succeeded_unit_ids | set(self.failed_unit_ids.keys())

    def is_current_step_terminal(self) -> bool:
        """
        Return True when the currently active step has no remaining work.

        A step is considered terminal when any of the following is true:
        - `step_index` is past the last configured step,
        - the current step is empty, or
        - every unit in the current step has completed (success or failure).

        This method is used to enforce intra-stage barrier semantics: the scheduler
        advances to the next step only after this returns True.
        """
        if self.step_index >= len(self.step_unit_ids):
            return True
        current_step_expected_unit_ids = set(self.step_unit_ids[self.step_index])
        if not current_step_expected_unit_ids:
            return True
        return current_step_expected_unit_ids.issubset(self._completed_unit_ids())

    def is_terminal(self) -> bool:
        terminal_count = len(self.succeeded_unit_ids) + len(self.failed_unit_ids)
        return terminal_count == len(self.expected_unit_ids)


@dataclass
class PlanExecutionState:
    """Mutable runtime state for a task plan as it moves stage-by-stage."""

    task_plan: TaskPlan
    stage_order: list[str]
    stage_states: dict[str, StageExecutionState]
    completion_future: Future[None] = field(default_factory=Future)
    stage_index: int = 0
    failed_units: dict[str, BaseException] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulerEvent:
    kind: str
    plan_id: str
    time: float
    stage_id: Optional[str] = None
    unit_id: Optional[str] = None
    executor_kind: Optional[str] = None
    priority_class: Optional[PriorityClass] = None
    success: Optional[bool] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class QueueTransitionEvent:
    sequence_id: int
    plan_id: str
    stage_id: str
    unit_id: str
    from_queue: Optional[str]
    to_queue: Optional[str]
    reason: str
    defer_count: int


@dataclass
class RecordingWorkScheduler:
    """In-memory event recorder for asserting scheduler behavior in tests."""

    events: list[SchedulerEvent] = field(default_factory=list)
    clock: Callable[[], float] = perf_counter

    def _record_event(
        self,
        *,
        kind: str,
        plan_id: str,
        stage_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        executor_kind: Optional[str] = None,
        priority_class: Optional[PriorityClass] = None,
        success: Optional[bool] = None,
        error: Optional[str] = None,
    ) -> None:
        self.events.append(
            SchedulerEvent(
                kind=kind,
                plan_id=plan_id,
                time=self.clock(),
                stage_id=stage_id,
                unit_id=unit_id,
                executor_kind=executor_kind,
                priority_class=priority_class,
                success=success,
                error=error,
            )
        )

    def on_plan_submitted(self, plan_id: str) -> None:
        self._record_event(kind="plan_submitted", plan_id=plan_id)

    def on_stage_enqueued(self, plan_id: str, stage_id: str) -> None:
        self._record_event(kind="stage_enqueued", plan_id=plan_id, stage_id=stage_id)

    def on_unit_submitted(
        self,
        plan_id: str,
        stage_id: str,
        unit_id: str,
        executor_kind: str,
        priority_class: PriorityClass,
    ) -> None:
        self._record_event(
            kind="unit_submitted",
            plan_id=plan_id,
            stage_id=stage_id,
            unit_id=unit_id,
            executor_kind=executor_kind,
            priority_class=priority_class,
        )

    def on_unit_done(
        self,
        plan_id: str,
        stage_id: str,
        unit_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        self._record_event(
            kind="unit_done",
            plan_id=plan_id,
            stage_id=stage_id,
            unit_id=unit_id,
            success=success,
            error=error,
        )

    def on_plan_completed(self, plan_id: str, success: bool, error: Optional[str] = None) -> None:
        self._record_event(kind="plan_completed", plan_id=plan_id, success=success, error=error)

    def print_timing_summary(self) -> None:
        """Print plan, stage, and unit completion timings derived from recorder events."""

        plan_ids_in_order = []
        seen_plan_ids: set[str] = set()
        for event in self.events:
            if event.plan_id not in seen_plan_ids:
                plan_ids_in_order.append(event.plan_id)
                seen_plan_ids.add(event.plan_id)

        for plan_id in plan_ids_in_order:
            plan_events = [event for event in self.events if event.plan_id == plan_id]
            plan_start = next((event.time for event in plan_events if event.kind == "plan_submitted"), None)
            plan_end = next(
                (event.time for event in reversed(plan_events) if event.kind == "plan_completed"), None
            )
            plan_duration = (
                f"{plan_end - plan_start:.6f}s" if plan_start is not None and plan_end is not None else "n/a"
            )
            print(f"Plan {plan_id} ({plan_duration})")

            stage_ids_in_order = []
            seen_stage_ids: set[str] = set()
            for event in plan_events:
                if event.stage_id is None or event.stage_id in seen_stage_ids:
                    continue
                stage_ids_in_order.append(event.stage_id)
                seen_stage_ids.add(event.stage_id)

            for stage_id in stage_ids_in_order:
                stage_events = [event for event in plan_events if event.stage_id == stage_id]
                stage_start = next(
                    (event.time for event in stage_events if event.kind == "stage_enqueued"),
                    None,
                )
                stage_done_times = [
                    event.time
                    for event in stage_events
                    if event.kind == "unit_done" and event.unit_id is not None
                ]
                stage_end = max(stage_done_times) if stage_done_times else None
                stage_duration = (
                    f"{stage_end - stage_start:.6f}s"
                    if stage_start is not None and stage_end is not None
                    else "n/a"
                )
                print(f"  Stage {stage_id} ({stage_duration})")

                unit_ids_in_order = []
                seen_unit_ids: set[str] = set()
                for event in stage_events:
                    if event.unit_id is None or event.unit_id in seen_unit_ids:
                        continue
                    unit_ids_in_order.append(event.unit_id)
                    seen_unit_ids.add(event.unit_id)

                for unit_id in unit_ids_in_order:
                    unit_submit = next(
                        (
                            event.time
                            for event in stage_events
                            if event.kind == "unit_submitted" and event.unit_id == unit_id
                        ),
                        None,
                    )
                    unit_done = next(
                        (
                            event.time
                            for event in stage_events
                            if event.kind == "unit_done" and event.unit_id == unit_id
                        ),
                        None,
                    )
                    unit_duration = (
                        f"{unit_done - unit_submit:.6f}s"
                        if unit_submit is not None and unit_done is not None
                        else "n/a"
                    )
                    print(f"    Unit {unit_id}: {unit_duration}")


def _priority_weight(priority: PriorityClass) -> float:
    if priority == PriorityClass.INTERACTIVE:
        return INTERACTIVE_EXECUTOR_BUDGET_WEIGHT
    if priority == PriorityClass.RENDER:
        return RENDER_EXECUTOR_BUDGET_WEIGHT
    return BACKGROUND_EXECUTOR_BUDGET_WEIGHT


def _allocate_priority_tokens(budget: int) -> Dict[PriorityClass, int]:
    """Split executor slots across priorities while guaranteeing at least one each."""

    if budget < PRIORITY_LANE_COUNT:
        raise ValueError(
            f"Budget must be >= {PRIORITY_LANE_COUNT} to guarantee non-zero per priority, got {budget}"
        )

    priorities = [PriorityClass.INTERACTIVE, PriorityClass.RENDER, PriorityClass.BACKGROUND]
    allocation: Dict[PriorityClass, int] = {p: 1 for p in priorities}
    remaining = budget - len(priorities)
    if remaining == 0:
        return allocation

    weighted = {p: remaining * _priority_weight(p) for p in priorities}
    # Allocate deterministic floor first, then distribute remaining slots by largest remainder.
    floor_parts = {p: int(weighted[p]) for p in priorities}
    for p in priorities:
        allocation[p] += floor_parts[p]

    distributed = sum(floor_parts.values())
    leftovers = remaining - distributed
    if leftovers <= 0:
        return allocation

    remainders = sorted(
        priorities,
        key=lambda p: (weighted[p] - floor_parts[p], _priority_weight(p)),
        reverse=True,
    )
    idx = 0
    for _ in range(leftovers):
        pick = remainders[idx % len(remainders)]
        allocation[pick] += 1
        idx += 1
    return allocation


def _normalize_explicit_priority_tokens(
    *,
    explicit_tokens: Dict[PriorityClass, int],
    executor_budget: int,
    executor_kind: str,
) -> Dict[PriorityClass, int]:
    priorities = [PriorityClass.INTERACTIVE, PriorityClass.RENDER, PriorityClass.BACKGROUND]
    missing_priorities = [priority for priority in priorities if priority not in explicit_tokens]
    unknown_priorities = [priority for priority in explicit_tokens if priority not in priorities]
    if missing_priorities:
        raise ValueError(
            f"Explicit {executor_kind} tokens missing priorities: "
            f"{[priority.value for priority in missing_priorities]}"
        )
    if unknown_priorities:
        raise ValueError(f"Explicit {executor_kind} tokens contain unknown priorities: {unknown_priorities}")

    normalized_tokens = {priority: int(explicit_tokens[priority]) for priority in priorities}
    for priority, token_count in normalized_tokens.items():
        if token_count < 0:
            raise ValueError(
                f"Explicit {executor_kind} tokens must be >= 0; got {token_count} for {priority.value}"
            )

    total_tokens = sum(normalized_tokens.values())
    if total_tokens > executor_budget:
        raise ValueError(
            f"Explicit {executor_kind} tokens sum ({total_tokens}) exceeds "
            f"{executor_kind} budget ({executor_budget})"
        )
    return normalized_tokens


def list_tracked_segments(smm):
    conn = smm._Client(smm._address, authkey=smm._authkey)
    try:
        return dispatch(conn, None, "list_segments")
    finally:
        conn.close()


class _RestartableExecutor:
    """An executor that rebuilds itself when it breaks.

    `concurrent.futures` executors can become permanently unusable: a
    `ProcessPoolExecutor` raises `BrokenProcessPool` once a worker process dies
    abruptly (OOM kill, segfault, native crash), and a `ThreadPoolExecutor`
    raises `BrokenThreadPool` if a worker fails to initialize. In both cases
    every subsequent `submit()` raises a `BrokenExecutor`, wedging the pool for
    the rest of the process lifetime.

    This wrapper builds a fresh executor from `factory` and retries the submit
    once, so a single break is recovered transparently. A second `BrokenExecutor`
    on a freshly built executor indicates a fatal environment problem and is
    propagated to the caller.

    Thread-safe on its own: submit and restart are serialized by an internal
    lock, so callers do not need to hold any external lock.
    """

    def __init__(self, name: str, factory: Callable[[], Executor]) -> None:
        self._name = name
        self._factory = factory
        self._lock = Lock()
        self._executor = factory()

    @property
    def executor(self) -> Executor:
        """The current underlying executor (primarily for tests/introspection)."""
        return self._executor

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        with self._lock:
            try:
                return self._executor.submit(fn, *args, **kwargs)
            except BrokenExecutor:
                self._restart_locked()
                return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._executor.shutdown(wait=wait)

    def _restart_locked(self) -> None:
        """Swap in a fresh executor and tear down the broken one. Caller holds `_lock`.

        Futures already dispatched to the broken executor keep their
        `BrokenExecutor` result; their existing done-callbacks still fire, so any
        capacity accounting the owner attached to them self-heals.
        """
        broken_executor = self._executor
        self._executor = self._factory()
        logger.warning("%s was broken; restarted with a fresh executor.", self._name)
        try:
            broken_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            logger.exception("Error shutting down broken %s", self._name)


class WorkScheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        storage_service: "StorageService",
        recorder: Optional[RecordingWorkScheduler] = None,
        task_manager: Optional["TaskManager"] = None,
    ):
        self._config = config
        self._process_budget = int(self._config._process_budget)
        self._thread_budget = int(self._config._thread_budget)
        self._ram_budget_bytes = int(self._config._ram_budget)
        self._in_flight_ram_bytes = 0
        self._defer_to_reserved_threshold = int(self._config._defer_to_reserved_threshold)
        self._recorder = recorder
        self._task_manager = task_manager

        if self._process_budget < PRIORITY_LANE_COUNT:
            raise ValueError(
                f"WorkScheduler requires process budget >= {PRIORITY_LANE_COUNT}, \
                              got {self._process_budget}"
            )
        if self._thread_budget < PRIORITY_LANE_COUNT:
            raise ValueError(
                f"WorkScheduler requires thread budget >= {PRIORITY_LANE_COUNT}, \
                             got {self._thread_budget}"
            )
        if storage_service is None:
            raise ValueError("WorkScheduler requires a storage_service")
        self._storage_service = storage_service

        # Retain the bootstrap so a broken ProcessPoolExecutor can be rebuilt
        # in place with an identical worker initializer (see
        # `_create_process_executor` / `_RestartableExecutor`).
        self._service_address, self._service_authkey = storage_service.get_connection_bootstrap()

        # On Linux, the default 'fork' start method is unsafe when Qt is
        # running: os.fork() triggers pthread_atfork handlers that try to
        # acquire Qt's internal mutexes, which may be held by Qt background
        # threads (especially after a QApplication teardown/recreation between
        # tests). This causes os.fork() to deadlock indefinitely inside
        # executor.submit(). 'forkserver' forks from a clean helper process
        # that has no Qt state, avoiding the deadlock entirely.
        # Issue # 526
        if sys.platform.startswith("linux"):
            self._mp_context = multiprocessing.get_context("forkserver")
        else:
            self._mp_context = multiprocessing.get_context("spawn")
        # Both pools are self-healing: if a worker dies (process) or fails to
        # initialize (thread), the wrapper rebuilds the pool on the next submit.
        self._process_pool = _RestartableExecutor("ProcessPoolExecutor", self._create_process_executor)
        self._thread_pool = _RestartableExecutor("ThreadPoolExecutor", self._create_thread_executor)

        if self._config._process_priority_tokens is not None:
            self._process_tokens = _normalize_explicit_priority_tokens(
                explicit_tokens=self._config._process_priority_tokens,
                executor_budget=self._process_budget,
                executor_kind="process",
            )
        else:
            self._process_tokens = _allocate_priority_tokens(self._process_budget)

        if self._config._thread_priority_tokens is not None:
            self._thread_tokens = _normalize_explicit_priority_tokens(
                explicit_tokens=self._config._thread_priority_tokens,
                executor_budget=self._thread_budget,
                executor_kind="thread",
            )
        else:
            self._thread_tokens = _allocate_priority_tokens(self._thread_budget)
        self._process_semaphores: Dict[PriorityClass, Semaphore] = {
            p: Semaphore(self._process_tokens[p]) for p in self._process_tokens
        }
        self._thread_semaphores: Dict[PriorityClass, Semaphore] = {
            p: Semaphore(self._thread_tokens[p]) for p in self._thread_tokens
        }
        self._process_queues: Dict[PriorityClass, Deque[QueuedWorkUnit]] = {
            p: deque() for p in self._process_tokens
        }
        self._process_blocked_queues: Dict[PriorityClass, Deque[QueuedWorkUnit]] = {
            p: deque() for p in self._process_tokens
        }
        self._thread_queues: Dict[PriorityClass, Deque[QueuedWorkUnit]] = {
            p: deque() for p in self._thread_tokens
        }
        self._thread_blocked_queues: Dict[PriorityClass, Deque[QueuedWorkUnit]] = {
            p: deque() for p in self._thread_tokens
        }
        self._reserved_tracker = ReservedTracker(
            interactive_reservation_window_size=3,
            render_reservation_window_size=2,
            background_reservation_window_size=1,
            on_defer_exceeded=self._on_reserved_defer_exceeded,
        )
        self._pending_done_callbacks: Deque[PendingDoneCallback] = deque()
        self._plan_states: Dict[str, PlanExecutionState] = {}
        self._queue_transition_log: list[QueueTransitionEvent] = []
        self._queue_transition_sequence_id = 0
        self._state_lock = Lock()

    def submit_process(
        self,
        priority: PriorityClass,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        sem = self._process_semaphores[priority]
        sem.acquire()
        future = self._process_pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda _f: sem.release())
        return future

    def submit_thread(
        self,
        priority: PriorityClass,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        sem = self._thread_semaphores[priority]
        sem.acquire()
        future = self._thread_pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda _f: sem.release())
        return future

    def _create_process_executor(self) -> ProcessPoolExecutor:
        """Factory for a ProcessPoolExecutor with the scheduler's worker initializer.

        Used by `_process_pool` for both the initial pool and every restart.
        """
        return ProcessPoolExecutor(
            max_workers=self._process_budget,
            mp_context=self._mp_context,
            initializer=initialize_process_storage_client,
            initargs=(self._service_address, self._service_authkey),
        )

    def _create_thread_executor(self) -> ThreadPoolExecutor:
        """Factory for a ThreadPoolExecutor. Used by `_thread_pool` on init and restart."""
        return ThreadPoolExecutor(
            max_workers=self._thread_budget,
            initializer=initialize_thread_worker,
        )

    def shutdown(self, wait: bool = True) -> None:
        self._thread_pool.shutdown(wait=wait)
        self._process_pool.shutdown(wait=wait)

    def process_tokens(self) -> Dict[PriorityClass, int]:
        return dict(self._process_tokens)

    def thread_tokens(self) -> Dict[PriorityClass, int]:
        return dict(self._thread_tokens)

    def get_queue_transition_log(self) -> list[QueueTransitionEvent]:
        with self._state_lock:
            return list(self._queue_transition_log)

    def get_queue_transition_log_for_unit(self, unit_id: str) -> list[QueueTransitionEvent]:
        with self._state_lock:
            return [event for event in self._queue_transition_log if event.unit_id == unit_id]

    def run_task_plan(self, task_plan: TaskPlan) -> Future[None]:
        """Validate, initialize, and enqueue a task plan for staged execution."""

        self._validate_task_plan(task_plan)
        if self._recorder is not None:
            self._recorder.on_plan_submitted(task_plan.plan_id)
        stage_order = self._ordered_stage_ids(task_plan)
        stage_states = {
            stage_id: StageExecutionState(
                stage_id=stage_id,
                expected_unit_ids=set(task_plan.stage_work_units.get(stage_id, [])),
                step_unit_ids=self._steps_for_stage(task_plan, stage_id),
            )
            for stage_id in stage_order
        }
        plan_state = PlanExecutionState(
            task_plan=task_plan,
            stage_order=stage_order,
            stage_states=stage_states,
        )
        with self._state_lock:
            self._plan_states[task_plan.plan_id] = plan_state
            # Stage execution is strictly sequential; enqueue only the first stage now.
            self._enqueue_stage_locked(plan_state, stage_order[0])
            self._drain_queues_locked()
        # Attach callbacks only after releasing scheduler state lock to avoid
        # immediate callback re-entry while lock is still held.
        self._flush_pending_done_callbacks()
        return plan_state.completion_future

    def cancel_plan(self, plan_id: str) -> None:
        """Remove queued work for a plan from scheduler-managed queues."""
        with self._state_lock:
            self._purge_plan_from_queues_locked(plan_id)
            plan_state = self._plan_states.pop(plan_id, None)
            if plan_state is not None:
                self._finalize_plan_outputs(plan_state, success=False, aborted=True)
            if plan_state is not None and not plan_state.completion_future.done():
                plan_state.completion_future.cancel()
            self._drain_queues_locked()

    def _validate_task_plan(self, task_plan: TaskPlan) -> None:
        """Ensure stage/unit mappings are internally consistent before scheduling."""

        if not task_plan.stage_work_units:
            raise ValueError("TaskPlan has no stage_work_units")
        for stage_id, unit_ids in task_plan.stage_work_units.items():
            for unit_id in unit_ids:
                if unit_id not in task_plan.work_units:
                    raise KeyError(f"TaskPlan stage {stage_id} references unknown work unit {unit_id}")
                unit = task_plan.work_units[unit_id]
                if unit.stage_id != stage_id:
                    raise ValueError(f"WorkUnit {unit_id} has stage_id={unit.stage_id}, expected {stage_id}")

            if stage_id not in task_plan.stage_steps:
                continue

            configured_steps = task_plan.stage_steps[stage_id]
            flattened_step_unit_ids: list[str] = []
            for step_unit_ids in configured_steps:
                for unit_id in step_unit_ids:
                    if unit_id not in task_plan.work_units:
                        raise KeyError(
                            f"TaskPlan stage_steps {stage_id} references unknown work unit {unit_id}"
                        )
                    unit = task_plan.work_units[unit_id]
                    if unit.stage_id != stage_id:
                        raise ValueError(
                            f"TaskPlan stage_steps {stage_id} contains work unit {unit_id} "
                            f"with stage_id={unit.stage_id}"
                        )
                    flattened_step_unit_ids.append(unit_id)

            if Counter(flattened_step_unit_ids) != Counter(unit_ids):
                raise ValueError(
                    f"TaskPlan stage_steps for stage {stage_id} must contain the same units as "
                    "stage_work_units (including multiplicity)."
                )

    def _ordered_stage_ids(self, task_plan: TaskPlan) -> list[str]:
        return sorted(task_plan.stage_work_units.keys(), key=self._stage_sort_key)

    def _stage_sort_key(self, stage_id: str) -> tuple[int, str]:
        digits = "".join(ch for ch in stage_id if ch.isdigit())
        if digits:
            return int(digits), stage_id
        return 10**9, stage_id

    def _steps_for_stage(self, task_plan: TaskPlan, stage_id: str) -> list[list[str]]:
        configured_steps = task_plan.stage_steps.get(stage_id)
        if configured_steps is not None:
            return [list(step_unit_ids) for step_unit_ids in configured_steps]
        # Backward compatibility: one parallel step with all stage units.
        return [list(task_plan.stage_work_units.get(stage_id, []))]

    def _enqueue_stage_locked(self, plan_state: PlanExecutionState, stage_id: str) -> None:
        """Queue current stage step work units into executor queues by priority and kind."""
        task_plan = plan_state.task_plan
        stage_state = plan_state.stage_states[stage_id]
        if stage_state.step_index >= len(stage_state.step_unit_ids):
            return

        if self._recorder is not None:
            # Emit once per stage, not once per step, for stable recorder semantics.
            if stage_state.step_index == 0:
                self._recorder.on_stage_enqueued(task_plan.plan_id, stage_id)
        for unit_id in stage_state.step_unit_ids[stage_state.step_index]:
            work_unit = task_plan.work_units[unit_id]
            item = QueuedWorkUnit(plan_id=task_plan.plan_id, stage_id=stage_id, work_unit=work_unit)
            if work_unit.executor_kind == "process":
                self._process_queues[work_unit.priority_class].append(item)
                self._log_queue_transition_locked(
                    item=item,
                    from_queue=None,
                    to_queue=self._main_queue_name(work_unit.executor_kind, work_unit.priority_class),
                    reason="stage_enqueued",
                )
            elif work_unit.executor_kind == "thread":
                self._thread_queues[work_unit.priority_class].append(item)
                self._log_queue_transition_locked(
                    item=item,
                    from_queue=None,
                    to_queue=self._main_queue_name(work_unit.executor_kind, work_unit.priority_class),
                    reason="stage_enqueued",
                )
            else:
                raise ValueError(f"Unknown executor kind: {work_unit.executor_kind!r}")

    def _drain_queues_locked(self) -> None:
        """Submit queued work while priority semaphores indicate available capacity."""

        made_progress = True
        while made_progress:
            made_progress = False
            # Reserved work gets first admission chance by policy.
            if self._attempt_submit_reserved_locked():
                made_progress = True
                continue
            for priority in (
                PriorityClass.INTERACTIVE,
                PriorityClass.RENDER,
                PriorityClass.BACKGROUND,
            ):
                process_sem = self._process_semaphores[priority]
                if self._attempt_submit_from_queue_locked(
                    main_queue=self._process_queues[priority],
                    blocked_queue=self._process_blocked_queues[priority],
                    sem=process_sem,
                ):
                    made_progress = True

                thread_sem = self._thread_semaphores[priority]
                if self._attempt_submit_from_queue_locked(
                    main_queue=self._thread_queues[priority],
                    blocked_queue=self._thread_blocked_queues[priority],
                    sem=thread_sem,
                ):
                    made_progress = True

    def _attempt_submit_reserved_locked(self) -> bool:
        """
        Attempt one reserved admission without mutating reserved order prematurely.

        We first peek for an admissible reserved candidate, then acquire the
        relevant semaphore, then re-check and finally remove the exact unit
        immediately before submitting.
        """
        candidate = self._reserved_tracker.next_admissible_reserved_unit(
            in_flight_ram_bytes=self._in_flight_ram_bytes,
            scheduler_ram_cap_bytes=self._ram_budget_bytes,
        )
        if candidate is None:
            return False
        priority_class = candidate.work_unit.priority_class
        sem = (
            self._process_semaphores[priority_class]
            if candidate.work_unit.executor_kind == "process"
            else self._thread_semaphores[priority_class]
        )
        if not sem.acquire(blocking=False):
            return False
        # Re-check under held token because memory/completion state may have changed.
        candidate_after_capacity_check = self._reserved_tracker.next_admissible_reserved_unit(
            in_flight_ram_bytes=self._in_flight_ram_bytes,
            scheduler_ram_cap_bytes=self._ram_budget_bytes,
        )
        if candidate_after_capacity_check is None:
            sem.release()
            return False
        if not self._reserved_tracker.remove_reserved_unit(candidate_after_capacity_check):
            sem.release()
            return False
        self._log_queue_transition_locked(
            item=candidate_after_capacity_check,
            from_queue=self._reserved_queue_name(candidate_after_capacity_check.work_unit.priority_class),
            to_queue=self._in_flight_queue_name(candidate_after_capacity_check.work_unit.executor_kind),
            reason="reserved_admitted",
        )
        return self._submit_runnable_item_locked(candidate_after_capacity_check, sem)

    def _attempt_submit_from_queue_locked(
        self,
        main_queue: Deque[QueuedWorkUnit],
        blocked_queue: Deque[QueuedWorkUnit],
        sem: Semaphore,
    ) -> bool:
        """Try one non-reserved submission attempt for a priority lane."""
        if not blocked_queue and not main_queue:
            return False
        if not sem.acquire(blocking=False):
            return False
        return self._submit_non_reserved_from_queues_locked(main_queue, blocked_queue, sem)

    def _submit_non_reserved_from_queues_locked(
        self,
        main_queue: Deque[QueuedWorkUnit],
        blocked_queue: Deque[QueuedWorkUnit],
        sem: Semaphore,
    ) -> bool:
        """
        Attempt one non-reserved submission with blocked-first, then main scanning.

        - We scan blocked queue candidates first and submit the first admissible one.
        - If no blocked candidate can run, we scan main queue candidates.
        - We stop after one successful submission, or after exhausting both queues.
        """
        for blocked_candidate in list(blocked_queue):
            if blocked_candidate not in blocked_queue:
                continue
            if self._can_admit_non_reserved(blocked_candidate):
                blocked_queue.remove(blocked_candidate)
                self._log_queue_transition_locked(
                    item=blocked_candidate,
                    from_queue=self._blocked_queue_name(
                        blocked_candidate.work_unit.executor_kind,
                        blocked_candidate.work_unit.priority_class,
                    ),
                    to_queue=self._in_flight_queue_name(blocked_candidate.work_unit.executor_kind),
                    reason="blocked_admitted",
                )
                return self._submit_runnable_item_locked(blocked_candidate, sem)

            blocked_candidate.defer_count += 1
            if blocked_candidate.defer_count > self._defer_to_reserved_threshold:
                blocked_queue.remove(blocked_candidate)
                self._reserved_tracker.enqueue_reserved_unit(blocked_candidate)
                self._log_queue_transition_locked(
                    item=blocked_candidate,
                    from_queue=self._blocked_queue_name(
                        blocked_candidate.work_unit.executor_kind,
                        blocked_candidate.work_unit.priority_class,
                    ),
                    to_queue=self._reserved_queue_name(blocked_candidate.work_unit.priority_class),
                    reason="defer_threshold_exceeded",
                )

        for main_candidate in list(main_queue):
            if main_candidate not in main_queue:
                continue
            if self._can_admit_non_reserved(main_candidate):
                main_queue.remove(main_candidate)
                self._log_queue_transition_locked(
                    item=main_candidate,
                    from_queue=self._main_queue_name(
                        main_candidate.work_unit.executor_kind,
                        main_candidate.work_unit.priority_class,
                    ),
                    to_queue=self._in_flight_queue_name(main_candidate.work_unit.executor_kind),
                    reason="main_admitted",
                )
                return self._submit_runnable_item_locked(main_candidate, sem)

            main_candidate.defer_count += 1
            main_queue.remove(main_candidate)
            if main_candidate.defer_count > self._defer_to_reserved_threshold:
                self._reserved_tracker.enqueue_reserved_unit(main_candidate)
                self._log_queue_transition_locked(
                    item=main_candidate,
                    from_queue=self._main_queue_name(
                        main_candidate.work_unit.executor_kind,
                        main_candidate.work_unit.priority_class,
                    ),
                    to_queue=self._reserved_queue_name(main_candidate.work_unit.priority_class),
                    reason="defer_threshold_exceeded",
                )
            else:
                blocked_queue.append(main_candidate)
                self._log_queue_transition_locked(
                    item=main_candidate,
                    from_queue=self._main_queue_name(
                        main_candidate.work_unit.executor_kind,
                        main_candidate.work_unit.priority_class,
                    ),
                    to_queue=self._blocked_queue_name(
                        main_candidate.work_unit.executor_kind,
                        main_candidate.work_unit.priority_class,
                    ),
                    reason="ram_gate_failed",
                )
        sem.release()
        return False

    def _can_admit_non_reserved(self, item: QueuedWorkUnit) -> bool:
        """Variant B gate for non-reserved work: cap minus reserved hold."""
        required_ram_bytes = item.work_unit.ram_peak_est_bytes
        available_ram_bytes = self._ram_budget_bytes - self._reserved_hold_bytes()
        return self._in_flight_ram_bytes + required_ram_bytes <= available_ram_bytes

    def _submit_runnable_item_locked(self, item: QueuedWorkUnit, sem: Semaphore) -> bool:
        """
        Submit a work unit that has already passed admission checks.

        This method assumes `_state_lock` is held and the caller already acquired
        `sem` for the unit's priority/executor lane. On success it:
          - marks the stage unit as submitted,
          - increments `_in_flight_ram_bytes`,
          - submits the unit to the appropriate executor, and
          - records a pending done-callback registration.

        If the plan is no longer active, it releases `sem` and returns `False`.
        """
        plan_state = self._plan_states.get(item.plan_id)
        if plan_state is None or plan_state.completion_future.done():
            sem.release()
            return False

        required_ram_bytes = item.work_unit.ram_peak_est_bytes
        stage_state = plan_state.stage_states[item.stage_id]
        pool = self._process_pool if item.work_unit.executor_kind == "process" else self._thread_pool
        # Submit before mutating in-flight bookkeeping: a broken pool is restarted
        # and the submit retried inside `pool.submit`, but if the retry still
        # fails we must not leave phantom RAM/unit accounting behind. Release the
        # held token and re-raise so the caller never accounts for work that
        # never started.
        try:
            future = pool.submit(self._execute_work_unit, item.work_unit)
        except BrokenExecutor:
            sem.release()
            raise
        # Mark as submitted after dispatch so stage accounting reflects in-flight work.
        stage_state.submitted_unit_ids.add(item.work_unit.unit_id)
        self._in_flight_ram_bytes += required_ram_bytes
        if self._recorder is not None:
            self._recorder.on_unit_submitted(
                plan_id=item.plan_id,
                stage_id=item.stage_id,
                unit_id=item.work_unit.unit_id,
                executor_kind=item.work_unit.executor_kind,
                priority_class=item.work_unit.priority_class,
            )
        # Defer callback attachment until after lock release to avoid immediate
        # callback re-entry (`add_done_callback` can invoke synchronously).
        self._pending_done_callbacks.append(
            PendingDoneCallback(
                future=future,
                plan_id=item.plan_id,
                stage_id=item.stage_id,
                unit_id=item.work_unit.unit_id,
                sem=sem,
                ram_peak_est_bytes=required_ram_bytes,
                executor_kind=item.work_unit.executor_kind,
            )
        )
        return True

    def _reserved_hold_bytes(self) -> int:
        """Total bytes withheld from non-reserved admissions."""
        return self._reserved_tracker.hold_bytes()

    def _main_queue_name(self, executor_kind: str, priority_class: PriorityClass) -> str:
        return f"main:{executor_kind}:{priority_class.value}"

    def _blocked_queue_name(self, executor_kind: str, priority_class: PriorityClass) -> str:
        return f"blocked:{executor_kind}:{priority_class.value}"

    def _reserved_queue_name(self, priority_class: PriorityClass) -> str:
        return f"reserved:{priority_class.value}"

    def _in_flight_queue_name(self, executor_kind: str) -> str:
        return f"in_flight:{executor_kind}"

    def _log_queue_transition_locked(
        self,
        item: QueuedWorkUnit,
        from_queue: Optional[str],
        to_queue: Optional[str],
        reason: str,
    ) -> None:
        self._log_queue_transition_by_fields_locked(
            plan_id=item.plan_id,
            stage_id=item.stage_id,
            unit_id=item.work_unit.unit_id,
            from_queue=from_queue,
            to_queue=to_queue,
            reason=reason,
            defer_count=item.defer_count,
        )

    def _log_queue_transition_by_fields_locked(
        self,
        plan_id: str,
        stage_id: str,
        unit_id: str,
        from_queue: Optional[str],
        to_queue: Optional[str],
        reason: str,
        defer_count: int,
    ) -> None:
        self._queue_transition_sequence_id += 1
        self._queue_transition_log.append(
            QueueTransitionEvent(
                sequence_id=self._queue_transition_sequence_id,
                plan_id=plan_id,
                stage_id=stage_id,
                unit_id=unit_id,
                from_queue=from_queue,
                to_queue=to_queue,
                reason=reason,
                defer_count=defer_count,
            )
        )

    def _on_unit_done(
        self,
        plan_id: str,
        stage_id: str,
        unit_id: str,
        fut: Future[Any],
        sem: Semaphore,
        ram_peak_est_bytes: int,
        executor_kind: str,
    ) -> None:
        """Handle unit completion, advance stages, and resolve the plan future."""
        try:
            with self._state_lock:
                plan_state: Optional[PlanExecutionState] = None
                try:
                    # Always return capacity token, even when state is already terminal/evicted.
                    sem.release()
                    self._in_flight_ram_bytes = max(0, self._in_flight_ram_bytes - ram_peak_est_bytes)
                    plan_state = self._plan_states.get(plan_id)
                    if plan_state is None:
                        return
                    if plan_state.completion_future.done():
                        return

                    stage_state = plan_state.stage_states[stage_id]
                    exc = fut.exception()

                    if exc is None:
                        stage_state.succeeded_unit_ids.add(unit_id)
                        self._log_queue_transition_by_fields_locked(
                            plan_id=plan_id,
                            stage_id=stage_id,
                            unit_id=unit_id,
                            from_queue=self._in_flight_queue_name(executor_kind),
                            to_queue="done",
                            reason="unit_succeeded",
                            defer_count=0,
                        )
                        if self._recorder is not None:
                            self._recorder.on_unit_done(plan_id, stage_id, unit_id, success=True)
                    else:
                        stage_state.failed_unit_ids[unit_id] = exc
                        plan_state.failed_units[unit_id] = exc
                        self._log_queue_transition_by_fields_locked(
                            plan_id=plan_id,
                            stage_id=stage_id,
                            unit_id=unit_id,
                            from_queue=self._in_flight_queue_name(executor_kind),
                            to_queue="done",
                            reason="unit_failed",
                            defer_count=0,
                        )
                        if self._recorder is not None:
                            self._recorder.on_unit_done(
                                plan_id,
                                stage_id,
                                unit_id,
                                success=False,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        if self._task_manager is not None:
                            self._task_manager.task_errored.emit((plan_id, f"{type(exc).__name__}: {exc}"))

                        if plan_state.task_plan.fail_fast:
                            # Cancel queued-but-not-submitted work and fail immediately.
                            fail_message = (
                                f"TaskPlan {plan_id} failed fast at stage {stage_id} due "
                                f"to work unit {unit_id}: {exc}\n\n"
                                f"Traceback:\n{exc.__traceback__}"
                            )
                            self._abort_plan_locked(
                                plan_id,
                                error=RuntimeError(fail_message),
                                error_message=fail_message,
                                emit_task_errored=False,
                            )
                            return

                    if self._task_manager is not None:
                        completed_units = sum(
                            len(stage_state.succeeded_unit_ids) + len(stage_state.failed_unit_ids)
                            for stage_state in plan_state.stage_states.values()
                        )
                        total_units = len(plan_state.task_plan.work_units)
                        self._task_manager.task_progressed.emit((plan_id, completed_units, total_units))

                    if not stage_state.is_current_step_terminal():
                        # Current step still has in-flight work; only attempt to fill open slots.
                        self._drain_queues_locked()
                        return

                    # Current step completed; enqueue the next step for this stage, if any.
                    if stage_state.step_index < len(stage_state.step_unit_ids) - 1:
                        stage_state.step_index += 1
                        self._enqueue_stage_locked(plan_state, stage_id)
                        self._drain_queues_locked()
                        return

                    at_last_stage = plan_state.stage_index >= len(plan_state.stage_order) - 1
                    if at_last_stage:
                        if plan_state.failed_units:
                            first_unit_id, first_exc = next(iter(plan_state.failed_units.items()))
                            fail_message = (
                                f"TaskPlan {plan_id} completed with failures; "
                                f"first failure unit={first_unit_id}: {first_exc}"
                            )
                            self._finalize_plan_outputs(plan_state, success=False)
                            plan_state.completion_future.set_exception(RuntimeError(fail_message))
                            if self._recorder is not None:
                                self._recorder.on_plan_completed(plan_id, success=False, error=fail_message)
                        else:
                            completion_callback = plan_state.task_plan.completion_callback
                            # We call the completion callback BEFORE we delete outputs in
                            # self._finalize_plan_outputs
                            if completion_callback is not None:
                                completion_callback(plan_state.task_plan.bindings)
                            self._finalize_plan_outputs(plan_state, success=True)
                            if self._task_manager is not None:
                                self._task_manager.task_finished.emit(plan_id)
                            plan_state.completion_future.set_result(None)
                            if self._recorder is not None:
                                self._recorder.on_plan_completed(plan_id, success=True)
                        self._plan_states.pop(plan_id, None)
                        self._drain_queues_locked()
                        return

                    # Stage is complete; enqueue the next stage and continue draining.
                    plan_state.stage_index += 1
                    next_stage_id = plan_state.stage_order[plan_state.stage_index]
                    self._enqueue_stage_locked(plan_state, next_stage_id)
                    self._drain_queues_locked()

                except Exception as scheduler_exc:
                    # An unhandled exception inside the scheduler's own logic (e.g. in a
                    # completion_callback) would otherwise be silently swallowed by
                    # concurrent.futures' done-callback mechanism, leaving
                    # completion_future pending forever. Surface it here instead.
                    fail_message = (
                        f"Unhandled scheduler error while processing unit {unit_id} "
                        f"of plan {plan_id}: {type(scheduler_exc).__name__}: {scheduler_exc}"
                    )
                    if plan_state is not None and not plan_state.completion_future.done():
                        if self._recorder is not None:
                            self._recorder.on_unit_done(
                                plan_id, stage_id, unit_id, success=False, error=fail_message
                            )
                        self._abort_plan_locked(
                            plan_id,
                            error=scheduler_exc,
                            error_message=fail_message,
                            emit_task_errored=True,
                        )
        finally:
            # _on_unit_done can enqueue more work under lock; flush registrations
            # now so newly submitted futures get callbacks attached lock-free.
            self._flush_pending_done_callbacks()

    def _finalize_plan_outputs(
        self,
        plan_state: PlanExecutionState,
        *,
        success: bool,
        aborted: bool = False,
    ) -> None:
        """
        Move plan-produced refs into a terminal producer state and release the plan's
        own planned-consumer hold on them.
        """

        if success:
            producer_update = self._storage_service.mark_producer_completed
        elif aborted:
            producer_update = self._storage_service.mark_producer_aborted
        else:
            producer_update = self._storage_service.mark_producer_failed

        for ref_id in plan_state.task_plan.produced_ref_ids:
            producer_update(ref_id)
            self._storage_service.release_plan_consumer(ref_id, plan_state.task_plan.plan_id)

    def _flush_pending_done_callbacks(self) -> None:
        """
        Attach queued done callbacks after lock-protected scheduling completes.

        We collect callback registrations while holding `_state_lock`, then
        attach them after releasing the lock so `Future.add_done_callback(...)`
        cannot synchronously invoke `_on_unit_done` re-entrantly under lock.
        """
        with self._state_lock:
            pending = list(self._pending_done_callbacks)
            self._pending_done_callbacks.clear()

        for pending_callback in pending:
            pending_callback.future.add_done_callback(
                lambda fut,
                pid=pending_callback.plan_id,
                sid=pending_callback.stage_id,
                uid=pending_callback.unit_id,
                s=pending_callback.sem,
                r=pending_callback.ram_peak_est_bytes,
                ek=pending_callback.executor_kind: self._on_unit_done(pid, sid, uid, fut, s, r, ek)
            )

    def _purge_plan_from_queues_locked(self, plan_id: str) -> None:
        for queue_map in (
            self._process_queues,
            self._thread_queues,
            self._process_blocked_queues,
            self._thread_blocked_queues,
        ):
            for priority in queue_map:
                retained = deque(item for item in queue_map[priority] if item.plan_id != plan_id)
                queue_map[priority] = retained
        self._reserved_tracker.remove_units_for_plan(plan_id)
        if self._task_manager is not None:
            self._task_manager.task_cancelled.emit(plan_id)

    def _on_reserved_defer_exceeded(self, queued: QueuedWorkUnit) -> None:
        """Reserved-tracker hook fired when a unit's defer_count exceeds MAX_DEFER_COUNT.

        The tracker calls this from inside our `_state_lock`, so we reuse the
        plan-abort path directly. The plan is failed because the unit can't be
        admitted (typically because `ram_peak_est_bytes` exceeds the scheduler
        budget) and would otherwise hang the plan forever.
        """
        error_message = (
            f"TaskPlan {queued.plan_id} aborted: reserved work unit "
            f"{queued.work_unit.unit_id!r} (stage {queued.stage_id!r}) failed "
            f"RAM admission {queued.defer_count} times "
            f"(ram_peak_est_bytes={queued.work_unit.ram_peak_est_bytes}, "
            f"scheduler_ram_cap_bytes={self._ram_budget_bytes}, "
            f"max_defer_count={MAX_DEFER_COUNT})."
        )
        self._abort_plan_locked(
            queued.plan_id,
            error=RuntimeError(error_message),
            error_message=error_message,
            emit_task_errored=True,
        )

    def _abort_plan_locked(
        self,
        plan_id: str,
        *,
        error: BaseException,
        error_message: Optional[str] = None,
        emit_task_errored: bool = False,
    ) -> None:
        """Centralized fail-and-cleanup path for a plan. Caller must hold `_state_lock`.

        Performs (idempotent) the steps shared by every abort site:
          - purge queued/blocked/reserved work for `plan_id`,
          - notify recorder that the plan completed unsuccessfully,
          - optionally notify the task manager (`task_errored`),
          - finalize plan outputs as failed,
          - set the exception on the plan's completion future,
          - remove the plan from `_plan_states`,
          - drain queues so other plans can make progress.

        If the plan is unknown or its completion future is already done, this
        is a no-op so callers don't need to guard separately.
        """
        plan_state = self._plan_states.get(plan_id)
        if plan_state is None or plan_state.completion_future.done():
            return
        if error_message is None:
            error_message = f"{type(error).__name__}: {error}"
        self._purge_plan_from_queues_locked(plan_id)
        if self._recorder is not None:
            self._recorder.on_plan_completed(plan_id, success=False, error=error_message)
        if emit_task_errored and self._task_manager is not None:
            self._task_manager.task_errored.emit((plan_id, error_message))
        self._finalize_plan_outputs(plan_state, success=False)
        plan_state.completion_future.set_exception(error)
        self._plan_states.pop(plan_id, None)
        self._drain_queues_locked()

    @staticmethod
    def _execute_work_unit(work_unit: WorkUnit) -> Any:
        return work_unit.fn()
