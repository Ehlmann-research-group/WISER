from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from collections import deque
from threading import Lock, Semaphore
from typing import Any, Callable, Deque, Dict, Optional, TYPE_CHECKING

from .primitives import PriorityClass
from .task_system import TaskPlan, WorkUnit
from .worker_runtime import initialize_process_storage_client, initialize_thread_worker

if TYPE_CHECKING:
    from wiser.utils.storage_service import StorageService

SCHEDULER_PROCESS_BUDGET = 6
SCHEDULER_RAM_BUDGET = 4_000_000_000
SCHEDULER_THREAD_BUDGET = 32


@dataclass
class SchedulerConfig:
    """
    Specifies the computer resources that we can use.
    """

    _process_budget: int = SCHEDULER_PROCESS_BUDGET
    _thread_budget: int = SCHEDULER_THREAD_BUDGET
    _ram_budget: int = SCHEDULER_RAM_BUDGET


@dataclass(frozen=True)
class QueuedWorkUnit:
    plan_id: str
    stage_id: str
    work_unit: WorkUnit


@dataclass(frozen=True)
class PendingDoneCallback:
    future: Future[Any]
    plan_id: str
    stage_id: str
    unit_id: str
    sem: Semaphore


@dataclass
class StageExecutionState:
    """Tracks submission and completion status for one stage in a task plan."""

    stage_id: str
    expected_unit_ids: set[str]
    submitted_unit_ids: set[str] = field(default_factory=set)
    succeeded_unit_ids: set[str] = field(default_factory=set)
    failed_unit_ids: dict[str, BaseException] = field(default_factory=dict)

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
    stage_id: Optional[str] = None
    unit_id: Optional[str] = None
    executor_kind: Optional[str] = None
    priority_class: Optional[PriorityClass] = None
    success: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class RecordingWorkScheduler:
    """In-memory event recorder for asserting scheduler behavior in tests."""

    events: list[SchedulerEvent] = field(default_factory=list)

    def on_plan_submitted(self, plan_id: str) -> None:
        self.events.append(SchedulerEvent(kind="plan_submitted", plan_id=plan_id))

    def on_stage_enqueued(self, plan_id: str, stage_id: str) -> None:
        self.events.append(SchedulerEvent(kind="stage_enqueued", plan_id=plan_id, stage_id=stage_id))

    def on_unit_submitted(
        self,
        plan_id: str,
        stage_id: str,
        unit_id: str,
        executor_kind: str,
        priority_class: PriorityClass,
    ) -> None:
        self.events.append(
            SchedulerEvent(
                kind="unit_submitted",
                plan_id=plan_id,
                stage_id=stage_id,
                unit_id=unit_id,
                executor_kind=executor_kind,
                priority_class=priority_class,
            )
        )

    def on_unit_done(
        self,
        plan_id: str,
        stage_id: str,
        unit_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        self.events.append(
            SchedulerEvent(
                kind="unit_done",
                plan_id=plan_id,
                stage_id=stage_id,
                unit_id=unit_id,
                success=success,
                error=error,
            )
        )

    def on_plan_completed(self, plan_id: str, success: bool, error: Optional[str] = None) -> None:
        self.events.append(
            SchedulerEvent(kind="plan_completed", plan_id=plan_id, success=success, error=error)
        )


def _priority_weight(priority: PriorityClass) -> float:
    if priority == PriorityClass.INTERACTIVE:
        return 1.0 / 2.0
    if priority == PriorityClass.RENDER:
        return 1.0 / 3.0
    return 1.0 / 6.0


def _allocate_priority_tokens(budget: int) -> Dict[PriorityClass, int]:
    """Split executor slots across priorities while guaranteeing at least one each."""

    if budget < 3:
        raise ValueError(f"Budget must be >= 3 to guarantee non-zero per priority, got {budget}")

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


class WorkScheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        storage_service: "StorageService",
        recorder: Optional[RecordingWorkScheduler] = None,
    ):
        self._config = config
        self._process_budget = int(self._config._process_budget)
        self._thread_budget = int(self._config._thread_budget)
        self._recorder = recorder

        if self._process_budget < 3:
            raise ValueError(f"WorkScheduler requires process budget >= 3, got {self._process_budget}")
        if self._thread_budget < 3:
            raise ValueError(f"WorkScheduler requires thread budget >= 3, got {self._thread_budget}")
        if storage_service is None:
            raise ValueError("WorkScheduler requires a storage_service")

        service_address, service_authkey = storage_service.get_connection_bootstrap()

        self._process_executor = ProcessPoolExecutor(
            max_workers=self._process_budget,
            initializer=initialize_process_storage_client,
            initargs=(service_address, service_authkey),
        )
        self._thread_executor = ThreadPoolExecutor(
            max_workers=self._thread_budget,
            initializer=initialize_thread_worker,
        )

        self._process_tokens = _allocate_priority_tokens(self._process_budget)
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
        self._thread_queues: Dict[PriorityClass, Deque[QueuedWorkUnit]] = {
            p: deque() for p in self._thread_tokens
        }
        self._pending_done_callbacks: Deque[PendingDoneCallback] = deque()
        self._plan_states: Dict[str, PlanExecutionState] = {}
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
        future = self._process_executor.submit(fn, *args, **kwargs)
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
        future = self._thread_executor.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda _f: sem.release())
        return future

    def shutdown(self, wait: bool = True) -> None:
        self._thread_executor.shutdown(wait=wait)
        self._process_executor.shutdown(wait=wait)

    def process_tokens(self) -> Dict[PriorityClass, int]:
        return dict(self._process_tokens)

    def thread_tokens(self) -> Dict[PriorityClass, int]:
        return dict(self._thread_tokens)

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

    def _ordered_stage_ids(self, task_plan: TaskPlan) -> list[str]:
        return sorted(task_plan.stage_work_units.keys(), key=self._stage_sort_key)

    def _stage_sort_key(self, stage_id: str) -> tuple[int, str]:
        digits = "".join(ch for ch in stage_id if ch.isdigit())
        if digits:
            return int(digits), stage_id
        return 10**9, stage_id

    def _enqueue_stage_locked(self, plan_state: PlanExecutionState, stage_id: str) -> None:
        """Queue all work units for a stage into executor queues by priority and kind."""
        task_plan = plan_state.task_plan
        if self._recorder is not None:
            self._recorder.on_stage_enqueued(task_plan.plan_id, stage_id)
        for unit_id in task_plan.stage_work_units.get(stage_id, []):
            work_unit = task_plan.work_units[unit_id]
            item = QueuedWorkUnit(plan_id=task_plan.plan_id, stage_id=stage_id, work_unit=work_unit)
            if work_unit.executor_kind == "process":
                self._process_queues[work_unit.priority_class].append(item)
            elif work_unit.executor_kind == "thread":
                self._thread_queues[work_unit.priority_class].append(item)
            else:
                raise ValueError(f"Unknown executor kind: {work_unit.executor_kind!r}")

    def _drain_queues_locked(self) -> None:
        """Submit queued work while priority semaphores indicate available capacity."""

        made_progress = True
        while made_progress:
            made_progress = False
            for priority in (
                PriorityClass.INTERACTIVE,
                PriorityClass.RENDER,
                PriorityClass.BACKGROUND,
            ):
                process_queue = self._process_queues[priority]
                process_sem = self._process_semaphores[priority]
                # Keep pulling while both work and capacity exist for this priority bucket.
                while process_queue and process_sem.acquire(blocking=False):
                    item = process_queue.popleft()
                    self._submit_queued_item_locked(item, process_sem)
                    made_progress = True

                thread_queue = self._thread_queues[priority]
                thread_sem = self._thread_semaphores[priority]
                while thread_queue and thread_sem.acquire(blocking=False):
                    item = thread_queue.popleft()
                    self._submit_queued_item_locked(item, thread_sem)
                    made_progress = True

    def _submit_queued_item_locked(self, item: QueuedWorkUnit, sem: Semaphore) -> None:
        """Submit one queued unit under lock and bind completion handling to token release."""

        plan_state = self._plan_states.get(item.plan_id)
        if plan_state is None or plan_state.completion_future.done():
            sem.release()
            return

        stage_state = plan_state.stage_states[item.stage_id]
        # Mark as submitted before dispatch so stage accounting reflects in-flight work.
        stage_state.submitted_unit_ids.add(item.work_unit.unit_id)
        if self._recorder is not None:
            self._recorder.on_unit_submitted(
                plan_id=item.plan_id,
                stage_id=item.stage_id,
                unit_id=item.work_unit.unit_id,
                executor_kind=item.work_unit.executor_kind,
                priority_class=item.work_unit.priority_class,
            )
        executor = (
            self._process_executor if item.work_unit.executor_kind == "process" else self._thread_executor
        )
        future = executor.submit(self._execute_work_unit, item.work_unit)
        # Defer callback attachment until after lock release to avoid immediate
        # callback re-entry (`add_done_callback` can invoke synchronously).
        self._pending_done_callbacks.append(
            PendingDoneCallback(
                future=future,
                plan_id=item.plan_id,
                stage_id=item.stage_id,
                unit_id=item.work_unit.unit_id,
                sem=sem,
            )
        )

    def _on_unit_done(
        self,
        plan_id: str,
        stage_id: str,
        unit_id: str,
        fut: Future[Any],
        sem: Semaphore,
    ) -> None:
        """Handle unit completion, advance stages, and resolve the plan future."""
        try:
            with self._state_lock:
                # Always return capacity token, even when state is already terminal/evicted.
                sem.release()
                plan_state = self._plan_states.get(plan_id)
                if plan_state is None:
                    return
                if plan_state.completion_future.done():
                    return

                stage_state = plan_state.stage_states[stage_id]
                exc = fut.exception()
                if exc is None:
                    stage_state.succeeded_unit_ids.add(unit_id)
                    if self._recorder is not None:
                        self._recorder.on_unit_done(plan_id, stage_id, unit_id, success=True)
                else:
                    stage_state.failed_unit_ids[unit_id] = exc
                    plan_state.failed_units[unit_id] = exc
                    if self._recorder is not None:
                        self._recorder.on_unit_done(
                            plan_id, stage_id, unit_id, success=False, error=f"{type(exc).__name__}: {exc}"
                        )

                    if plan_state.task_plan.fail_fast:
                        # Cancel queued-but-not-submitted work and fail immediately.
                        self._purge_plan_from_queues_locked(plan_id)
                        fail_message = (
                            f"TaskPlan {plan_id} failed fast at stage {stage_id} due "
                            f"to work unit {unit_id}: {exc}"
                        )
                        plan_state.completion_future.set_exception(RuntimeError(fail_message))
                        if self._recorder is not None:
                            self._recorder.on_plan_completed(plan_id, success=False, error=fail_message)
                        self._plan_states.pop(plan_id, None)
                        self._drain_queues_locked()
                        return

                if not stage_state.is_terminal():
                    # Stage still has in-flight work; only attempt to fill any open slots.
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
                        plan_state.completion_future.set_exception(RuntimeError(fail_message))
                        if self._recorder is not None:
                            self._recorder.on_plan_completed(plan_id, success=False, error=fail_message)
                    else:
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
        finally:
            # _on_unit_done can enqueue more work under lock; flush registrations
            # now so newly submitted futures get callbacks attached lock-free.
            self._flush_pending_done_callbacks()

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
                s=pending_callback.sem: self._on_unit_done(pid, sid, uid, fut, s)
            )

    def _purge_plan_from_queues_locked(self, plan_id: str) -> None:
        for queue_map in (self._process_queues, self._thread_queues):
            for priority in queue_map:
                retained = deque(item for item in queue_map[priority] if item.plan_id != plan_id)
                queue_map[priority] = retained

    @staticmethod
    def _execute_work_unit(work_unit: WorkUnit) -> Any:
        return work_unit.fn()
