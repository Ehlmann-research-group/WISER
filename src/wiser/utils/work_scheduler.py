from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Semaphore
from typing import Any, Callable, Dict, TYPE_CHECKING

from .primitives import PriorityClass
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


def _priority_weight(priority: PriorityClass) -> float:
    if priority == PriorityClass.INTERACTIVE:
        return 1.0 / 2.0
    if priority == PriorityClass.RENDER:
        return 1.0 / 3.0
    return 1.0 / 6.0


def _allocate_priority_tokens(budget: int) -> Dict[PriorityClass, int]:
    if budget < 3:
        raise ValueError(f"Budget must be >= 3 to guarantee non-zero per priority, got {budget}")

    priorities = [PriorityClass.INTERACTIVE, PriorityClass.RENDER, PriorityClass.BACKGROUND]
    allocation: Dict[PriorityClass, int] = {p: 1 for p in priorities}
    remaining = budget - len(priorities)
    if remaining == 0:
        return allocation

    weighted = {p: remaining * _priority_weight(p) for p in priorities}
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
    def __init__(self, config: SchedulerConfig, storage_service: "StorageService"):
        self._config = config
        self._process_budget = int(self._config._process_budget)
        self._thread_budget = int(self._config._thread_budget)

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
