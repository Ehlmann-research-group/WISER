from typing import TYPE_CHECKING

from PySide2.QtCore import QObject

import os
import uuid

from wiser.gui.activity_monitor import ActivityMonitorDialog
from wiser.utils.storage_service import StorageService
from wiser.utils.task_system import PlanningContext, SimpleChunkingPolicy, TaskManager, TaskPlanner
from wiser.utils.work_scheduler import (
    SchedulerConfig,
    WorkScheduler,
    SchedulerConcurrencyMode,
    PRIORITY_LANE_COUNT,
)
from wiser.utils.worker_runtime import initialize_process_storage_client, reset_process_storage_client

if TYPE_CHECKING:
    from wiser.gui.activity_monitor import ActivityMonitorDialog


class AppServices(QObject):
    def __init__(
        self,
        activity_monitor: "ActivityMonitorDialog" = None,
        parent=None,
    ):
        super().__init__()
        self._closed = False
        self._debug_uuid = uuid.uuid4().hex
        self._storage_service = StorageService(
            ram_byte_limit=2_000_000_000,
            # TODO (Joshua G-K): Change this to be based on remaining data at app start up time
            disk_byte_limit=16_000_000_000,
        )

        # Initialize the storage client for the main process
        listener_address, listener_authkey = self._storage_service.get_connection_bootstrap()
        initialize_process_storage_client(listener_address, listener_authkey)

        if activity_monitor is not None:
            self._owns_activity_monitor = False
            self._task_manager = TaskManager(activity_monitor)
        else:
            activity_monitor = ActivityMonitorDialog(parent=parent)
            self._owns_activity_monitor = True
            self._task_manager = TaskManager(activity_monitor)
        self._activity_monitor = activity_monitor

        scheduler_mode = os.getenv("WISER_LOW_CONCURRENCY_SCHEDULER")
        if scheduler_mode == SchedulerConcurrencyMode.HIGH.value:
            scheduler_config = SchedulerConfig()
        else:
            scheduler_config = SchedulerConfig(
                _process_budget=PRIORITY_LANE_COUNT, _thread_budget=PRIORITY_LANE_COUNT
            )
        # recorder = RecordingWorkScheduler()
        self._scheduler = WorkScheduler(
            config=scheduler_config,
            storage_service=self._storage_service,
            task_manager=self._task_manager,
            # recorder=recorder,
        )

        self._task_planner = TaskPlanner(
            PlanningContext(
                sched_cfg=scheduler_config,
                storage=self._storage_service,
                chunking_policy=SimpleChunkingPolicy(),
            )
        )

    @property
    def storage_service(self):
        return self._storage_service

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def task_planner(self):
        return self._task_planner

    @property
    def task_manager(self):
        return self._task_manager

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        try:
            self._scheduler.shutdown(wait=True)
        finally:
            try:
                self._storage_service.close()
            finally:
                try:
                    reset_process_storage_client()
                finally:
                    if self._owns_activity_monitor and self._activity_monitor is not None:
                        self._activity_monitor.close()
