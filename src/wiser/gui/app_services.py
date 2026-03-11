from typing import TYPE_CHECKING

from PySide2.QtCore import QObject

from wiser.gui.activity_monitor import ActivityMonitorWidget
from wiser.utils.storage_client import StorageClient
from wiser.utils.storage_service import StorageService
from wiser.utils.task_system import PlanningContext, SimpleChunkingPolicy, TaskManager, TaskPlanner
from wiser.utils.work_scheduler import SchedulerConfig, WorkScheduler
from wiser.utils.worker_runtime import initialize_process_storage_client

if TYPE_CHECKING:
    from wiser.gui.activity_monitor import ActivityMonitorWidget


class AppServices(QObject):
    def __init__(self, activity_monitor: "ActivityMonitorWidget", parent=None):
        self._storage_service = StorageService(
            ram_byte_limit=2_000_000_000,
            # TODO (Joshua G-K): Change this to be based on remaining data at app start up time
            disk_byte_limit=16_000_000_000,
        )

        # Initialize the storage client for the main process
        listener_address, listener_authkey = self._storage_service.get_connection_bootstrap()
        initialize_process_storage_client(listener_address, listener_authkey)

        if activity_monitor is not None:
            self._task_manager = TaskManager(activity_monitor)
        else:
            activity_monitor = ActivityMonitorWidget()
            self._task_manager = TaskManager(activity_monitor)

        scheduler_config = SchedulerConfig()
        self._scheduler = WorkScheduler(
            config=scheduler_config,
            storage_service=self._storage_service,
            task_manager=self._task_manager,
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
