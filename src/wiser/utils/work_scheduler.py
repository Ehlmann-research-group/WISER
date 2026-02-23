SCHEDULER_PROCESS_BUDGET = 6
SCHEDULER_RAM_BUDGET = 4000000000
SCHEDULER_THREAD_BUDGET = 32


class SchedulerConfig:
    """
    Specifies the computer resources that we can use
    """

    def __init__(self):
        self._process_budget = SCHEDULER_PROCESS_BUDGET
        self._thread_budget = SCHEDULER_THREAD_BUDGET
        self._ram_budget = SCHEDULER_RAM_BUDGET


class WorkScheduler:
    def __init__(self, config):
        self._config = config
