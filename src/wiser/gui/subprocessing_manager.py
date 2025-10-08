import multiprocessing as mp
import multiprocessing.connection as mp_conn

from typing import Callable, Dict, List, Union

from concurrent.futures import ProcessPoolExecutor

from PySide2.QtCore import *

from wiser.gui.parallel_task import ParallelTaskProcess, ParallelTaskProcessPool
from wiser.utils.multiprocessing_context import CTX

import traceback

SENTINEL_RESULT = "__RESULT__"
SENTINEL_ERROR = "__ERROR__"


def child_trampoline(
    op: Callable, child_conn: mp_conn.Connection, return_queue: mp.Queue, **kwargs
):
    try:
        op(child_conn=child_conn, return_queue=return_queue, **kwargs)
    except Exception:
        tb = traceback.format_exc()
        # send both ways so you always see it
        try:
            child_conn.send(["process_error", {"type": "error", "traceback": tb}])
        except Exception:
            pass
        try:
            return_queue.put((SENTINEL_ERROR, tb))
        except Exception:
            pass
        # re-raise so exitcode is nonzero
        raise
    finally:
        try:
            child_conn.close()
        except Exception:
            pass


class ProcessManager(QObject):
    """
    This class is used to manage a single process. It is used to run a function in a separate process.
    and get the process ID for that process but also set up the communication between the parent and
    child processes. The function passed into this class should use this inter-process communication.
    Specifically, the function should use the child_conn to send data to the parent process and use the
    return_queue to get the return value from the child process. We use this class instead of
    MultiprocessingManager so we can get the process id of the process at creation time.

    Attributes:
        _parent_conn (mp.Pipe): The parent connection.
        _child_conn (mp.Pipe): The child connection.
        _return_q (mp.Queue): The return queue.
        _process (mp.Process): The process.
        _task (ParallelTaskProcess): The task.
        _pid (int): The process ID.

    Example Use:
    ```
    def operation(child_conn, return_queue, x):
        time.sleep(10)
        child_conn.send([1, 2, 'Running'])
        time.sleep(10)
        child_conn.send([2, 2, 'Running'])
        return_queue.put([x])

    process_manager = ProcessManager(operation, kwargs)
    process_manager.get_task().start()
    # This should be in a thread
    while process_manager.get_process().is_alive():
        child_comms = process_manager.get_parent_conn().recv()
        # You can use child_comms for status bars or other comms
        print(child_comms)
    result = process_manager.get_task().get_result()
    ```
    """

    _next_process_id = 0

    def __init__(self, operation: Callable, kwargs: Dict = {}):
        super().__init__()
        self._parent_conn, self._child_conn = CTX.Pipe(duplex=False)
        self._return_q = CTX.Queue()
        kwargs["op"] = operation
        kwargs["child_conn"] = self._child_conn
        kwargs["return_queue"] = self._return_q
        self._process = CTX.Process(target=child_trampoline, kwargs=kwargs)
        self._task = ParallelTaskProcess(
            self._process, self._parent_conn, self._child_conn, self._return_q
        )
        self._process_manager_id = type(self)._next_process_id
        type(self)._next_process_id += 1

    def get_task(self) -> ParallelTaskProcess:
        return self._task

    def start_task(self, blocking=False):
        self._task.start()
        if blocking:
            self._task.wait()

    def get_pid(self) -> Union[int, None]:
        return self._task.get_process_id()

    def get_process(self) -> mp.Process:
        return self._process

    def get_process_manager_id(self) -> int:
        return self._process_manager_id


class MultiprocessingManager(QObject):
    """
    Requirements:
    - Takes in a function to run asychronously, prepares that function to be run
    - Lets user set up callback function, error, cancelled, started, finished signals.

    - Multiprocmanager can make parallel running tasks, users should use a class called
    - multiprocfunction that has a pipe and this is passed into create task. When a task
    - is started, multiproc function gets the process id of the task and
    """

    def __init__(self):
        super().__init__()
        self._process_pool_executor = ProcessPoolExecutor()

        # You can check to see if the task has finished by called
        # task.isFinished()
        self._tasks: List[ParallelTaskProcessPool] = []

        self._next_process_pool_id = 0

    def get_next_process_pool_id(self):
        id = self._next_process_pool_id
        self._next_process_pool_id += 1
        return id

    def create_task(self, operation: Callable, kwargs: Dict = {}):
        task = ParallelTaskProcessPool(self._process_pool_executor, operation, kwargs)
        self._tasks.append(task)
        return task
