import logging

from typing import Any, Callable, Dict, Optional, Union

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import multiprocessing as mp
import multiprocessing.connection as mp_conn
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

class ParallelTask(QThread):
    '''
    This class represents a long-running task that needs to execute "in the
    background" while the GUI continues to remain responsive.  Note that this
    is focused on  parallelism and not concurrency. We mean to ensure that long
    running tasks both don't affect the GUI and don't slow down the rest of
    WISER. Tasks submitted here are expected to have simple inputs and outputs
    due to inter-process communication restrictions.
    '''

    # Signal:  The task has been started.  The argument of the signal is this
    # task object.
    started = Signal(object)

    # Signal: The task's status. This can be used for progress or error messages or anything else.
    status = Signal(object)

    # Signal:  The task has finished.  The argument of the signal is this task
    # object.
    succeeded = Signal(object)

    # Signal:  The task has been cancelled/terminated.  The argument of the signal is this
    # task object.
    cancelled = Signal(object)

    # Signal:  The task has thrown an error.  The argument of the signal is this
    # task object.
    error = Signal(object)

    def __init__(self, operation: Callable = None, kwargs: Dict = {}):
        super().__init__()

        # The long-running operation to perform.
        self._operation = operation

        # Any keyword arguments to pass to the operation.
        self._kwargs: Dict = kwargs

        # Any exception that is thrown is stored in this field.
        self._error: Optional[Exception] = None

        # If the operation completes successfully and returns a value, this
        # field is set to the returned value.
        self._result: Any = None

        # Whether the task has been cancelled.
        self._cancelled = False

    def cancel(self, **kwargs):
        raise NotImplementedError("This method should be overridden by the subclass")

    def has_error(self):
        return (self._error is not None)

    def get_error(self):
        return self._error

    def get_result(self):
        return self._result

class ParallelTaskProcess(ParallelTask):
    '''
    This class is used to run a function in a separate process.

    Args:
        process (mp.Process): The process to use.
        return_queue (mp.Queue): The return queue to get the return value from the child process.
        operation (Callable): The function to run in the separate process.
        kwargs (Dict): The keyword arguments to pass to the function.
    '''

    def __init__(self, process: mp.Process = None, parent_conn: mp_conn.Connection = None, \
                 child_conn: mp_conn.Connection = None, return_queue: mp.Queue = None,
                 operation: Callable = None, kwargs: Dict = {}):
        super().__init__(operation, kwargs)

        # The process to use.
        self._process = process

        # The parent connection to get the return value from the child process.
        self._parent_conn = parent_conn

        # The child connection to send data to the child process.
        self._child_conn = child_conn

        # The return queue to get the return value from the child process.
        self._return_queue = return_queue

        self._process_id: int | None = None

        self._exit_code: int | None = None

    def cancel(self, **kwargs):
        try:
            self._process.terminate()
        except ValueError:
            # This error occurs when the process is already terminated
            # or hasn't started.
            pass

    def run(self):
        # Include the queue's reader in the wait set so we can drain results
        queue_reader = self._return_queue._reader  # Connection (private)
        self.started.emit(self)
        try:
            self._process.start()
            self._process_id = self._process.pid
            self._child_conn.close()
            while True:
                ready = mp_conn.wait([self._process.sentinel, queue_reader, self._parent_conn])
                # If there is a message available we read it. 
                if queue_reader in ready:
                    try:
                        msg = self._return_queue.get_nowait()
                        self._result = msg
                    except Exception as e:
                        logger.exception("Failed to read return value: %s", e)
                        self._result = None
                        self._error = e
                        self.error.emit(self)

                if self._parent_conn in ready:
                    try:
                        msg = self._parent_conn.recv()
                        self.status.emit(msg)
                    except (EOFError, OSError):
                        # Child closed its end; keep waiting for sentinel
                        pass
                    except Exception as e:
                        self._error = e
                        self.error.emit(self)
                # If the process is ready, we need to flush all the messages on the parent
                # pipe and close the process
                if self._process.sentinel in ready:
                    # drain any remaining messages without blocking
                    try:
                        while self._parent_conn.poll():
                            try:
                                msg = self._parent_conn.recv()
                                self.status.emit(msg)
                            except (EOFError, OSError):
                                break
                    except Exception:
                        pass
                    finally:
                        self._process.join()
                        self._exit_code = self._process.exitcode
                        self._process.close()
                    break
            # Once we reach here, the process has finished, so we get the stuff
            # on the return queue.
            try:
                # return queue should be empty, but if its not, we will get the result
                if not self._return_queue.empty():
                    self._result = self._return_queue.get()
            except Exception as e:
                logger.exception("Failed to read return value: %s", e)
                self._result = None
                self._error = e
                self.error.emit(self)
        except Exception as e:
            logger.exception(f'Error starting process {self._process_id}: {e}')
            self._error = e
            self.error.emit(self)
            return
        
        self._parent_conn.close()
        self._return_queue.close()
        if self._exit_code == 0:
            self.succeeded.emit(self)
        elif self._exit_code is not None and self._exit_code < 0:
            self.cancelled.emit(self)
        else:
            self.error.emit(self)
    
    def get_process_id(self) -> Union[int, None]:
        return self._process_id

class ParallelTaskProcessPool(ParallelTask):
    '''
    This class is unfinished. But it's goal is to use a ProcessPoolExecutor to run a function in a separate process.
    '''

    def __init__(self, process_pool_executor: ProcessPoolExecutor = None, 
                 operation: Callable = None, kwargs: Dict = {}):
        super().__init__(operation, kwargs)

        # The process pool executor to use.
        self._process_pool_executor = process_pool_executor

    def run(self):
        '''
        DO NOT DIRECTLY CALL THIS METHOD. Use .start() instead.
        '''
        self.started.emit(self)

        try:
            self._future = self._process_pool_executor.submit(self._operation, **self._kwargs)
            self._result = self._future.result()
        except Exception as e:
            self._error = e
            self.error.emit(self)
            return
        
        self.succeeded.emit(self)