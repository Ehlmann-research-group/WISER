from typing import Any, Callable, Dict, Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from concurrent.futures import ProcessPoolExecutor

class ParallelRunningTask(QThread):
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

    # Signal:  The task has finished.  The argument of the signal is this task
    # object.
    succeeded = Signal(object)

    # Signal:  The task has been cancelled.  The argument of the signal is this
    # task object and a boolean indicating whether the task was cancelled.
    cancelled = Signal(object, bool)

    # Signal:  The task has thrown an error.  The argument of the signal is this
    # task object.
    error = Signal(object)

    def __init__(self, process_pool_executor: ProcessPoolExecutor = None, 
                 operation: Callable = None, kwargs: Dict = {}):
        super().__init__()

        # The process pool executor to use.
        self._process_pool_executor = process_pool_executor

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

    def cancel(self):
        if self._future is not None:
            self._future.cancel()
            self.cancelled.emit(self, True)
        else:
            self.cancelled.emit(self, False)

    def has_error(self):
        return (self._error is not None)

    def get_error(self):
        return self._error

    def get_result(self):
        return self._result