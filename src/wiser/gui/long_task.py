from typing import Any, Dict

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class LongRunningTask(QThread):
    '''
    This class represents a long-running task that needs to execute "in the
    background" while the GUI continues to remain responsive.  Note that this
    is focused on concurrency, not parallelism; ensuring that long running tasks
    don't prevent the main GUI event loop from responding to user interactions.
    '''

    # Signal:  The task has been started.  The argument of the signal is this
    # task object.
    started = Signal(object)

    # Signal:  The task has finished.  The argument of the signal is this task
    # object.
    finished = Signal(object)


    def __init__(self, operation, args: Dict = {}):
        super().__init__()

        # The long-running operation to perform.
        self._operation = operation

        # Any keyword arguments to pass to the operation.
        self._args: Dict = args

        # Any exception that is thrown is stored in this field.
        self._error: Optional[Exception] = None

        # If the operation completes successfully and returns a value, this
        # field is set to the returned value.
        self._result: Any = None


    def run(self):
        self.started.emit(self)
        try:
            self._result = self._operation(**self._args)
        except Exception as e:
            self._error = e

        self.finished.emit(self)

    def has_error(self):
        return (self._error is not None)

    def get_error(self):
        return self._error

    def get_result(self):
        return self._result
