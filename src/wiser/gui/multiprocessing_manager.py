from typing import Callable, Dict, List

from concurrent.futures import ProcessPoolExecutor

from PySide2.QtCore import *

from wiser.gui.parallel_task import ParallelRunningTask

class MultiprocessingManager(QObject):
    '''
    Requirements:
    - Takes in a function to run asychronously, prepares that function to be run
    - Lets user set up callback function, error, cancelled, started, finished signals. 
    '''
    
    def __init__(self):
        super().__init__()
        self._process_pool_executor = ProcessPoolExecutor()

        # You can check to see if the task has finished by called
        # task.isFinished()
        self._tasks: List[ParallelRunningTask] = []

        self._next_process_pool_id = 0

    def get_next_process_pool_id(self):
        id = self._next_process_pool_id
        self._next_process_pool_id += 1
        return id

    def create_task(self, operation: Callable, kwargs: Dict = {}):
        task = ParallelRunningTask(self._process_pool_executor, operation, kwargs)
        self._tasks.append(task)
        return task
