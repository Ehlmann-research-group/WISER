import os
import threading
from typing import Optional, Dict, Tuple

from concurrent.futures import ThreadPoolExecutor, Future
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QObject, Signal

class MultiThreadingManager(QObject):
    """
    Manages background work on a ThreadPoolExecutor and forwards
    results or errors back to the GUI thread via Qt signals.
    """
    finished = Signal(int, object)   # emits (task_id, result)
    error    = Signal(int, Exception)  # emits (task_id, exception)

    def __init__(self, max_workers: Optional[int]=None, parent=None):
        super().__init__(parent)
        # Default to number of CPUs if not specified
        if max_workers is None:
            max_workers = os.cpu_count() or 1

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[int, Future] = {}
        self._next_id = 0
        self._lock = threading.Lock()

        # Ensure clean shutdown when the app quits
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def submit(self, fn, *args, **kwargs) -> Tuple[int, Future]:
        """
        Schedule a callable on the thread pool.
        Returns a unique integer task ID.
        """
        # Allocate a new task ID under lock
        with self._lock:
            task_id = self._next_id
            self._next_id += 1

        # Submit the work
        future = self._executor.submit(fn, *args, **kwargs)

        # Track it and register our callback
        with self._lock:
            self._futures[task_id] = future
        future.add_done_callback(lambda fut, tid=task_id: self._on_done(tid, fut))

        return task_id, future

    def _on_done(self, task_id: int, future: Future):
        """
        Internal callback: emit finished or error, then clean up.
        Runs in the worker thread but emits signals to the GUI.
        """
        try:
            result = future.result()      # May raise if the task errored
            self.finished.emit(task_id, result)
        except Exception as exc:
            self.error.emit(task_id, exc)
        finally:
            # Remove from our dict under lock
            with self._lock:
                self._futures.pop(task_id, None)

    def get_future(self, task_id: int) -> Optional[Future]:
        """
        Retrieve the Future for a given task ID, or None if unknown.
        """
        with self._lock:
            return self._futures.get(task_id)

    def shutdown(self, wait: bool = True):
        """
        Shut down the thread pool. If wait=True, will block until tasks finish.
        """
        self._executor.shutdown(wait=wait)
