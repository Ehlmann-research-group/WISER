import os

from typing import Optional

from concurrent.futures import ProcessPoolExecutor, Future
from multiprocessing.shared_memory import SharedMemory
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QObject, Signal

class MultiProcessingManager(QObject):
    # emits (task_id, result)
    finished = Signal(int, object)
    # emits (task_id, exception)
    error = Signal(int, Exception)

    def __init__(self, max_workers=None, parent=None):
        super().__init__(parent)
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        # ensure clean shutdown on app exit
        QApplication.instance().aboutToQuit.connect(self.shutdown)  # safe GUI-thread slot binding :contentReference[oaicite:2]{index=2}
        self._futures: dict[int, Future] = {}
        self._next_id = 0

    def submit(self, fn, *args, **kwargs) -> int:
        """
        Schedule a pickleable callable in a separate process.
        Returns a unique integer task ID.
        """
        task_id = self._next_id
        self._next_id += 1
        print(f"task ID: {task_id}")
        fut = self._executor.submit(fn, *args, **kwargs)  # uses internal call queue :contentReference[oaicite:3]{index=3}
        print(f"future: {fut}")
        self._futures[task_id] = fut
        # callbacks run in the submitting (GUI) thread :contentReference[oaicite:4]{index=4}
        fut.add_done_callback(lambda f, tid=task_id: self._on_done(tid, f))
        return task_id

    def _on_done(self, task_id: int, future: Future):
        """
        Internal callback: emit finished or error, then clean up.
        """
        print(f'DONE WITH: {task_id}')
        try:
            result = future.result()  # will raise if the task failed :contentReference[oaicite:5]{index=5}
            print(f'result: {result}')
            print(f'type result: {type(result)}')
            self.finished.emit(task_id, result)
        except Exception as exc:
            print(f"Error:\n{exc}")
            self.error.emit(task_id, exc)  # separate error signal
        finally:
            del self._futures[task_id]

    def get_future(self, task_id) -> Optional[Future]:
        if task_id in self._futures:
            return self._futures[task_id]
        return None

    def shutdown(self, wait: bool = True):
        """
        Shut down the pool: wait for running tasks if wait=True.
        Prevents orphaned worker processes :contentReference[oaicite:6]{index=6}.
        """
        print(f"WE SHUTTING DOWN")
        self._executor.shutdown(wait=wait)