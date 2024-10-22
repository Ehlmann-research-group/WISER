from PySide2.QtCore import (QRunnable, QThreadPool, Signal, \
                            QObject, QThread, QEventLoop)

MAX_THREAD_COUNT = 4

thread_pool = QThreadPool()
thread_pool.setMaxThreadCount(MAX_THREAD_COUNT)

class WorkerSignals(QObject):
    update = Signal(str)
    finished = Signal(object)

class Worker(QRunnable):

    def __init__(self, func, *args, **kwargs):
        '''
        Callback is a function assumed to have no arguments
        '''
        super(Worker, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()  # Set up signal
        self.result = None
        # self.callback = callback

    def run(self):
        print(f"args: {self.args}")
        print(f"kwargs: {self.kwargs}")
        self.result = self.func(*self.args, **self.kwargs)
        # if self.kwargs:
        #     self.result = self.func(*self.args, **self.kwargs)
        # else:
        #     print("kwargs is empty")
        #     self.result = self.func(*self.args)
        # if self.callback is not None:
        #     self.callback(self.result)
        
        self.signals.finished.emit(self.result)

class TrackedEventLoop(QEventLoop):
    # Class-level set to track active event loops
    active_event_loops = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add this instance to the set of active event loops
        TrackedEventLoop.active_event_loops.add(self)

    def exec_(self):
        """Override exec_ to start the event loop and track it."""
        print(f"Num active loops from exec_ {len(TrackedEventLoop.active_event_loops)}")
        try:
            return super().exec_()
        except BaseException as e:
            print(f"Error: {e}")
            self.quit()
        # finally:
        #     # Ensure that we remove this event loop after it is finished
        #     print("REMOVING AFTER EXEC AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        #     self._remove_from_active_loops()

    def quit(self):
        """Override quit to stop the event loop and remove it from tracking."""
        super().quit()
        self._remove_from_active_loops()

    def _remove_from_active_loops(self):
        """Helper method to remove this instance from the active loops set."""
        print(f"Num active loops from before in _quit {len(TrackedEventLoop.active_event_loops)}")
        if self in TrackedEventLoop.active_event_loops:
            print("Self removed from active loops")
            TrackedEventLoop.active_event_loops.remove(self)
            print(f"Num active loops from _remove _quit {len(TrackedEventLoop.active_event_loops)}")

    @classmethod
    def quit_all_active_loops(cls):
        """Quit all active event loops."""
        print("================================================TRYING TO QUIT=====================================")
        while cls.active_event_loops:
            print("Active loop??????????????????????????????????????????????????????")
            event_loop = cls.active_event_loops.pop()
            event_loop.quit()

class WorkerThread(QThread):

    def __init__(self, func, *args, **kwargs):
        super(WorkerThread, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        result = self.func(*self.args, **self.kwargs)