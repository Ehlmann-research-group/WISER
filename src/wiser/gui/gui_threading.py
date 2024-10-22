from PySide2.QtCore import QRunnable, QThreadPool, Signal, QObject, QThread

MAX_THREAD_COUNT = 1

thread_pool = QThreadPool()
thread_pool.setMaxThreadCount(MAX_THREAD_COUNT)

class WorkerSignals(QObject):
    update_signal = Signal(str)

class Worker(QRunnable):

    def __init__(self, func, *args, **kwargs):
        super(Worker, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()  # Set up signal

    def run(self):
        result = self.func(*self.args, **self.kwargs)

class WorkerThread(QThread):

    def __init__(self, func, *args, **kwargs):
        super(WorkerThread, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        result = self.func(*self.args, **self.kwargs)