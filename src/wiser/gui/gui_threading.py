from PySide2.QtCore import (QRunnable, QThreadPool, Signal, \
                            QObject)

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

    def run(self):
        self.result = self.func(*self.args, **self.kwargs)
        
        self.signals.finished.emit(self.result)
