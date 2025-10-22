from PySide2.QtCore import QRunnable, Signal, QObject


class WorkerSignals(QObject):
    update = Signal(str)
    finished = Signal(object)
    error = Signal(BaseException)


class Worker(QRunnable):
    def __init__(self, current_call, total_calls, func, *args, **kwargs):
        """
        Callback is a function assumed to have no arguments
        """
        super(Worker, self).__init__()
        self.func = func
        self.current_call = current_call  # The current index of the call we are on
        self.total_calls = total_calls  # The total index of the call we are on
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.result = None

    def run(self):
        # print(f"args: {self.args}")
        # print(f"kwargs: {self.kwargs}")
        try:
            self.result = self.func(*self.args, **self.kwargs)

            self.signals.finished.emit((self.current_call, self.total_calls, self.result))
        except BaseException as e:
            print(f"Error: \n {e}")
            self.signals.error.emit(e)
