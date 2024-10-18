from PySide2.QtCore import QRunnable, QThreadPool, Signal, QObject
from PySide2.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
import time

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