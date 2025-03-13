import sys

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class FunctionEvent(QEvent):
    """
    A custom event carrying a single callable (func) that we want
    to run inside the Qt event loop.
    """
    def __init__(self, func):
        # We use the custom event type stored on MyWidget.functionEventType.
        super().__init__(TestingWidget.functionEventType)
        self._func = func

    def run(self):
        """Execute the function carried by this event."""
        self._func()

class TestingWidget(QWidget):
    # Register a unique QEvent type
    functionEventType = QEvent.Type(QEvent.registerEventType())

    def __init__(self):
        super().__init__()

    def event(self, event):
        """
        Override the base event handler to catch our FunctionEvent
        and run its stored callable.
        """
        if event.type() == TestingWidget.functionEventType:
            event.run()   # call the stored function
            return True   # tell Qt we've handled this event
        # Otherwise, let the normal event handling continue
        return super().event(event)
