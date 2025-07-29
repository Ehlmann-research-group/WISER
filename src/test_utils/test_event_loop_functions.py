"""
This module provides two important classes. A class that carries a single
callable (FunctionEvent) that we want to be called in the event loop and
a class that lets us put a widget inside of the application (TestingWidget)
so that we can run testing code more easily.
"""
import sys

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

class FunctionEvent(QEvent):
    """
    A class to wrap a callable

    A custom event carrying a single callable (func) that we want
    to run inside the Qt event loop. Some functionality we can only
    test if the event loop is running, which is why this is needed.

    Args:
        func (callable): A callable function that you want called in the event loop

    Attributes:
        func (callable): A callable function that you want called in the event loop

    """
    def __init__(self, func):
        # We use the custom event type stored on MyWidget.functionEventType.
        super().__init__(TestingWidget.functionEventType)
        self._func = func

    def run(self):
        """Execute the function carried by this event."""
        self._func()

class TestingWidget(QWidget):
    """
    A transparent, frameless QWidget that acts as an event proxy to execute 
    callables via a custom QEvent type.

    This widget does not accept mouse events (they pass through it) and is
    visually invisible. It is useful for dispatching function calls via the
    event system, typically from other threads or contexts where direct
    interaction with the main GUI thread is not allowed.

    Attributes:
        functionEventType (QEvent.Type): A custom event type used to register and 
            identify events carrying callable functions to be executed.
    """
    # Register a unique QEvent type
    functionEventType = QEvent.Type(QEvent.registerEventType())

    def __init__(self):
        super().__init__()
        # This attribute ensures the widget does NOT handle mouse events
        # (so clicks pass through to whatever is beneath).
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # Remove any background to make it visually invisible
        self.setStyleSheet("background: transparent;")

        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

    def event(self, event):
        """
        Handle incoming events and process custom FunctionEvents.

        If the event is of type `functionEventType`, its `run()` method is
        called to execute the stored function, and the event is marked as handled.

        Args:
            event (QEvent): The event to be processed.

        Returns:
            bool: True if the event was handled; otherwise, delegates to
            the base class implementation.
        """
        if event.type() == TestingWidget.functionEventType:
            event.run()
            return True

        return super().event(event)
