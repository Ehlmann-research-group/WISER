"""Decorator for running WiserTestModel methods in the Qt event loop.

This module provides a decorator that ensures a function is executed
synchronously within the Qt main thread using a custom FunctionEvent.
Useful for safely interacting with GUI components during testing.
"""
from typing import TYPE_CHECKING
import functools

from .test_event_loop_functions import FunctionEvent

if TYPE_CHECKING:
    from .test_model import WiserTestModel


def run_in_wiser_decorator(func):
    """Decorator to run a function inside the Wiser application's event loop.

    This decorator ensures that the decorated function is executed within the Qt
    main thread by posting a custom `FunctionEvent` to the application's event loop
    using the `testing_widget`. The function's result is returned after the event
    is processed by the event loop.

    This is particularly useful for running GUI-affecting logic or Qt-bound code
    from contexts like test threads or background threads where direct calls would
    be unsafe.

    Args:
        func (Callable): The method of `WiserTestModel` to be executed in the Qt event loop.

    Returns:
        Callable: A wrapped function that runs the original function in the event loop
        and returns its result.

    Raises:
        Any exception raised by the wrapped function will propagate normally.
    """

    @functools.wraps(func)
    def arg_wrapper(self: "WiserTestModel", *args, **kwargs):
        result_container: dict = {}

        def func_event():
            result_container["value"] = func(self, *args, **kwargs)

        function_event = FunctionEvent(func_event)
        self.app.postEvent(self.testing_widget, function_event)
        self.run()
        return result_container.get("value")

    return arg_wrapper
