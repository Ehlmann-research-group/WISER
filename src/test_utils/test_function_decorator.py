from typing import TYPE_CHECKING
import functools

from.test_event_loop_functions import FunctionEvent

if TYPE_CHECKING:
    from .test_model import WiserTestModel

def run_in_wiser_decorator(func):
    @functools.wraps(func)
    def arg_wrapper(self: 'WiserTestModel', *args, **kwargs):
        def func_event():
            result = func(self, *args, **kwargs)

            return result

        function_event = FunctionEvent(func_event)
        self.app.postEvent(self.testing_widget, function_event)
        self.run()
    return arg_wrapper
