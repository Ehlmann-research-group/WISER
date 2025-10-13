import functools
import inspect
import logging

from typing import Optional


def log_exceptions(logger: Optional[logging.Logger] = None):
    """
    A decorator to log exceptions thrown by a function, on the specified logger.
    The decorator takes one optional argument ``logger``, allowing the specific
    logger to use to be specified.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                # Log the exception, then reraise it.
                if inspect.ismethod(func):
                    msg = f"{func.__self__.__class__.__name__}.{func.__name__} " + "raised an exception"
                else:
                    msg = f"{func.__name__} raised an exception"

                if logger is not None:
                    logger.exception(msg)
                else:
                    logging.exception(msg)

                raise

        return wrapper

    return decorator
