"""Process-aware logging configuration for WISER.

WISER writes a single rotating log file (``wiser.log``) in the config directory.
Rotation renames ``wiser.log`` -> ``wiser.log.1``, and on Windows that rename
fails with ``PermissionError: [WinError 32]`` if *any* other handle to the file
is open.  Keeping the log file open in exactly one process is therefore a
correctness requirement, not a tidiness preference.  See issue #502.

This module owns the two things that requirement needs:

* :func:`is_worker_process` -- an *import-time* answer to "am I a multiprocessing
  worker?", which is harder than it looks (see below).
* :class:`SafeRotatingFileHandler` -- a rotating handler that degrades to "keep
  writing to the current file" instead of raising, for the second-handle cases
  we cannot prevent (antivirus, a search indexer, a tail'ing editor, or a second
  copy of WISER started by the user).
"""

from __future__ import annotations

import logging
import logging.config
import multiprocessing
import multiprocessing.spawn
import os
import sys
import time
from logging.handlers import RotatingFileHandler

# How long to wait before retrying a rollover that failed because another
# process held the log file open. Without a cooldown, every subsequent record
# re-attempts the rename (the file is still over maxBytes) and each failure
# prints a "--- Logging error ---" traceback, which is the console spam in #502.
ROLLOVER_RETRY_SECONDS = 60.0

# Log volume drivers. Numba emits a DEBUG record per bytecode instruction, per
# SSA block, per typing pass while compiling; with the root logger at DEBUG that
# is tens of MB per compile, and it is what drives wiser.log to its rotation
# threshold in the first place.
NOISY_LIBRARY_LOGGERS = ("matplotlib", "numba")


def is_worker_process() -> bool:
    """Return True if this interpreter is a multiprocessing child, not the app.

    This must be answerable *while the main module is still executing its
    top-level code*, which rules out the obvious check.
    ``multiprocessing.parent_process()`` reads a module global that is assigned
    inside ``BaseProcess._bootstrap()``, and in a child that runs *after* the
    main module has been re-executed:

    * Frozen (PyInstaller): the child is a fresh ``WISER.exe`` launched with
      ``--multiprocessing-fork``.  The bootloader runs ``wiser/__main__.py`` from
      the top, and only the ``multiprocessing.freeze_support()`` call at the very
      bottom hands control to ``spawn_main() -> _bootstrap()``.
    * Source: ``spawn._main()`` calls ``prepare()`` -- which re-runs the main
      module -- before it calls ``self._bootstrap()``.

    Either way ``parent_process()`` still returns ``None`` throughout the
    top-level code, so a child that asks it believes it is the main process.

    The checks below are the ones that *are* true that early, in the same order
    the stdlib and PyInstaller use them.
    """
    # Already bootstrapped (fork children, or any import after startup).
    if multiprocessing.parent_process() is not None:
        return True

    # `python -m ...` / frozen spawn child: argv is
    # [exe, "--multiprocessing-fork", "pipe_handle=...", ...]. This is the same
    # test `multiprocessing.spawn.freeze_support()` and PyInstaller's
    # pyi_rth_multiprocessing hook use to recognize a child.
    if multiprocessing.spawn.is_forking(sys.argv):
        return True

    # Helper processes the frozen bootloader also starts by re-running the entry
    # script: the resource tracker and the forkserver (macOS / Linux).
    if len(sys.argv) >= 2 and sys.argv[-2] == "-c":
        if sys.argv[-1].startswith(
            ("from multiprocessing.resource_tracker", "from multiprocessing.forkserver")
        ):
            return True

    # Source spawn child re-running the main module: `spawn._main()` sets
    # `_inheriting` on the current process before `prepare()` re-executes the
    # main module, and deletes it afterwards.
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        return True

    return False


class SafeRotatingFileHandler(RotatingFileHandler):
    """A RotatingFileHandler that survives a rename blocked by another handle.

    Windows refuses to rename an open file, so a rollover raises
    ``PermissionError`` (WinError 32) whenever something else holds the log open.
    The stock handler leaves ``self.stream`` closed and lets the error escape to
    ``handleError``, which prints a full traceback -- and because the file is
    still oversized, the *next* record tries to roll over again.  That feedback
    loop is the multi-minute console spam users see.

    Here a failed rollover is absorbed: the stream is reopened on the existing
    file and rollover is not re-attempted until ``retry_seconds`` has passed, so
    logging continues uninterrupted and the log simply grows past ``maxBytes``
    until whoever holds the second handle lets go.
    """

    def __init__(self, *args, retry_seconds: float = ROLLOVER_RETRY_SECONDS, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._retry_seconds = retry_seconds
        self._rollover_blocked_until = 0.0
        self._reported_blocked_rollover = False

    def shouldRollover(self, record: logging.LogRecord) -> int:
        if time.monotonic() < self._rollover_blocked_until:
            return 0
        return super().shouldRollover(record)

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except OSError as exc:
            self._rollover_blocked_until = time.monotonic() + self._retry_seconds

            # doRollover() closes the stream before renaming, so a failure part
            # way through leaves us with nowhere to write. Reattach to the file
            # we failed to rename; it still exists, and append mode picks up
            # where we left off.
            if self.stream is None:
                self.stream = self._open()

            # Say this once per handler rather than once per record: the whole
            # point of the class is to not spam the console.
            if not self._reported_blocked_rollover:
                self._reported_blocked_rollover = True
                sys.stderr.write(
                    f"WISER: could not rotate log file {self.baseFilename} "
                    f"({exc}); another process is holding it open. Continuing to "
                    f"log to the existing file.\n"
                )


def configure_logging(logfile_path: str) -> None:
    """Install WISER's logging configuration for the current process.

    The main process logs WARNING+ to stderr and DEBUG+ to a rotating
    ``logfile_path``.  Worker processes get no handlers at all: they must not
    hold a second handle on the log file (#502), and worker task code prints
    rather than logs.
    """
    if is_worker_process():
        _configure_worker_logging()
        return

    logging.config.dictConfig(
        {
            "version": 1,
            "formatters": {
                "simpleFormatter": {
                    "format": "%(asctime)s %(levelname)-5s %(name)s : %(message)s",
                },
            },
            "handlers": {
                "consoleHandler": {
                    "class": "logging.StreamHandler",
                    "level": "WARNING",
                    "formatter": "simpleFormatter",
                    "stream": sys.stderr,
                },
                "fileHandler": {
                    # Passed as a callable rather than a dotted path so this
                    # keeps working in the PyInstaller build, where dictConfig's
                    # string import of a bundled module is one more thing to go
                    # wrong.
                    "()": SafeRotatingFileHandler,
                    "level": "DEBUG",
                    "formatter": "simpleFormatter",
                    "filename": logfile_path,
                    "maxBytes": 10_000_000,
                    "backupCount": 5,
                },
            },
            "loggers": {
                "root": {
                    "level": "DEBUG",
                    "handlers": ["consoleHandler", "fileHandler"],
                },
                # Capped at WARNING so their DEBUG firehose does not fill the log
                # (and force rollovers) on every numba compile / plot redraw.
                **{name: {"level": "WARNING"} for name in NOISY_LIBRARY_LOGGERS},
            },
        }
    )


def _configure_worker_logging() -> None:
    """Give a worker process a root logger that writes nowhere.

    A NullHandler keeps ``logging.lastResort`` from dumping worker records onto
    the console, and the WARNING level means numba's per-instruction DEBUG calls
    short-circuit in ``isEnabledFor()`` instead of building records that are only
    going to be discarded.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if not isinstance(handler, logging.NullHandler):
            root_logger.removeHandler(handler)
    if not any(isinstance(h, logging.NullHandler) for h in root_logger.handlers):
        root_logger.addHandler(logging.NullHandler())
    root_logger.setLevel(logging.WARNING)


def open_log_file_handles(logfile_path: str) -> list[logging.Handler]:
    """Return the handlers in this process that hold ``logfile_path`` open.

    Used by the tests to assert the invariant this module exists to maintain:
    exactly one process opens the log file.
    """
    target = os.path.abspath(logfile_path)
    return [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == target
    ]
