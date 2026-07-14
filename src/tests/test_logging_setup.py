"""Regression tests for the wiser.log rotation crash on Windows (#502).

The failure in the wild: the scheduler's pool workers each opened their own
handle on ``wiser.log``, and when one of them tried to roll the file over,
Windows refused to rename a file the other processes still had open:

    PermissionError: [WinError 32] The process cannot access the file because
    it is being used by another process:
    '...\\wiser.log' -> '...\\wiser.log.1'

``TestWorkerDoesNotOpenTheLogFile`` is the regression test: it runs the real
``wiser/__main__.py`` in a subprocess set up exactly as a spawned worker is, and
asserts the worker opens no handle on the log file.  It fails against the guard
that shipped on main.

``TestRolloverWithForeignHandle`` covers the second-handle cases we cannot
prevent (antivirus, an editor tailing the log, a second copy of WISER), where
rotation must degrade instead of melting down.
"""

import json
import logging
import logging.handlers
import os
import subprocess
import sys

import pytest

import wiser
from wiser.logging_setup import SafeRotatingFileHandler
from wiser.utils.multiprocessing_context import CTX

pytestmark = [pytest.mark.functional, pytest.mark.smoke, pytest.mark.regression]

WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Only Windows refuses to rename a file that another process holds open.",
)

# Runs wiser/__main__.py's top-level code -- the logging configuration under test --
# without calling main(), then reports what that left behind. importlib is used
# rather than `python -m wiser` precisely so the `if __name__ == "__main__"` block
# (which starts Qt) does not run.
PROBE_SOURCE = """
import importlib, json, logging, os, sys

importlib.import_module("wiser.__main__")

from wiser.gui.app_config import get_wiser_config_dir

logfile = os.path.join(get_wiser_config_dir(), "wiser.log")
print("WISER_PROBE " + json.dumps({
    "argv": sys.argv,
    "log_file_handlers": [
        os.path.basename(h.baseFilename)
        for h in logging.getLogger().handlers
        if isinstance(h, logging.FileHandler)
    ],
    "log_file_exists": os.path.exists(logfile),
    "numba_debug_enabled": logging.getLogger("numba.core.ssa").isEnabledFor(logging.DEBUG),
    "wiser_debug_enabled": logging.getLogger("wiser.gui").isEnabledFor(logging.DEBUG),
}))
"""


def _run_probe(tmp_path, *extra_argv: str) -> dict:
    """Run PROBE_SOURCE in a fresh interpreter, with WISER's config dir in tmp_path.

    `extra_argv` lands in the child's sys.argv after the `-c`, which is how we
    reproduce a spawned worker: the frozen bootloader launches
    `WISER.exe --multiprocessing-fork pipe_handle=...`, and the entry script runs
    top-to-bottom with that argv long before multiprocessing takes control.
    """
    env = dict(os.environ)
    # get_app_data_dir() builds the config dir from these (LOCALAPPDATA on
    # Windows, HOME elsewhere), so the probe writes any log into tmp_path.
    env["LOCALAPPDATA"] = str(tmp_path)
    env["HOME"] = str(tmp_path)
    env["USERPROFILE"] = str(tmp_path)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(wiser.__file__)))
    env.pop("QT_QPA_PLATFORM", None)

    result = subprocess.run(
        [sys.executable, "-c", PROBE_SOURCE, *extra_argv],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    for line in result.stdout.splitlines():
        if line.startswith("WISER_PROBE "):
            return json.loads(line[len("WISER_PROBE ") :])
    raise AssertionError(
        f"probe did not report a result (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


@pytest.mark.skipif(
    getattr(sys, "frozen", False),
    reason="The probe needs `python -c`; sys.executable is WISER.exe in a frozen run.",
)
class TestWorkerDoesNotOpenTheLogFile:
    """Only one process may hold wiser.log open, or rotation cannot rename it."""

    def test_worker_opens_no_handle_on_the_log_file(self, tmp_path):
        """The regression test for #502.

        This is the state a scheduler pool worker is really in: in the frozen
        build it is a fresh WISER.exe launched with --multiprocessing-fork, whose
        bootloader executes wiser/__main__.py from the top and only reaches the
        freeze_support() call at the bottom afterwards.

        The guard that shipped on main asked `multiprocessing.parent_process() is
        None` to detect this, but that global is not assigned until
        BaseProcess._bootstrap() -- which the worker reaches *through* the bottom
        of __main__.py. So the worker reported no parent, believed it was the app,
        and opened wiser.log. This test fails against that guard.
        """
        probe = _run_probe(tmp_path, "--multiprocessing-fork", "pipe_handle=123", "parent_pid=456")

        assert probe["log_file_handlers"] == [], (
            "a worker process opened a handle on the log file; when the app "
            "rotates wiser.log, Windows will refuse the rename (WinError 32)"
        )
        assert not probe["log_file_exists"], "a worker created the log file it must not touch"

    def test_app_still_opens_exactly_one_log_file_handle(self, tmp_path):
        """Control: the same probe without a worker's argv must still log to file.

        Without this, `test_worker_opens_no_handle_on_the_log_file` would also
        pass if logging were simply broken everywhere.
        """
        probe = _run_probe(tmp_path)

        assert probe["log_file_handlers"] == ["wiser.log"]
        assert probe["log_file_exists"]

    def test_numba_debug_logging_is_capped_in_the_app(self, tmp_path):
        """The other half of #502: what drives wiser.log to the rotation threshold.

        Numba emits a DEBUG record per bytecode instruction and per SSA block while
        it compiles. With the root logger at DEBUG and no cap on the numba logger,
        one spectral-feature-fitting run writes megabytes into wiser.log -- which is
        what forces the rollover that then fails. WISER's own loggers must keep
        their DEBUG output.
        """
        probe = _run_probe(tmp_path)

        assert not probe["numba_debug_enabled"], "numba's per-instruction DEBUG log is filling wiser.log"
        assert probe["wiser_debug_enabled"], "WISER's own DEBUG logging was lost"


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord("test", logging.INFO, __file__, 1, message, None, None)


def _hold_file_open(path: str, opened, release) -> None:
    """Child process: hold an OS handle on `path` until told to let go."""
    with open(path, "a", encoding="utf-8"):
        opened.set()
        release.wait(timeout=30)


@pytest.mark.multiprocessing
class TestRolloverWithForeignHandle:
    """Rotation while another *process* holds the log file open."""

    @pytest.fixture
    def logfile(self, tmp_path):
        return str(tmp_path / "wiser.log")

    @pytest.fixture
    def foreign_handle(self, logfile):
        """Run a real child process that keeps an OS handle on the log file."""
        open(logfile, "w", encoding="utf-8").close()

        opened = CTX.Event()
        release = CTX.Event()
        child = CTX.Process(target=_hold_file_open, args=(logfile, opened, release), daemon=True)
        child.start()
        try:
            assert opened.wait(timeout=30), "child never opened the log file"
            yield release
        finally:
            release.set()
            child.join(timeout=30)

    @staticmethod
    def _fill_past_max_bytes(handler: logging.handlers.RotatingFileHandler) -> None:
        handler.stream.write("x" * (handler.maxBytes + 1))
        handler.stream.flush()

    @WINDOWS_ONLY
    def test_stock_handler_raises_winerror_32(self, logfile, foreign_handle):
        """Characterization of the OS behaviour this whole module works around.

        Not a regression test -- it pins down stdlib/Windows, which our change does
        not alter. It documents that the rename in doRollover() is genuinely
        refused while another process holds the file, which is the mechanism behind
        the repeating "--- Logging error --- ... PermissionError" wall users saw.
        """
        handler = logging.handlers.RotatingFileHandler(logfile, maxBytes=100, backupCount=5)
        try:
            self._fill_past_max_bytes(handler)

            with pytest.raises(PermissionError) as exc_info:
                handler.doRollover()

            assert exc_info.value.winerror == 32
        finally:
            handler.close()

    def test_safe_handler_keeps_logging_through_a_blocked_rollover(self, logfile, foreign_handle):
        """A blocked rename must not raise, and must not lose records."""
        handler = SafeRotatingFileHandler(logfile, maxBytes=100, backupCount=5)
        try:
            self._fill_past_max_bytes(handler)

            handler.doRollover()  # PermissionError on Windows; absorbed

            assert handler.stream is not None, "handler was left with no stream to write to"
            handler.emit(_record("survived the blocked rollover"))
            handler.flush()
        finally:
            handler.close()

        assert "survived the blocked rollover" in open(logfile, encoding="utf-8").read()

    @WINDOWS_ONLY
    def test_safe_handler_does_not_retry_the_rollover_on_every_record(self, logfile, foreign_handle):
        """No retry storm.

        The file stays over maxBytes after a failed rollover, so a naive handler
        re-attempts (and re-fails) the rename for every single record -- which is
        why the console spam lasted minutes rather than printing once.
        """
        handler = SafeRotatingFileHandler(logfile, maxBytes=100, backupCount=5, retry_seconds=60.0)
        try:
            self._fill_past_max_bytes(handler)
            handler.doRollover()

            assert handler.shouldRollover(_record("next")) == 0, "handler is still trying to roll over"

            for i in range(50):
                handler.emit(_record(f"record {i}"))
            handler.flush()
        finally:
            handler.close()

        assert "record 49" in open(logfile, encoding="utf-8").read()

    def test_safe_handler_rotates_once_the_other_handle_is_released(self, logfile, foreign_handle):
        """Degradation is temporary: normal rotation resumes when the file frees up."""
        handler = SafeRotatingFileHandler(logfile, maxBytes=100, backupCount=5, retry_seconds=0.0)
        try:
            self._fill_past_max_bytes(handler)
            handler.doRollover()

            foreign_handle.set()  # child closes its handle and exits
            for _ in range(600):
                if os.path.exists(logfile + ".1"):
                    break
                handler.emit(_record("keep writing until rotation succeeds"))
                handler.flush()

            assert os.path.exists(logfile + ".1"), "rotation never recovered after the handle was released"
        finally:
            handler.close()
