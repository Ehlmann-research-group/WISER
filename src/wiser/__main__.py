# ruff: noqa: E402
import argparse
import faulthandler
import importlib
import logging
import logging.config
import os
import sys
import traceback

import multiprocessing

# ============================================================================
# Cap BLAS/OpenMP threads BEFORE numpy loads (hence the position at module top).
# WISER parallelizes at the process level (scheduler ProcessPoolExecutor); if
# OpenBLAS/OpenMP also spawn one thread per core in *every* worker, the
# (workers x cores) threads oversubscribe and their buffers exhaust memory
# ("OpenBLAS: Memory allocation failed" / BrokenProcessPool). Give each worker a
# fair share instead -- cores / worker_budget -- where the budget mirrors
# SCHEDULER_PROCESS_BUDGET = min(6, cores) (can't import it; that pulls in numpy).
# setdefault() lets a shell/conda var override. Keep in sync with src/tests/conftest.py.
_cpu_count = os.cpu_count() or 1
_worker_budget = min(6, _cpu_count)  # mirrors SCHEDULER_PROCESS_BUDGET
_blas_threads = str(max(1, _cpu_count // _worker_budget))
for _blas_thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_blas_thread_var, _blas_threads)

# ============================================================================
# Load gdal plugins into path and set gdal environment variables
#
if getattr(sys, "frozen", False):
    # If PyInstaller has placed gdal_netCDF.dll, etc. into a "gdalplugins" folder
    # relative to sys._MEIPASS:
    plugin_path = os.path.join(sys._MEIPASS, "gdalplugins")
    os.environ["GDAL_DRIVER_PATH"] = plugin_path

# ============================================================================
# ESSENTIAL DEBUG CONFIGURATION
#

CONFIG_FILE = "wiser-conf.json"
LOG_CONF_FILE = "logging.conf"

# Do this as early as possible so we can catch crashes at load time.
# (Especially before loading Qt libraries.)
faulthandler.enable()

from wiser.gui.app_config import (
    get_wiser_config_dir,
    check_create_wiser_config_dir,
    resource_path,
    ApplicationConfig,
)

if getattr(sys, "frozen", False):
    # We are in a PyInstaller bundle
    app_data_dir = get_wiser_config_dir()
else:
    # Running as a script
    app_data_dir = os.path.dirname(os.path.abspath(__file__))
numba_cache_dir = os.path.join(app_data_dir, "Cache", "numba_jit")
os.makedirs(numba_cache_dir, exist_ok=True)

os.environ["NUMBA_CACHE_DIR"] = numba_cache_dir

# Try to create the WISER config directory if it doesn't exist.  If an error
# occurs, make sure to give the user a chance to find out.
try:
    check_create_wiser_config_dir()
except Exception:
    print(f"Couldn't create WISER config directory:  {get_wiser_config_dir()}")
    input("Press Enter to terminate the program.")
    sys.exit(1)


# Hard-code the logging configuration to remove the need for a log-config file.
# The if statement below ensures that only the main process and no subprocesses
# can get a file handle to the logger because this can cause a file rotation.
# The issue is noted in #502
if multiprocessing.parent_process() is None:  # This is the main process
    logfile_path = os.path.join(get_wiser_config_dir(), "wiser.log")
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
                    "class": "logging.handlers.RotatingFileHandler",
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
                "matplotlib": {
                    "level": "WARNING",
                    "handlers": ["consoleHandler", "fileHandler"],
                    "qualname": "matplotlib",
                },
            },
        }
    )
else:
    # Remove the "lastResort" handler, so only the null handler is left.
    root_logger = logging.getLogger()
    # Remove all handlers except NullHandler
    for handler in root_logger.handlers[:]:
        if not isinstance(handler, logging.NullHandler):
            root_logger.removeHandler(handler)
    # Ensure at least a NullHandler is present
    if not any(isinstance(h, logging.NullHandler) for h in root_logger.handlers):
        root_logger.addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)

# TODO(donnie):  Do this before importing matplotlib, to get rid of the
#     annoying warnings generated from the PyInstaller-frozen version.
#     See https://stackoverflow.com/a/60470942 for details.
import warnings

warnings.filterwarnings("ignore", "(?s).*MATPLOTLIBDATA.*", category=UserWarning)

# ============================================================================
# QT AND MATPLOTLIB IMPORTS
#
# If a failure occurs on these imports, it may be due to a missing library.
# This is why we do the debugging setup first.
#

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import matplotlib

# Use absolute paths
from wiser.gui import bug_reporting
from wiser.gui.startup_splash import StartupSplash, WiserGuiImportThread


# ============================================================================
# MAIN APPLICATION-LAUNCH CODE
#


def qt_debug_callback(*args, **kwargs):
    # TODO(donnie):  This is an experiment to see if we can get useful info
    #     out of Qt5 for WISER.  So far it has not panned out.  (2021-04-14)
    logger.debug(f"qt_debug_callback:  args={args}  kwargs={kwargs}")


def run_tests(tests: list[str]) -> int:
    # Pre-import cv2 while sys.path is still in the order established by the
    # pyi_rth_cv2 runtime hook (the bundled cv2/python-* native folder first).
    # This order is important because it ensures that the cv2 dll is found before
    # the cv2 python module so the recursion error doesn't happen.
    # However, pytest.main() below mutates sys.path during collection, which messes
    # up this order, causing the first cv2 import (which is cv2's python module)
    # to resolve to the package's __init__.py bootstrap (cv2's python module)
    # and raise "recursion is detected during loading of cv2 binary extensions"
    # in the frozen build. Loading it once here caches it in sys.modules so
    # later imports during collection don't cause any triggers after importing.
    # Normal startup already works because it imports cv2 at the start, so paths can't
    # be reordered.

    # TLDR: Prevent distributable --test_mode failure.
    import cv2  # noqa: F401

    import pytest

    os.environ["WISER_LOW_CONCURRENCY_SCHEDULER"] = "1"
    enabled_plugins = []  # e.g., ["-p", "pytester"] if you truly need a plugin
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # If we are in a frozen build _MEIPASS will be set so that's where our tests will be
    base = getattr(sys, "_MEIPASS", os.path.join(current_dir, ".."))
    test_folder_path = os.path.join(base, "tests")
    # If the user didn't pass in any tests, we just use the full testing directory
    if len(tests) == 0:
        tests = [test_folder_path]
    # In pytest the multiprocessing tests need to be run with -s . I don't know why.
    pytest_args = ["--disable-plugin-autoload", "-v", "-s"] + tests + enabled_plugins
    return pytest.main(pytest_args)


def main():
    """
    Main entry-point for the WISER application.
    """

    logger.info("=== STARTING WISER APPLICATION ===")

    # ========================================================================
    # Load Command-Line Arguments and Configuration File

    parser = argparse.ArgumentParser(
        description="Workbench for Imaging Spectroscopy Exploration and Research"
    )

    parser.add_argument(
        "data_files",
        metavar="file",
        nargs="*",
        help="An optional list of data files to open in WISER",
    )
    parser.add_argument("--config_path", help="The path to read WISER configuration from")
    parser.add_argument(
        "--test_mode",
        nargs="*",
        default=None,
        help=(
            "Whether to run WISER tests \
        before startup. If included, you may specify what files to run tests on. If no files \
        are specified, tests will be run on the full test directory."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Launch check: start Qt and show the splash, then exit 0 without opening the app.",
    )
    # TODO(donnie):  Provide a way to specify Qt arguments

    args = parser.parse_args()

    if args.smoke and args.test_mode is not None:
        parser.error("--smoke and --test_mode cannot be combined.")

    if (
        args.smoke
        and sys.platform.startswith("linux")
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    ):
        # Headless Linux has no window server, so fall back to the offscreen Qt
        # platform. Windows/macOS (including the CI runners) have a display and use
        # their native platform, so the smoke check does not hard-depend on the
        # bundled offscreen plugin.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Load the WISER configuration file.  If the user specifies a config file,
    # load it and exit if we can't load it.  Otherwise, if we see a WISER-config
    # file in the local directory, load it; if we don't see one, don't try to
    # load anything.

    config_path: str = get_wiser_config_dir()
    if args.config_path is not None:
        # User specified config path; try to load it.
        config_path = args.config_path

    logger.info(f"Loading WISER configuration from path {config_path}")

    config: ApplicationConfig = ApplicationConfig()
    wiser_conf_path: str = os.path.join(config_path, "wiser-conf.json")
    loaded_config: bool = False

    if os.path.isfile(wiser_conf_path):
        logger.info(f"Trying to load wiser-conf.json from path {wiser_conf_path}")

        try:
            config.load(wiser_conf_path)
            loaded_config = True

        except:
            # Couldn't load the file.  Try to move it out of the way and create
            # a new one.

            logger.exception("Couldn't load WISER config file, is it corrupt?")
            logger.info("Renaming bad WISER config file and creating new config.")
            error_conf_path = os.path.join(config_path, "wiser-conf.json.error")
            os.replace(wiser_conf_path, error_conf_path)

    if not loaded_config:
        # The config file couldn't be loaded, either because it doesn't
        # exist, or because it was corrupt.  Try to create a new default
        # config file.  Failure is not fatal; WISER will still use the
        # default configuration.
        logger.info(f"Creating a new WISER config file at {config_path}")
        try:
            # Try to save the WISER configuration file
            config.save(wiser_conf_path)
        except OSError:
            logger.exception(f"Couldn't create new WISER config file at {config_path}")

    logger.debug(f"WISER configuration:\n{config.to_string()}")

    # ========================================================================
    # Configure BugSnag

    bug_reporting.initialize(config)

    # ========================================================================
    # Qt Platform Initialization

    qInstallMessageHandler(qt_debug_callback)

    tests = args.test_mode
    if tests is not None:
        code = run_tests(tests)
        sys.exit(code)
    else:
        os.environ["WISER_LOW_CONCURRENCY_SCHEDULER"] = "0"
        # TODO(donnie):  Pass Qt arguments
        app = QApplication([])
        icon_path = resource_path("icons", "wiser.iconset", "icon_256x256.png")
        icon = QIcon(icon_path)
        app.setWindowIcon(icon)
        pixmap = icon.pixmap(256, 256)
        splash = StartupSplash(pixmap, icon)
        splash.show()
        app.processEvents()

        if args.smoke:
            logger.info("Smoke check passed: Qt started and splash shown; exiting 0.")
            sys.exit(0)

        data_files: list[str] = list(args.data_files) if args.data_files else []

        import_thread = WiserGuiImportThread(app)

        def on_import_succeeded(app_module) -> None:
            if app.closingDown() or splash.is_cancelled():
                return
            try:
                wiser_ui = app_module.DataVisualizerApp(config_path=config_path, config=config)
                screen_size = app.screens()[0].size()
                wiser_ui.resize(screen_size * 0.7)
                splash.finish_and_show_main(wiser_ui)
                for file_path in data_files:
                    wiser_ui._app_state.open_file(file_path)
            except Exception:
                logger.exception("Failed to construct WISER main window")
                splash.set_startup_failed(traceback.format_exc())

        def on_import_failed(message: str) -> None:
            if app.closingDown() or splash.is_cancelled():
                return
            splash.set_startup_failed(message)

        import_thread.succeeded.connect(on_import_succeeded)
        import_thread.failed.connect(on_import_failed)
        # Import runs on a worker thread so the splash stays responsive (e.g. cancel via ×).
        QTimer.singleShot(0, import_thread.start)

        # ========================================================================
        # WISER Application Initialization

        # If the WISER config file was created for the first time, ask the user if
        # they would like to opt in to online bug reporting.
        if not loaded_config:
            logger.debug("WISER config not loaded.  Asking user to opt-in for " + "online bug reporting.")
            dialog = bug_reporting.BugReportingDialog()
            dialog.exec()

            auto_notify = dialog.user_wants_bug_reporting()
            config.set("general.online_bug_reporting", auto_notify)
            bug_reporting.set_enabled(auto_notify)

            try:
                # Try to save the WISER configuration file
                logger.debug(f'Saving initial WISER config:  "{wiser_conf_path}"')
                config.save(wiser_conf_path)
            except OSError:
                logger.exception(f"Couldn't save WISER config file at {config_path}")

        # Command-line data files are opened in on_import_succeeded after the main window exists.

        sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Spawn generally does not work on linux and on linux, the above command
        # is not able to set the start method to spawn. However, we do use a
        # start context with spawn, so this shouldn't really mater
        logger.debug(f"Start method not set to spawn. Defaulting to {multiprocessing.get_start_method()}")
        pass
    main()
