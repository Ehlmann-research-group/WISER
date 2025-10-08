# ruff: noqa: E402
import argparse
import faulthandler
import importlib
import logging
import logging.config
import os
import sys

import multiprocessing

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
    ApplicationConfig,
)

# Try to create the WISER config directory if it doesn't exist.  If an error
# occurs, make sure to give the user a chance to find out.
try:
    check_create_wiser_config_dir()
except Exception:
    print(f"Couldn't create WISER config directory:  {get_wiser_config_dir()}")
    input("Press Enter to terminate the program.")
    sys.exit(1)


# Hard-code the logging configuration to remove the need for a log-config file.
# TODO(donnie):  This is probably a BAD idea and needs to be revised in the future.
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
                "maxBytes": 10000000,
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

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import matplotlib

# Use absolute imports in __main__.py so that we can pass the file to
# pyinstaller.
from wiser.gui.app import DataVisualizerApp
from wiser.gui import bug_reporting


# ============================================================================
# MAIN APPLICATION-LAUNCH CODE
#


def qt_debug_callback(*args, **kwargs):
    # TODO(donnie):  This is an experiment to see if we can get useful info
    #     out of Qt5 for WISER.  So far it has not panned out.  (2021-04-14)
    logger.debug(f"qt_debug_callback:  args={args}  kwargs={kwargs}")


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
    # TODO(donnie):  Provide a way to specify Qt arguments

    args = parser.parse_args()

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

    # Turn on high-DPI application scaling in Qt.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # TODO(donnie):  Pass Qt arguments
    app = QApplication([])

    # ========================================================================
    # WISER Application Initialization

    wiser_ui = DataVisualizerApp(config_path=config_path, config=config)

    # Set the initial window size to be 70% of the screen size.
    screen_size = app.screens()[0].size()
    wiser_ui.resize(screen_size * 0.7)
    wiser_ui.show()

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

    # If any data files are specified on the command-line, open them now
    for file_path in sys.argv[1:]:
        logger.info(f'Opening file "{file_path}" specified on command-line')
        wiser_ui._app_state.open_file(file_path)

    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Context already set (e.g., Windows default 'spawn'); safe to ignore
        assert multiprocessing.get_start_method() == "spawn"
    multiprocessing.freeze_support()
    main()
