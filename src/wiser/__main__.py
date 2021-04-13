import argparse
import faulthandler
import importlib
import logging
import logging.config
import os
import sys

CONFIG_FILE = 'wiser-conf.json'
LOG_CONF_FILE = 'logging.conf'

# Do this as early as possible so we can catch crashes at load time.
# (Yes, even before loading Qt libraries.)
faulthandler.enable()

if os.path.isfile(LOG_CONF_FILE):
    logging.config.fileConfig(LOG_CONF_FILE)

logger = logging.getLogger(__name__)

from typing import Dict

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

# TODO(donnie):  Do this before importing matplotlib, to get rid of the
#     annoying warnings generated from the PyInstaller-frozen version.
#     See https://stackoverflow.com/a/60470942 for details.
import warnings
warnings.filterwarnings('ignore', '(?s).*MATPLOTLIBDATA.*', category=UserWarning)

import matplotlib

from .gui.app import DataVisualizerApp
from .gui.app_config import (get_wiser_config_dir, ApplicationConfig)
from .gui import bug_reporting


def main():
    '''
    Main entry-point for the WISER application.
    '''

    logger.info('=== STARTING WISER APPLICATION ===')

    #========================================================================
    # Load Command-Line Arguments and Configuration File

    parser = argparse.ArgumentParser(
        description='Workbench for Imaging Spectroscopy Exploration and Research')

    parser.add_argument('data_files', metavar='file', nargs='*',
        help='An optional list of data files to open in WISER')
    parser.add_argument('--config_path',
        help='The path to read WISER configuration from')
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

    logger.info(f'Loading WISER configuration from path {config_path}')

    config: ApplicationConfig = ApplicationConfig()
    wiser_conf_path: str = os.path.join(config_path, 'wiser-conf.json')
    loaded_config: bool = False

    if os.path.isfile(wiser_conf_path):
        logger.info(f'Trying to load wiser-conf.json from path {wiser_conf_path}')

        try:
            config.load(wiser_conf_path)
            loaded_config = True

        except:
            # Couldn't load the file.  Try to move it out of the way and create
            # a new one.

            logger.exception('Couldn\'t load WISER config file, is it corrupt?')
            logger.info('Renaming bad WISER config file and creating new config.')
            error_conf_path = os.path.join(config_path, 'wiser-conf.json.error')
            os.replace(wiser_conf_path, error_conf_path)

    if not loaded_config:
        # The config file couldn't be loaded, either because it doesn't
        # exist, or because it was corrupt.  Try to create a new default
        # config file.  Failure is not fatal; WISER will still use the
        # default configuration.
        logger.info(f'Creating a new WISER config file at {config_path}')
        try:
            # Try to save the WISER configuration file
            config.save(wiser_conf_path)
        except OSError:
            logger.exception(f'Couldn\'t create new WISER config file at {config_path}')

    logger.debug(f'WISER configuration:\n{config.to_string()}')

    #========================================================================
    # Configure BugSnag

    bug_reporting.initialize(config)

    #========================================================================
    # Qt Platform Initialization

    # Turn on high-DPI application scaling in Qt.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # TODO(donnie):  Pass Qt arguments
    app = QApplication([])

    #========================================================================
    # WISER Application Initialization

    wiser_ui = DataVisualizerApp(config_path=config_path, config=config)

    # Set the initial window size to be 70% of the screen size.
    screen_size = app.screens()[0].size()
    wiser_ui.resize(screen_size * 0.7)
    wiser_ui.show()

    # If the WISER config file was created for the first time, ask the user if
    # they would like to opt in to online bug reporting.
    if not loaded_config:
        dialog = bug_reporting.BugReportingDialog()
        dialog.exec()

        auto_notify = dialog.user_wants_bug_reporting()
        config.set('general.online_bug_reporting', auto_notify)
        bug_reporting.set_enabled(auto_notify)

        try:
            # Try to save the WISER configuration file
            config.save(wiser_conf_path)
        except OSError:
            logger.exception(f'Couldn\'t save WISER config file at {config_path}')

    # If any data files are specified on the command-line, open them now
    for file_path in sys.argv[1:]:
        wiser_ui._app_state.open_file(file_path)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
