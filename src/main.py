import argparse
import faulthandler
import importlib
import json
import logging
import logging.config
import os
import sys

# Do this as early as possible so we can catch crashes at load time.
# (Yes, even before loading Qt libraries.)
faulthandler.enable()

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

from typing import Dict

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import bugsnag

# TODO(donnie):  Do this before importing matplotlib, to get rid of the
#     annoying warnings generated from the PyInstaller-frozen version.
#     See https://stackoverflow.com/a/60470942 for details.
import warnings
warnings.filterwarnings('ignore', '(?s).*MATPLOTLIBDATA.*', category=UserWarning)

import matplotlib

from gui.app import DataVisualizerApp

import version


CONFIG_FILE = 'wiser-conf.json'


def init_matplotlib():
    matplotlib.rcParams['font.size'] = 9


def load_wiser_conf(filename: str) -> Dict:
    '''
    Load the specified WISER configuration file.  If the file cannot be loaded
    for some reason, this function will log the error and then terminate the
    application.
    '''
    logger.info(f'Loading WISER configuration file "{filename}"')
    try:
        with open(filename) as f:
            config = json.load(f)
    except:
        logger.exception(f'Couldn\'t open WISER configuration file')
        sys.exit(1)

    return config


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
    parser.add_argument('--config',
        help='The path and filename of an optional WISER configuration file')
    # TODO(donnie):  Provide a way to specify Qt arguments

    args = parser.parse_args()

    # Load the WISER configuration file.  If the user specifies a config file,
    # load it and exit if we can't load it.  Otherwise, if we see a WISER-config
    # file in the local directory, load it; if we don't see one, don't try to
    # load anything.
    config = {}
    if args.config is not None:
        config = load_wiser_conf(args.config)
    elif os.path.isfile(CONFIG_FILE):
        config = load_wiser_conf(CONFIG_FILE)

    logger.debug(f'Loaded WISER configuration:\n{json.dumps(config, sort_keys=True, indent=4)}')

    #========================================================================
    # Configure BugSnag

    # TODO(donnie):  Later want to default online bug reporting to False.
    auto_notify = bool(config.get('online-bug-reporting', True))
    bugsnag.configure(
        api_key='29bf39226c3071461f3d0630c9ced4b6',
        app_version=version.VERSION,
        auto_notify=auto_notify,
    )

    #========================================================================
    # Qt Platform Initialization

    # Turn on high-DPI application scaling in Qt.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # TODO(donnie):  Pass Qt arguments
    app = QApplication([])

    #========================================================================
    # WISER Application Initialization

    # TODO(donnie):  Remove?
    # init_matplotlib()

    wiser_ui = DataVisualizerApp(config=config)

    # Set the initial window size to be 70% of the screen size.
    screen_size = app.screens()[0].size()
    wiser_ui.resize(screen_size * 0.7)
    wiser_ui.show()

    # If any data files are specified on the command-line, open them now
    for file_path in sys.argv[1:]:
        wiser_ui._app_state.open_file(file_path)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
