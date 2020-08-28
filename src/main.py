import argparse, faulthandler, importlib, json, os, sys
# Do this as early as possible so we can catch crashes at load time.
faulthandler.enable()

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


CONFIG_FILE = 'wiserconf.json'


def init_config():
    '''
    Initializes the configuration for the Workbench.  This configuration is
    loaded from the default file 'wiserconf.json', if that file is found in the
    local working directory.
    '''
    config = {}

    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            config = json.load(f)

    return config

"""
def init_loader(config):
    '''
    This function initializes the raster dataset loader to use within the
    workbench application.
    '''
    fq_name = config.get('loader.class', 'raster.gdal_dataset.GDALRasterDataLoader')
    parts = fq_name.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]

    module = importlib.import_module(module_name)
    class_type = getattr(module, class_name)
    loader = class_type(config=config)

    return loader
"""

def init_matplotlib():
    matplotlib.rcParams['font.size'] = 9


def main():
    # Configure BugSnag
    bugsnag.configure(
        api_key='29bf39226c3071461f3d0630c9ced4b6',
        app_version=version.VERSION,
        auto_notify=False,
    )

    # Turn on high-DPI application scaling in Qt.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)

    # config = init_config()
    # loader = init_loader(config)
    init_matplotlib()

    ui = DataVisualizerApp()

    # Set the initial window size to be 70% of the screen size.
    screen_size = app.screens()[0].size()
    ui.resize(screen_size * 0.7)
    ui.show()

    # If any data files are specified on the command-line, open them now
    for file_path in sys.argv[1:]:
        ui._app_state.open_file(file_path)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
