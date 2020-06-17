import argparse, faulthandler, importlib, json, os, sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import matplotlib

from gui.app import DataVisualizerApp


CONFIG_FILE = 'iswb.json'


def init_config():
    '''
    Initializes the configuration for the Imaging Spectroscopy Workbench.  This
    configuration is loaded from the default file 'iswb.json', if that file is
    found in the local working directory.
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


if __name__ == '__main__':
    faulthandler.enable()
    
    # TODO(donnie):  This is supposed to be how you turn on high-DPI
    #     application scaling in Qt.  This does not seem to be required on
    #     MacOSX though.  Saving this code in case it is needed on Linux or
    #     Windows 10.
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

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
