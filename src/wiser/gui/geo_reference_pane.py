from typing import List, Union, Dict

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import wiser.gui.generated.resources

from wiser.raster.dataset import RasterDataSet

from .rasterview import ScaleToFitMode, RasterView
from .rasterpane import RasterPane
from .dataset_chooser import DatasetChooser
from .util import add_toolbar_action
from .app_state import ApplicationState

class GeoReferencerPane(RasterPane):
    # We don't want a roi chooser

    # We don't want a dataset chooser *

    # We do want zoom options *

    # We want the band chooser *

    def __init__(self, app_state, parent=None):
        super().__init__(app_state=app_state, parent=parent,
            max_zoom_scale=16, zoom_options=[0.25, 0.5, 0.75, 1, 2, 4, 8, 16],
            initial_zoom=1)
    
    def _init_dataset_tools(self):
        self._dataset_chooser = None

        self._act_band_chooser = None
    
    def _init_select_tools(self):
        '''
        We don't want this to initialize any of the select tools.
        The select tools currently are just the ROI tools
        '''
        return
