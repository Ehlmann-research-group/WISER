from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .rasterpane import RasterPane



class ZoomPane(RasterPane):
    """
    This widget provides a dockable zoom pane in the user interface.  The zoom
    pane always shows the raster image zoomed in at a very large magnification.
    The specific amount of magnification can be set by the user.
    """

    def __init__(self, app_state, parent=None):
        super().__init__(
            app_state=app_state,
            parent=parent,
            size_hint=QSize(400, 400),
            min_zoom_scale=1,
            max_zoom_scale=16,
            zoom_options=range(1, 17),
            initial_zoom=8,
        )

    def _zoom_in_scale(self, scale):
        """
        Zoom in the display by 1x more than the previous scale.
        """
        return scale + 1

    def _zoom_out_scale(self, scale):
        """
        Zoom out the display by 1x less than the previous scale.
        """
        return scale - 1
