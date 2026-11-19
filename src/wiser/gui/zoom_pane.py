import math
from typing import Callable

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .rasterpane import RasterPane
from .util import add_toolbar_action

from wiser.raster.dataset import find_truecolor_bands


class ZoomPane(RasterPane):
    """
    This widget provides a dockable zoom pane in the user interface.  The zoom
    pane always shows the raster image zoomed in at a very large magnification.
    The specific amount of magnification can be set by the user.

    The first time the pane is visible while showing a dataset, its zoom is set
    to INITIAL_ZOOM_FACTOR times the scale returned by ``main_view_scale``,
    rounded up to a whole zoom level and clamped to the pane's 1x-16x range.
    Zoom levels set after that are kept.

    ``main_view_linked`` returns whether the main view's views are linked.
    While they are, a pixel selected in a linked dataset is shown and centered
    in the dataset this pane shows.
    """

    INITIAL_ZOOM_FACTOR = 4

    def __init__(
        self,
        app_state,
        main_view_scale: Callable[[], float],
        main_view_linked: Callable[[], bool],
        parent=None,
    ):
        super().__init__(
            app_state=app_state,
            parent=parent,
            size_hint=QSize(400, 400),
            min_zoom_scale=1,
            max_zoom_scale=16,
            zoom_options=range(1, 17),
        )
        self._main_view_scale = main_view_scale
        self._main_view_linked = main_view_linked
        self._initial_scale_set = False

    def _views_linked(self) -> bool:
        return self._main_view_linked()

    def show_dataset(self, dataset, rasterview_pos=(0, 0)):
        super().show_dataset(dataset, rasterview_pos)
        self._set_initial_scale()

    def showEvent(self, event):
        # Blocked so the rescale does not report a viewport change that would
        # scroll the Main View; the visibility_change signal emitted by the
        # base class reports the rescaled viewport.
        with QSignalBlocker(self):
            self._set_initial_scale()
        super().showEvent(event)

    def _set_initial_scale(self):
        if self._initial_scale_set or not self.isVisible() or self.get_current_dataset() is None:
            return

        self._initial_scale_set = True
        scale = math.ceil(self.INITIAL_ZOOM_FACTOR * self._main_view_scale())
        self.set_scale(min(max(scale, self._min_zoom_scale), self._max_zoom_scale))
        self._update_zoom_widgets()

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
