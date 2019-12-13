from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .rasterpane import RasterPane
from .util import add_toolbar_action

from raster.dataset import find_truecolor_bands


class ZoomPane(RasterPane):
    '''
    This widget provides a dockable zoom pane in the user interface.  The zoom
    pane always shows the raster image zoomed in at a very large magnification.
    The specific amount of magnification can be set by the user.
    '''

    def __init__(self, app_state, parent=None):
        super().__init__(app_state=app_state, parent=parent,
            size_hint=QSize(400, 400),
            min_zoom_scale=1, max_zoom_scale=16, zoom_options=range(1, 17),
            initial_zoom=8)

        # Register for events from the application state

        self._rasterview.viewport_change.connect(self._on_raster_viewport_changed)
        self._rasterview.mouse_click.connect(self._on_raster_mouse_clicked)


    def _zoom_in_scale(self, scale):
        '''
        Zoom in the display by 1x more than the previous scale.
        '''
        return scale + 1


    def _zoom_out_scale(self, scale):
        '''
        Zoom out the display by 1x less than the previous scale.
        '''
        return scale - 1


    def _on_raster_viewport_changed(self, visible_area):
        self._app_state.set_view_attribute('zoom.visible_area', visible_area)


    def _on_raster_mouse_clicked(self, point, event):
        self._app_state.set_view_attribute('zoom.current_pixel', point)


    def _on_view_attr_changed(self, attr_name):
        if attr_name == 'image.current_pixel':
            pixel = self._app_state.get_view_attribute('image.current_pixel')
            self._rasterview.make_point_visible(pixel.x(), pixel.y())

            self._rasterview.update()


    def _afterRasterPaint(self, widget, paint_event):
        current_pixel = self._app_state.get_view_attribute('image.current_pixel')
        if current_pixel is None:
            return

        # Draw the visible area on the summary view.
        painter = QPainter(widget)
        painter.setPen(QPen(Qt.red))

        x = current_pixel.x()
        y = current_pixel.y()

        # Draw a box around the currently selected pixel
        scaled = QRect(x * self._scale_factor, y * self._scale_factor,
                       self._scale_factor, self._scale_factor)

        painter.drawRect(scaled)

        painter.end()
