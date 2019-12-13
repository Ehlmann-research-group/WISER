import os

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .dataset_chooser import DatasetChooser
from .rasterpane import RasterPane
from .util import add_toolbar_action


class MainViewWidget(RasterPane):
    '''
    This widget provides the main raster-data view in the user interface.
    '''

    # Signal:  the displayed region has changed
    display_area_changed = Signal( (int, int, int, int) )


    def __init__(self, app_state, parent=None):
        super().__init__(app_state=app_state, parent=parent, embed_toolbar=False,
            max_zoom_scale=16, zoom_options=[0.25, 0.5, 0.75, 1, 2, 4, 8, 16],
            initial_zoom=1)

        self._app_state.view_attr_changed.connect(self._on_view_attr_changed)

        # Raster image view widget

        # self._rasterview.viewport_change.connect(self._on_raster_viewport_changed)
        # self._rasterview.mouse_click.connect(self._on_raster_mouse_clicked)


    def _init_zoom_tools(self):
        '''
        Initialize zoom toolbar buttons.  This subclass adds a few extra zoom
        buttons to the zoom tools.
        '''
        super()._init_zoom_tools()

        # Zoom to Actual Size
        self._act_zoom_to_actual = add_toolbar_action(self._toolbar,
            'resources/zoom-to-actual.svg', self.tr('Zoom to actual size'), self,
            before=self._act_cbox_zoom)
        self._act_zoom_to_actual.triggered.connect(self._on_zoom_to_actual)

        # Zoom to Fit
        self._act_zoom_to_fit = add_toolbar_action(self._toolbar,
            'resources/zoom-to-fit.svg', self.tr('Zoom to fit'), self,
            before=self._act_cbox_zoom)
        self._act_zoom_to_fit.triggered.connect(self._on_zoom_to_fit)


    def _on_dataset_added(self, index):
        if self._app_state.num_datasets() == 1:
            # We finally have a dataset!
            self._dataset_index = 0
            self._update_image()


    def _on_dataset_removed(self, index):
        num = self._app_state.num_datasets()
        if num == 0 or self._dataset_index == index:
            self._dataset_index = min(self._dataset_index, num - 1)
            if self._dataset_index == -1:
                self._dataset_index = None

            self._update_image()


    def _on_dataset_changed(self, act):
        self._dataset_index = act.data()
        self._update_image()


    def _on_raster_viewport_changed(self, visible_area):
        self._app_state.set_view_attribute('image.visible_area', visible_area)


    def _on_raster_mouse_clicked(self, point, event):
        self._app_state.set_view_attribute('image.current_pixel', point)


    def _on_view_attr_changed(self, attr_name):
        # print('Main:  view attr changed:  ' + attr_name)
        if attr_name in ['zoom.visible_area', 'zoom.visible']:
            self._rasterview.update()


    def _on_zoom_to_actual(self, evt):
        ''' Zoom the view to 100% scale. '''

        self._rasterview.scale_image(1.0)
        self._update_zoom_widgets()


    def _on_zoom_to_fit(self):
        ''' Zoom the view such that the entire image fits in the view. '''
        self._rasterview.scale_image_to_fit()
        self._update_zoom_widgets()


    def _afterRasterPaint(self, widget, paint_event):
        zoom_visible = self._app_state.get_view_attribute('zoom.visible')
        if not zoom_visible:
            return

        visible_area = self._app_state.get_view_attribute('zoom.visible_area')
        if visible_area is None:
            return

        # Draw the visible area on the summary view.
        painter = QPainter(widget)
        painter.setPen(QPen(Qt.green))

        scaled = QRect(visible_area.x() * self._scale_factor,
                       visible_area.y() * self._scale_factor,
                       visible_area.width() * self._scale_factor,
                       visible_area.height() * self._scale_factor)

        painter.drawRect(scaled)

        painter.end()
