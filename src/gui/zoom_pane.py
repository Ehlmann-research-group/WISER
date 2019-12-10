from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .band_chooser import BandChooserDialog
from .dataset_chooser import DatasetChooser
from .rasterview import RasterView
from .util import add_toolbar_action

from raster.dataset import find_truecolor_bands


class DetailRasterView(RasterView):
    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state


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


class ZoomPane(QDockWidget):
    '''
    This widget provides a dockable zoom pane in the user interface.  The zoom
    pane always shows the raster image zoomed in at a very large magnification.
    The specific amount of magnification can be set by the user.
    '''

    def __init__(self, app_state, parent=None):
        super().__init__('Zoom', parent=parent)

        # Initialize widget's internal state

        self._app_state = app_state
        self._dataset_index = None

        self._min_zoom_scale = 1
        self._max_zoom_scale = 16
        self._zoom_options = range(1, 17)

        # Initialize contents of the widget

        self._init_ui()

        # Initialize docking details

        # TODO(donnie):  Overview pane always stays on top!  See if we can
        #     make it go underneath other windows.
        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)

        self.visibilityChanged.connect(self._on_visibility_changed)

        # Register for events from the application state

        self._app_state.dataset_added.connect(self._on_dataset_added)
        self._app_state.dataset_removed.connect(self._on_dataset_removed)
        self._app_state.view_attr_changed.connect(self._on_view_attr_changed)

        self._rasterview.viewport_change.connect(self._on_raster_viewport_changed)
        self._rasterview.mouse_click.connect(self._on_raster_mouse_clicked)


    def _init_ui(self):
        ''' Initialize the contents of this widget '''

        toolbar = QToolBar(self.tr('Toolbar'), parent=self)
        toolbar.setIconSize(QSize(20, 20))

        # Dataset / Band Tools

        self._dataset_chooser = DatasetChooser(self._app_state)
        toolbar.addWidget(self._dataset_chooser)
        self._dataset_chooser.triggered.connect(self._on_dataset_changed)

        self._act_band_chooser = add_toolbar_action(toolbar,
            'resources/choose-bands.svg', self.tr('Band chooser'), self)
        # TODO(donnie):  If we just pop up a widget...
        # self._band_chooser = BandChooser(self._app_state)
        # toolbar.addWidget(self._band_chooser)
        self._act_band_chooser.triggered.connect(self._on_band_chooser)

        self._act_choose_true_colors = add_toolbar_action(toolbar,
            'resources/choose-truecolor.svg', self.tr('Choose true colors'), self)
        self._act_choose_true_colors.triggered.connect(self._on_choose_true_colors)

        toolbar.addSeparator()

        # Zoom In / Out Tools

        # Zoom In
        self._act_zoom_in = add_toolbar_action(toolbar,
            'resources/zoom-in.svg', self.tr('Zoom in'), self, QKeySequence.ZoomIn)
        self._act_zoom_in.triggered.connect(self._on_zoom_in)

        # Zoom Out
        self._act_zoom_out = add_toolbar_action(toolbar,
            'resources/zoom-out.svg', self.tr('Zoom out'), self, QKeySequence.ZoomOut)
        self._act_zoom_out.triggered.connect(self._on_zoom_out)

        # Zoom Level
        self._cbox_zoom = QComboBox()
        for v in self._zoom_options:
            self._cbox_zoom.addItem(f'{int(v * 100)}%', v)
        self._cbox_zoom.setEditable(False) # TODO(donnie):  may want to support editing
        self._cbox_zoom.currentIndexChanged.connect(self._on_zoom_cbox)
        toolbar.addWidget(self._cbox_zoom)

        toolbar.addSeparator()

        # Raster image view widget

        self._rasterview = DetailRasterView(self._app_state, parent=self)

        # Widget layout

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        layout.setMenuBar(toolbar)
        layout.addWidget(self._rasterview)

        widget = QWidget(parent=self)
        widget.setLayout(layout)

        self.setWidget(widget)


    def toggleViewAction(self):
        '''
        Returns a QAction object that can be used to toggle the visibility of
        this dockable pane.  This class overrides the QDockWidget implementation
        to specify a nice icon and tooltip on the action.
        '''
        act = super().toggleViewAction()
        act.setIcon(QIcon('resources/zoom-pane.svg'))
        act.setToolTip(self.tr('Show/hide zoom pane'))
        return act


    def sizeHint(self):
        ''' The default size of the zoom-pane widget is 400x400. '''
        return QSize(400, 400)


    def resizeEvent(self, event):
        ''' Update the raster-view image when this widget is resized. '''
        # TODO(donnie):  Why was this necessary in the zoom pane?  In other
        #     panes we may need to update scaling details.
        # self._update_image()


    def _on_visibility_changed(self, visible):
        self._app_state.set_view_attribute('zoom.visible', visible)

        # Work around a known Qt bug:  if a dockable window is floating, and is
        # closed while floating, it can't be redocked unless we toggle its
        # floating state.
        if self.isFloating() and not visible:
            self.setFloating(False)
            self.setFloating(True)


    def _on_dataset_added(self, index):
        if self._app_state.num_datasets() == 1:
            # We finally have a dataset!
            self._dataset_index = 0
            self._update_image()

            self._rasterview.scale_image(8)
            self._update_zoom_widgets()



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


    def _on_band_chooser(self, act):
        dataset = self._app_state.get_dataset(self._dataset_index)
        display_bands = self._rasterview.get_display_bands()

        dialog = BandChooserDialog(dataset, display_bands, parent=self)
        dialog.setModal(True)

        if dialog.exec_() == QDialog.Accepted:
            # TODO(donnie) - set display bands!
            display_bands = dialog.get_display_bands()
            self._rasterview.set_display_bands(display_bands)

    def _on_choose_true_colors(self, act):
        dataset = self._app_state.get_dataset(self._dataset_index)
        display_bands = find_truecolor_bands(dataset)

        if display_bands is not None:
            self._rasterview.set_display_bands(display_bands)
        else:
            QMessageBox.warning(self, self.tr('No Visible Color Bands'),
                self.tr('This raster data does not contain bands corresponding '
                        'to visible color wavelengths.'))


    def _on_zoom_in(self, evt):
        ''' Zoom in the zoom-view by one level. '''

        scale = self._rasterview.get_scale()
        new_scale = scale + 1
        if new_scale <= self._max_zoom_scale:
            self._rasterview.scale_image(new_scale)

        self._update_zoom_widgets()

    def _on_zoom_out(self, evt):
        ''' Zoom out the zoom-view by one level. '''

        scale = self._rasterview.get_scale()
        new_scale = scale - 1
        if new_scale >= self._min_zoom_scale:
            self._rasterview.scale_image(new_scale)

        self._update_zoom_widgets()


    def _on_zoom_cbox(self, index):
        ''' Zoom the zoom-view to the specified option in the zoom combo-box. '''
        self._rasterview.scale_image(self._cbox_zoom.currentData())
        self._update_zoom_widgets()


    def _update_zoom_widgets(self):
        scale = self._rasterview.get_scale()

        # Enable / disable zoom buttons based on scale
        self._act_zoom_out.setEnabled(scale >= self._min_zoom_scale)
        self._act_zoom_in.setEnabled(scale <= self._max_zoom_scale)

        # Set the zoom-level value
        if scale in self._zoom_options:
            scale_idx = self._zoom_options.index(scale)
            self._cbox_zoom.setCurrentIndex(scale_idx)
        else:
            scale_str = f'{int(scale * 100)}%'
            self._cbox_zoom.setCurrentText(scale_str)


    def _on_raster_viewport_changed(self, visible_area):
        self._app_state.set_view_attribute('zoom.visible_area', visible_area)


    def _on_raster_mouse_clicked(self, point, event):
        self._app_state.set_view_attribute('zoom.current_pixel', point)


    def _on_view_attr_changed(self, attr_name):
        if attr_name == 'image.current_pixel':
            pixel = self._app_state.get_view_attribute('image.current_pixel')
            self._rasterview.make_point_visible(pixel.x(), pixel.y())

            self._rasterview.update()


    def _update_image(self):
        dataset = None
        if self._app_state.num_datasets() > 0:
            dataset = self._app_state.get_dataset(self._dataset_index)

        # TODO(donnie):  Only do this when the raster dataset actually changes,
        #     or the displayed bands change, etc.
        self._rasterview.set_raster_data(dataset)

        if dataset is None:
            return
