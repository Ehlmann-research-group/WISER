from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .band_chooser import BandChooserDialog
from .dataset_chooser import DatasetChooser
from .rasterview import RasterView
from .util import add_toolbar_action

from raster.dataset import find_truecolor_bands


class RasterPane(QWidget):
    '''
    This widget provides a raster-view with an associated toolbar.
    '''

    def __init__(self, app_state, parent=None, size_hint=None,
                 embed_toolbar=True,
                 min_zoom_scale=None, max_zoom_scale=None, zoom_options=None,
                 initial_zoom=None):
        super().__init__(parent=parent)

        # Initialize widget's internal state

        self._app_state = app_state
        self._dataset_index = None

        self._size_hint = size_hint
        self._embed_toolbar = embed_toolbar

        self._min_zoom_scale = min_zoom_scale
        self._max_zoom_scale = max_zoom_scale
        self._zoom_options = zoom_options
        self._initial_zoom = initial_zoom

        # Initialize contents of the widget

        self._init_ui()

        # Register for events from the application state

        self._app_state.dataset_added.connect(self._on_dataset_added)
        self._app_state.dataset_removed.connect(self._on_dataset_removed)
        self._app_state.view_attr_changed.connect(self._on_view_attr_changed)

        # self._rasterview.viewport_change.connect(self._on_raster_viewport_changed)
        # self._rasterview.mouse_click.connect(self._on_raster_mouse_clicked)


    def _init_ui(self):
        ''' Initialize the contents of this widget '''

        self._toolbar = QToolBar(self.tr('Toolbar'), parent=self)

        if self._embed_toolbar:
            self._toolbar.setIconSize(QSize(20, 20))

        self._init_dataset_band_tools()
        self._toolbar.addSeparator()
        self._init_zoom_tools()

        # Raster image view widget

        self._rasterview = RasterView(parent=self)
        self._rasterview.set_after_raster_paint(self._after_raster_paint)

        # Widget layout

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        if self._embed_toolbar:
            layout.setMenuBar(self._toolbar)

        layout.addWidget(self._rasterview)

        self.setLayout(layout)

        self._update_zoom_widgets()


    def _init_dataset_band_tools(self):
        '''
        Initialize dataset-selection and band-selection toolbar buttons.
        '''

        # Dataset / Band Tools

        self._dataset_chooser = DatasetChooser(self._app_state)
        self._toolbar.addWidget(self._dataset_chooser)
        self._dataset_chooser.triggered.connect(self._on_dataset_changed)

        self._act_band_chooser = add_toolbar_action(self._toolbar,
            'resources/choose-bands.svg', self.tr('Band chooser'), self)
        # TODO(donnie):  If we just pop up a widget...
        # self._band_chooser = BandChooser(self._app_state)
        # toolbar.addWidget(self._band_chooser)
        self._act_band_chooser.triggered.connect(self._on_band_chooser)

        self._act_choose_true_colors = add_toolbar_action(self._toolbar,
            'resources/choose-truecolor.svg', self.tr('Choose true colors'), self)
        self._act_choose_true_colors.triggered.connect(self._on_choose_true_colors)


    def _init_zoom_tools(self):
        '''
        Initialize zoom toolbar buttons.
        '''

        # Zoom In
        self._act_zoom_in = add_toolbar_action(self._toolbar,
            'resources/zoom-in.svg', self.tr('Zoom in'), self, QKeySequence.ZoomIn)
        self._act_zoom_in.triggered.connect(self._on_zoom_in)

        # Zoom Out
        self._act_zoom_out = add_toolbar_action(self._toolbar,
            'resources/zoom-out.svg', self.tr('Zoom out'), self, QKeySequence.ZoomOut)
        self._act_zoom_out.triggered.connect(self._on_zoom_out)

        # Zoom Level
        self._cbox_zoom = QComboBox()
        self._cbox_zoom.setEditable(True)
        self._cbox_zoom.setInsertPolicy(QComboBox.NoInsert)

        for v in self._zoom_options:
            self._cbox_zoom.addItem(f'{int(v * 100)}%', v)

        # self._cbox_zoom.setEditable(False) # TODO(donnie):  may want to support editing

        self._cbox_zoom.activated.connect(self._on_zoom_cbox)
        self._cbox_zoom.lineEdit().editingFinished.connect(self._on_zoom_cbox_edit_text)

        self._act_cbox_zoom = self._toolbar.addWidget(self._cbox_zoom)

    def _on_zoom_cbox_activated(self, data):
        print(f'Zoom combo-box activated:  {data}')


    def get_toolbar(self):
        return self._toolbar


    def sizeHint(self):
        ''' The default size of the zoom-pane widget is 400x400. '''
        if self._size_hint is None:
            return super().sizeHint()

        return self._size_hint


    def _on_dataset_added(self, index):
        if self._app_state.num_datasets() == 1:
            # We finally have a dataset!
            self._dataset_index = 0
            self._update_image()

            if self._initial_zoom is not None:
                self._rasterview.scale_image(self._initial_zoom)

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
            self._rasterview.set_display_bands(dialog.get_display_bands())


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

        new_scale = self._zoom_in_scale(scale)

        if self._max_zoom_scale is None or new_scale <= self._max_zoom_scale:
            self._rasterview.scale_image(new_scale)

        self._update_zoom_widgets()


    def _on_zoom_out(self, evt):
        ''' Zoom out the zoom-view by one level. '''
        scale = self._rasterview.get_scale()

        new_scale = self._zoom_out_scale(scale)

        if self._min_zoom_scale is None or new_scale >= self._min_zoom_scale:
            self._rasterview.scale_image(new_scale)

        self._update_zoom_widgets()


    def _zoom_in_scale(self, scale):
        '''
        Zoom in the display by 20%.  Subclasses can override this function to
        specify different zoom-in behaviors.
        '''
        return scale * 1.25


    def _zoom_out_scale(self, scale):
        '''
        Zoom out the display by 20%.  Subclasses can override this function to
        specify different zoom-out behaviors.
        '''
        return scale * 0.8


    def _on_zoom_cbox(self, index):
        ''' Zoom the zoom-view to the specified option in the zoom combo-box. '''
        self._cbox_zoom.lineEdit().clearFocus()
        self._rasterview.scale_image(self._cbox_zoom.currentData())
        self._update_zoom_widgets()


    def _on_zoom_cbox_edit_text(self):
        text = self._cbox_zoom.lineEdit().text()
        self._cbox_zoom.lineEdit().clearFocus()

        # Clean up the input text, and remove any percent sign on the end.
        text = text.strip()
        if len(text) == 0:
            return

        # print(f'Zoom edit-text changed to {text}')

        if text[-1] == '%':
            text = text[:-1]

        # Try to parse the text into a number.
        try:
            new_scale = float(text) / 100
        except:
            QMessageBox.warning(self, self.tr('Invalid Zoom Value'),
                self.tr('The zoom percentage must be a number.'))

            self._update_zoom_widgets()
            return

        # TODO(donnie):  These pop-up messages really cause problems with the UI!!

        # Make sure the zoom level is not too little.
        if self._min_zoom_scale is not None and new_scale < self._min_zoom_scale:
            QMessageBox.warning(self, self.tr('Invalid Zoom Value'),
                self.tr(f'The zoom percentage must be at least {int(self._min_zoom_scale * 100)}.'))

            self._update_zoom_widgets()
            return

        # Make sure the zoom level is not too much.
        if self._max_zoom_scale is not None and new_scale > self._max_zoom_scale:
            QMessageBox.warning(self, self.tr('Invalid Zoom Value'),
                self.tr(f'The zoom percentage must be at most {int(self._max_zoom_scale * 100)}.'))

            self._update_zoom_widgets()
            return

        # If we got here, the zoom level is a valid number within range.
        # Apply it!
        self._rasterview.scale_image(new_scale)
        self._update_zoom_widgets()


    def _update_zoom_widgets(self):
        scale = self._rasterview.get_scale()

        # Enable / disable zoom buttons based on scale
        self._act_zoom_out.setEnabled(self._min_zoom_scale is None or scale >= self._min_zoom_scale)
        self._act_zoom_in.setEnabled(self._max_zoom_scale is None or scale <= self._max_zoom_scale)

        # Set the zoom-level value
        if scale in self._zoom_options:
            self._cbox_zoom.lineEdit().clear()

            scale_idx = self._zoom_options.index(scale)
            self._cbox_zoom.setCurrentIndex(scale_idx)
        else:
            scale_str = f'{int(scale * 100)}%'
            # print(f'Setting line-edit text to {scale_str}')
            # self._cbox_zoom.setCurrentText(scale_str)
            self._cbox_zoom.lineEdit().setText(scale_str)


    def _update_image(self):
        dataset = None
        if self._app_state.num_datasets() > 0:
            dataset = self._app_state.get_dataset(self._dataset_index)

        # Only do this when the raster dataset actually changes,
        # or the displayed bands change, etc.
        if dataset != self._rasterview.get_raster_data():
            self._rasterview.set_raster_data(dataset)

        if dataset is None:
            return


    # TODO(donnie):  Make this function take a QPainter argument???
    # TODO(donnie):  Only pass in the bounding rectangle from the paint event???
    def _after_raster_paint(self, widget, paint_event):
        '''
        This method may be implemented by subclasses to draw additional
        information on top of the raster data.  The widget argument is the
        widget to draw into; a painter can be constructed like this:

            painter = QPainter(widget)
            ... # Draw stuff
            painter.end()

        The paint-event that prompted the call to this method is provided, so
        that data to draw may be clipped to the specified rectangle.
        '''
        pass
