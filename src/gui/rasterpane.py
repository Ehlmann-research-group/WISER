from enum import Enum

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .band_chooser import BandChooserDialog
from .dataset_chooser import DatasetChooser
from .rasterview import RasterView
from .util import add_toolbar_action, get_painter

from raster.dataset import find_display_bands, find_truecolor_bands

from raster.selection import SelectionType, Selection
from .selection_creator import RectangleSelectionCreator, \
    PolygonSelectionCreator, MultiPixelSelectionCreator


class RecenterMode(Enum):
    '''
    The recentering mode to use when important regions are being displayed in
    the raster pane.  Depending on the origin of an event, the recentering mode
    may be different for different panes.
    '''

    # Don't recenter the pane's display, regardless of the position of the
    # region of interest.
    NEVER = 0

    # Always recenter the pane's display such that the region of interest is
    # in the center.
    ALWAYS = 1

    # Recenter the pane's display if the region of interest is not visible.
    IF_NOT_VISIBLE = 2


class PixelReticleType(Enum):
    '''
    This enumeration specifies the different options for how a selected pixel
    is highlighted in the user interface.
    '''

    # Draw a "small cross" - the horizontal and vertical lines will only have
    # a relatively small extent.
    SMALL_CROSS = 1

    # Draw a "large cross" - the horizontal and vertical lines will extend to
    # the edges of the view.
    LARGE_CROSS = 2

    # Draw a "small cross" at low magnifications, but above a certain
    # magnification level (e.g. 4x), start drawing a box around the selected
    # pixel.
    SMALL_CROSS_BOX = 3


class RasterPane(QWidget):
    '''
    This widget provides a raster-view with an associated toolbar.
    '''

    # Signal:  the raster pane was shown or hidden
    #   - The value is True for "visible", False for "invisible".
    visibility_change = Signal(bool)

    # Signal:  the display bands in this view were changed
    #   - The int is the 0-based index of the data set whose display bands are
    #     changing
    #   - The tuple is either a 1-tuple or 3-tuple specifying the display bands
    #   - The Boolean argument is True for "global change," or False for "this
    #     view only"
    display_bands_change = Signal(int, tuple, bool)


    # Signal:  for when the user selects a raster pixel.  The coordinates of the
    # pixel in the raster image are reported:  QPoint(x, y).
    click_pixel = Signal(QPoint)


    # Signal:  the user created a selection in the pane.  The created selection
    # is passed as an argument.
    create_selection = Signal(Selection)


    # Signal:  the raster view's display area has changed.  The rectangle of the
    # new display area is reported to the signal handler, using raster dataset
    # coordinates:  QRect(x, y, width, height).
    viewport_change = Signal(QRect)


    def __init__(self, app_state, parent=None, size_hint=None,
                 embed_toolbar=True, select_tools=True,
                 min_zoom_scale=None, max_zoom_scale=None, zoom_options=None,
                 initial_zoom=None):
        super().__init__(parent=parent)

        # Initialize widget's internal state

        self._app_state = app_state

        # The index of the data-set being displayed
        self._dataset_index = None

        # The bands to display for each dataset.  Each entry in the list is a
        # tuple of the bands to display.
        self._display_bands = []

        self._size_hint = size_hint
        self._embed_toolbar = embed_toolbar

        self._min_zoom_scale = min_zoom_scale
        self._max_zoom_scale = max_zoom_scale
        self._zoom_options = zoom_options
        self._initial_zoom = initial_zoom

        self._viewport_highlight = None
        self._pixel_highlight = None

        self._creator = None

        # Initialize contents of the widget

        self._init_ui(select_tools=select_tools)

        # Register for events from the application state

        self._app_state.dataset_added.connect(self._on_dataset_added)
        self._app_state.dataset_removed.connect(self._on_dataset_removed)


    def _init_ui(self, select_tools=True):
        ''' Initialize the contents of this widget '''

        # TOOLBAR
        #=========

        self._toolbar = QToolBar(self.tr('Toolbar'), parent=self)

        if self._embed_toolbar:
            self._toolbar.setIconSize(QSize(20, 20))

        self._init_dataset_band_tools()
        self._toolbar.addSeparator()
        self._init_zoom_tools()

        if select_tools:
            self._toolbar.addSeparator()
            self._init_select_tools()

        # Raster image view widget

        forward = {
            'mousePressEvent'   : self._onRasterMousePress,
            'mouseReleaseEvent' : self._onRasterMouseRelease,
            'mouseMoveEvent'    : self._onRasterMouseMove,
            'keyPressEvent'     : self._onRasterKeyPress,
            'keyReleaseEvent'   : self._onRasterKeyRelease,
            'paintEvent'        : self._afterRasterPaint,
            'scrollContentsBy'  : self._afterRasterScroll,
        }
        self._rasterview = RasterView(parent=self, forward=forward)

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

    def _init_select_tools(self):
        '''
        Initialize the selection / region of interest tools
        '''

        # The selection-tools chooser is a drop down menu of selection tools.
        # First, populate the menu of tools, then create the chooser button.

        chooser = QToolButton()
        chooser.setIcon(QIcon('resources/select.svg'))
        chooser.setToolTip(self.tr('Selection tools'))

        # Without the parent= argument, the chooser doesn't show the menu.
        menu = QMenu(parent=chooser)
        chooser.setMenu(menu)
        chooser.setPopupMode(QToolButton.InstantPopup)

        act = menu.addAction(self.tr('Rectangle selection'))
        act.setData(SelectionType.RECTANGLE)

        act = menu.addAction(self.tr('Polygon selection'))
        act.setData(SelectionType.POLYGON)

        act = menu.addAction(self.tr('Multi-pixel selection'))
        act.setData(SelectionType.MULTI_PIXEL)

        act = menu.addAction(self.tr('Predicate selection'))
        act.setData(SelectionType.PREDICATE)

        self._toolbar.addWidget(chooser)

        chooser.triggered.connect(self._on_create_selection)


    def resizeEvent(self, event):
        '''
        Override the QtWidget resizeEvent() virtual method to fire an event that
        the visible region of the raster-view has changed.
        '''
        self._emit_viewport_change()


    def _onRasterMousePress(self, widget, mouse_event):
        if self._creator is not None:
            self._update_creator(self._creator.onMousePress(widget, mouse_event))

    def _onRasterMouseMove(self, widget, mouse_event):
        if self._creator is not None:
            self._update_creator(self._creator.onMouseMove(widget, mouse_event))

    def _onRasterMouseRelease(self, widget, mouse_event):
        '''
        When the display image is clicked on, this method gets invoked, and it
        translates the click event's coordinates into the location on the
        raster data set.
        '''
        if self._creator is not None:
            self._update_creator(self._creator.onMouseRelease(widget, mouse_event))

        else:
            # Map the coordinate of the mouse-event to the actual raster-image
            # pixel that was clicked, then emit a signal.
            r_coord = self._rasterview.image_coord_to_raster_coord(mouse_event.localPos())
            self.click_pixel.emit(r_coord)


    def _onRasterKeyPress(self, widget, key_event):
        if self._creator is not None:
            self._update_creator(self._creator.onKeyPress(widget, key_event))

    def _onRasterKeyRelease(self, widget, key_event):
        if self._creator is not None:
            self._update_creator(self._creator.onKeyRelease(widget, key_event))


    def _afterRasterScroll(self, widget, dx, dy):
        '''
        This function is called when the scroll-area moves around.  Fire an
        event that the visible region of the raster-view has changed.
        '''
        self._emit_viewport_change()


    def _update_creator(self, create_done):
        if create_done:
            selection = self._creator.get_selection()
            print(f'TODO:  Store selection {selection} on application state')
            self._creator = None

        self._rasterview.update()


    def _emit_viewport_change(self):
        ''' A helper that emits the viewport-changed event. '''
        self.viewport_change.emit(self._rasterview.get_visible_region())



    def _on_zoom_cbox_activated(self, data):
        # print(f'Zoom combo-box activated:  {data}')
        pass


    def get_rasterview(self):
        return self._rasterview


    def get_toolbar(self):
        return self._toolbar


    def get_current_dataset(self):
        if self._dataset_index is None:
            return None

        return self._app_state.get_dataset(self._dataset_index)


    def set_display_bands(self, index, bands):
        if index < 0 or index >= self._app_state.num_datasets():
            raise ValueError(f'index must be in the range [0, {self._app_state.num_datasets()}); got {index}')

        if len(bands) not in [1, 3]:
            raise ValueError(f'bands must be either a 1-tuple or 3-tuple; got {bands}')

        # print(f'Display-band information:  {self._display_bands}')

        self._display_bands[index] = bands

        # If the specified data set is the one currently being displayed, update
        # the UI display.
        if index == self._dataset_index:
            self._rasterview.set_display_bands(bands)


    def make_point_visible(self, x, y):
        self._rasterview.make_point_visible(x, y)


    def set_viewport_highlight(self, viewport):
        # print(f'{self}:  Setting viewport highlight to {viewport}')

        self._viewport_highlight = viewport

        # If the specified viewport highlight region is not entirely within this
        # raster-view's visible area, scroll such that the viewport highlight is
        # in the middle of the raster-view's visible area.

        visible = self._rasterview.get_visible_region()
        if visible is None or viewport is None:
            self._rasterview.update()
            return

        if not visible.contains(viewport):
            center = viewport.center()
            self._rasterview.make_point_visible(center.x(), center.y())

        # Repaint raster-view
        self._rasterview.update()


    def set_pixel_highlight(self, pixel, recenter=RecenterMode.ALWAYS):
        self._pixel_highlight = pixel
        visible = self._rasterview.get_visible_region()
        if visible is None or pixel is None:
            self._rasterview.update()
            return

        do_recenter = False
        if recenter == RecenterMode.ALWAYS:
            do_recenter = True
        elif recenter == RecenterMode.IF_NOT_VISIBLE:
            do_recenter = not visible.contains(pixel)

        if do_recenter:
            # Scroll the raster-view such that the pixel is in the middle of the
            # raster-view's visible area.
            self._rasterview.make_point_visible(pixel.x(), pixel.y())

        # Repaint raster-view
        self._rasterview.update()


    def sizeHint(self):
        '''
        If a preferred size was passed to the constructor, report it as the
        size-hint for this widget.
        '''
        if self._size_hint is None:
            return super().sizeHint()

        return self._size_hint


    def showEvent(self, event):
        super().showEvent(event)
        self.visibility_change.emit(True)


    def hideEvent(self, event):
        super().hideEvent(event)
        self.visibility_change.emit(False)


    def _on_dataset_added(self, index):
        new_dataset = self._app_state.get_dataset(index)

        bands = find_display_bands(new_dataset)
        self._display_bands.insert(index, bands)

        # print(f'on_dataset_added:  band info:  {self._display_bands}')

        if self._app_state.num_datasets() == 1:
            # We finally have a dataset!
            self._dataset_index = 0
            self._update_image()

            if self._initial_zoom is not None:
                self._rasterview.scale_image(self._initial_zoom)

            self._update_zoom_widgets()


    def _on_dataset_removed(self, index):
        del self._display_bands[index]

        # print(f'on_dataset_removed:  band info:  {self._display_bands}')

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
        dataset = self.get_current_dataset()
        display_bands = self._rasterview.get_display_bands()

        dialog = BandChooserDialog(dataset, display_bands, parent=self)
        dialog.setModal(True)

        if dialog.exec_() == QDialog.Accepted:
            bands = dialog.get_display_bands()
            is_global = dialog.apply_globally()

            self.display_bands_change.emit(self._dataset_index, bands, is_global)

            # Only update our display bands if the change was not global, since
            # if it was, the main application controller will change everybody's
            # display bands.
            if not is_global:
                self.set_display_bands(self._dataset_index, bands)


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


    def _on_create_selection(self, act):
        selection_type = act.data()

        if selection_type == SelectionType.RECTANGLE:
            self._creator = RectangleSelectionCreator(self)
            # TODO:  Update status bar to indicate the creation of the rectangle
            #        selection.

        elif selection_type == SelectionType.POLYGON:
            self._creator = PolygonSelectionCreator(self)
            # TODO:  Update status bar to indicate the creation of the polygon
            #        selection.

        elif selection_type == SelectionType.MULTI_PIXEL:
            self._creator = MultiPixelSelectionCreator(self)
            # TODO:  Update status bar to indicate the creation of the
            #        multi-pixel selection.

        elif selection_type == SelectionType.PREDICATE:
            ok, pred_text = QInputDialog.getText(self,
                self.tr('Create predicate selection'),
                self.tr('Enter a predicate specifying what pixels to select.'))

            if ok and pred_text:
                print(f'TODO:  Create selection from predicate {pred_text}')

        else:
            QMessageBox.warning(self, self.tr('Unsupported Feature'),
                f'ISWB does not yet support selections of type {selection_type}')


    def _update_image(self):
        dataset = None
        if self._app_state.num_datasets() > 0:
            dataset = self.get_current_dataset()

        # Only do this when the raster dataset actually changes,
        # or the displayed bands change, etc.
        if dataset != self._rasterview.get_raster_data():
            bands = self._display_bands[self._dataset_index]
            self._rasterview.set_raster_data(dataset, bands)

        if dataset is None:
            return


    # TODO(donnie):  Make this function take a QPainter argument???
    # TODO(donnie):  Only pass in the bounding rectangle from the paint event???
    def _afterRasterPaint(self, widget, paint_event):
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

        # Draw the viewport highlight, if there is one
        self._draw_viewport_highlight(widget, paint_event)

        # Draw the pixel highlight, if there is one
        self._draw_pixel_highlight(widget, paint_event)

        if self._creator is not None:
            with get_painter(widget) as painter:
                self._creator.draw_state(painter)


    def _draw_viewport_highlight(self, widget, paint_event):
        '''
        This helper function draws the viewport highlight in this raster-pane.
        The color to draw with is taken from the application state's config.

        If there is no viewport highlight, this is a no-op.
        '''

        if self._viewport_highlight is None:
            return

        # Draw the viewport highlight.
        with get_painter(widget) as painter:
            color = self._app_state.get_color_of('viewport-highlight')
            painter.setPen(QPen(color))

            box = self._viewport_highlight
            scale = self._rasterview.get_scale()

            scaled = QRect(box.x() * scale, box.y() * scale,
                           box.width() * scale, box.height() * scale)

            if scaled.width() >= widget.width():
                scaled.setWidth(widget.width() - 1)

            if scaled.height() >= widget.height():
                scaled.setHeight(widget.height() - 1)

            painter.drawRect(scaled)


    def _draw_pixel_highlight(self, widget, paint_event):
        '''
        This helper function draws the "currently selected pixel" highlight in
        this raster-pane.  The color to draw with is taken from the application
        state's config.

        If there is no "currently selected pixel" highlight, this is a no-op.
        '''

        if self._pixel_highlight is None:
            return

        with get_painter(widget) as painter:
            color = self._app_state.get_color_of('pixel-highlight')
            painter.setPen(QPen(color))

            # (ds_x, ds_y) is the coordinate within the data-set.
            ds_x = self._pixel_highlight.x()
            ds_y = self._pixel_highlight.y()

            # This is the size of individual data-set pixels in the display
            # coordinate system.
            scale = self._rasterview.get_scale()

            # This is the center of the highlighted pixel.
            screen_x = (ds_x + 0.5) * scale
            screen_y = (ds_y + 0.5) * scale


            # Draw a reticle centered on the highlighted pixel.

            reticle_type = self._app_state.get_config('pixel-reticle-type', PixelReticleType.SMALL_CROSS)
            if reticle_type == PixelReticleType.SMALL_CROSS:
                painter.drawLine(screen_x - 15, screen_y, screen_x + 15, screen_y)
                painter.drawLine(screen_x, screen_y - 15, screen_x, screen_y + 15)

            elif reticle_type == PixelReticleType.LARGE_CROSS:
                painter.drawLine(0, screen_y, widget.width(), screen_y)
                painter.drawLine(screen_x, 0, screen_x, widget.height())

            else:
                raise ValueError(f'Unrecognized reticle-type {reticle_type}')

            # TODO(donnie):  Figure out how to incorporate this with the above.
            '''
            # Draw a box around the highlighted pixel, but only if it's larger
            # than a certain scale.
            if scale >= 4:
                # Compute the rectangle that will border the specified pixel.
                # Subtract 1 from the width and height to keep the rectangle
                # from spilling into the neighboring pixel.
                scaled = QRect(screen_x, screen_y, scale, scale)
                painter.drawRect(scaled)
            '''
