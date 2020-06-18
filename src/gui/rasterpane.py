from enum import Enum
from typing import List, Optional, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .app_config import PixelReticleType
from .band_chooser import BandChooserDialog
from .dataset_chooser import DatasetChooser
from .rasterview import RasterView
from .util import add_toolbar_action, get_painter

from raster.dataset import RasterDataSet, find_display_bands, find_truecolor_bands
from raster.selection import SelectionType, Selection, SinglePixelSelection

from .roi import draw_roi

from .ui_selection_rectangle import RectangleSelectionCreator, RectangleSelectionEditor
from .ui_selection_polygon import PolygonSelectionCreator # , PolygonSelectionEditor
from .ui_selection_multi_pixel import MultiPixelSelectionCreator # , MultiPixelSelectionEditor


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


class RasterPane(QWidget):
    '''
    This widget provides a raster-view with an associated toolbar.
    '''

    # Signal:  the raster pane was shown or hidden
    #   - The value is True for "visible", False for "invisible".
    visibility_change = Signal(bool)

    views_changed = Signal(tuple)

    # Signal:  the display bands in this view were changed
    #   - The int is the numeric ID of the dataset whose display bands are
    #     changing
    #   - The tuple is either a 1-tuple or 3-tuple specifying the display bands
    #   - The Boolean argument is True for "global change," or False for "this
    #     view only"
    display_bands_change = Signal(int, tuple, bool)


    # Signal:  for when the user selects a raster pixel.  The signal reports the
    # (row, col) coordinates of the raster-view that was clicked in, and also
    # the coordinates of the pixel in the raster image.
    click_pixel = Signal(tuple, QPoint)


    # Signal:  the user created a selection in the pane.  The created selection
    # is passed as an argument.
    create_selection = Signal(Selection)


    # Signal:  one or more raster-views changed their display viewport.  The
    # signal reports one optional 2-tuple with the raster-view's position in the
    # pane, or the argument will be None if all raster-views changed their
    # display viewport.
    viewport_change = Signal(tuple)


    def __init__(self, app_state, parent=None, size_hint=None,
                 embed_toolbar=True, select_tools=True,
                 min_zoom_scale=None, max_zoom_scale=None, zoom_options=None,
                 initial_zoom=None):
        super().__init__(parent=parent)

        # Initialize widget's internal state

        self._app_state = app_state

        # The numeric ID of the data-set being displayed
        # TODO(donnie):  Since panes can show multiple raster data sets at once,
        #     this value is no longer useful.
        # self._dataset_id: Optional[int] = None

        # The bands to display for each dataset.  Keys are dataset IDs, and
        # entries are the corresponding tuple of the bands to display for the
        # dataset.
        self._display_bands: Dict[int, Tuple] = {}

        self._size_hint = size_hint
        self._embed_toolbar = embed_toolbar

        self._min_zoom_scale = min_zoom_scale
        self._max_zoom_scale = max_zoom_scale
        self._zoom_options = zoom_options
        self._initial_zoom = initial_zoom

        self._viewport_highlight = None
        self._pixel_highlight: SinglePixelSelection = None

        self._task_delegate = None

        # Initialize contents of the widget

        self._init_ui(select_tools=select_tools)

        # Register for events from the application state

        self._app_state.dataset_added.connect(self._on_dataset_added)
        self._app_state.dataset_removed.connect(self._on_dataset_removed)
        self._app_state.stretch_changed.connect(self._on_stretch_changed)


    def _init_ui(self, select_tools=True):
        ''' Initialize the contents of this widget '''

        #=========
        # TOOLBAR

        self._toolbar = QToolBar(self.tr('Toolbar'), parent=self)

        if self._embed_toolbar:
            self._toolbar.setIconSize(QSize(20, 20))

        # Raster-view widget(s) and layout

        # Invalid value, ensures _init_rasterviews will initialize the views.
        self._num_views = (0, 0)

        self._rasterviews = {}
        self._rasterview_layout = QGridLayout()
        self._rasterview_layout.setContentsMargins(QMargins(0, 0, 0, 0))
        self._rasterview_layout.setHorizontalSpacing(0)
        self._rasterview_layout.setVerticalSpacing(0)

        self._init_rasterviews()  # Default dimension is (1, 1)

        # Toolbar and other layout details

        # Initialize the toolbar after the raster-views, because the dataset
        # chooser widget references the rasterviews.
        self._init_toolbar()

        if self._embed_toolbar:
            self._rasterview_layout.setMenuBar(self._toolbar)

        self.setLayout(self._rasterview_layout)

        self._update_zoom_widgets()


    def _init_toolbar(self):
        self._init_dataset_tools()
        self._toolbar.addSeparator()
        self._init_zoom_tools()
        self._toolbar.addSeparator()
        self._init_select_tools()


    def _init_dataset_tools(self):
        '''
        Initialize dataset-management toolbar buttons.
        '''

        # Dataset / Band Tools

        self._dataset_chooser = DatasetChooser(self, self._app_state)
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


    def _init_rasterviews(self, num_views: Tuple[int, int]=(1, 1)):
        if num_views[0] < 1 or num_views[1] < 1:
            raise ValueError(f'Minimum number of raster-view rows/cols is 1, got {num_views}')

        # If the current raster-view layout is the same as the requested
        # raster-view layout, ignore.
        if self._num_views == num_views:
            return

        forward = {
            'mousePressEvent'   : self._onRasterMousePress,
            'mouseReleaseEvent' : self._onRasterMouseRelease,
            'mouseMoveEvent'    : self._onRasterMouseMove,
            'keyPressEvent'     : self._onRasterKeyPress,
            'keyReleaseEvent'   : self._onRasterKeyRelease,
            'contextMenuEvent'  : self._onRasterContextMenu,
            'paintEvent'        : self._afterRasterPaint,
            'scrollContentsBy'  : self._afterRasterScroll,
        }

        # There are existing raster-views; clean them all up.
        # TODO(donnie):  Just clean up the ones that are now "out of bounds" for
        #     the new layout.
        if len(self._rasterviews) != 0:
            # Remove each raster-view from the grid layout
            for (position, rasterview) in self._rasterviews.items():
                # Remove the rasterview from the layout, and hide it.
                self._rasterview_layout.removeWidget(rasterview)
                rasterview.hide()

            # Clear out the dictionary of raster-views
            self._rasterviews.clear()

        # Create new raster-views as specified by the arguments
        multiviews = (num_views != (1, 1))
        (rows, cols) = num_views
        for row in range(rows):
            for col in range(cols):
                rasterview = RasterView(parent=self, forward=forward)
                rasterview.setContextMenuPolicy(Qt.DefaultContextMenu)

                rv_container = rasterview
                if multiviews:
                    # TODO(donnie):  All this malarkey is so we can add a toolbar to
                    #     the individual rasterviews.
                    rv_container = QWidget()
                    rv_layout = QGridLayout()
                    rv_layout.setContentsMargins(QMargins(0, 0, 0, 0))
                    rv_container.setLayout(rv_layout)
                    rv_layout.addWidget(rasterview)

                    rv_toolbar = QToolBar(
                        self.tr('RasterView [{row}, {col}] Toolbar').format(row=row, col=col),
                        parent=rv_container)
                    rv_toolbar.setIconSize(QSize(16, 16))

                    # rv_ds_name = QComboBox()
                    # rv_toolbar.addWidget(rv_ds_name)

                    rv_dataset_chooser = DatasetChooser(None, self._app_state)
                    rv_toolbar.addWidget(rv_dataset_chooser)
                    # TODO:  rv_dataset_chooser.triggered.connect(self._on_dataset_changed)

                    rv_act_band_chooser = add_toolbar_action(rv_toolbar,
                        'resources/choose-bands.svg', self.tr('Band chooser'), rv_container)
                    # TODO:  rv_act_band_chooser.triggered.connect(self._on_band_chooser)


                position = (row, col)
                # TODO(donnie):  THIS NEEDS TO CHANGE IF THIS CONTAINER STUFF WORKS
                # self._rasterviews[position] = (rasterview, rv_container)
                self._rasterviews[position] = rasterview
                # self._rasterview_layout.addWidget(rasterview, row, col)
                self._rasterview_layout.addWidget(rv_container, row, col)

        self._num_views = num_views
        self.views_changed.emit(self._num_views)


    def _get_rasterview_position(self, rasterview) -> Optional[Tuple[int, int]]:
        '''
        Returns the position of the rasterview in the raster-pane, as a
        (row, col) tuple.

        If the specified rasterview is not recognized in the pane, None is
        returned.
        '''
        for pos, rv in self._rasterviews.items():
            if rv is rasterview:
                return pos

        return None


    def get_num_views(self) -> Tuple[int, int]:
        '''
        Returns a 2-tuple of the form (rows, cols) specifying the number of
        views currently showing in this raster pane.
        '''
        return self._num_views


    def is_multi_view(self) -> bool:
        '''
        Returns True if this raster pane is currently showing more than one
        view, or False otherwise.
        '''
        return len(self._rasterviews) > 1


    def get_rasterview(self, rasterview_pos=(0, 0)):
        '''
        Returns the raster-view at the specified (row, column) position in the
        raster-pane.  The default (row, column) value is (0, 0), which can be
        used by panes that will only ever have one raster-view.
        '''
        return self._rasterviews[rasterview_pos]


    def get_all_visible_regions(self) -> List:
        '''
        Returns a list of all visible regions
        '''
        # Get the list of visible regions, but filter out empty regions.
        regions = [rv.get_visible_region() for rv in self._rasterviews.values()]
        regions = [r for r in regions if r is not None]
        return regions


    def get_scale(self):
        '''
        Returns the current zoom scale of this raster pane.  Even when a pane
        contains multiple views, all views use the same scale, so this function
        simply returns the scale of the top left view in the pane.
        '''
        return self._rasterviews[(0, 0)].get_scale()


    def set_scale(self, scale):
        '''
        Sets the zoom scale of this raster pane.  When a pane contains multiple
        raster views, the scale is set on all of the views.

        As one would expect, this will generate a viewport-changed event.
        '''
        for rasterview in self._rasterviews.values():
            rasterview.scale_image(scale)



    def resizeEvent(self, event):
        '''
        Override the QtWidget resizeEvent() virtual method to fire an event that
        the visible region of the raster-view has changed.
        '''
        self._emit_viewport_change()


    def _onRasterMousePress(self, rasterview, mouse_event):
        if self._task_delegate is not None:
            done = self._task_delegate.on_mouse_press(widget, mouse_event)
            self._update_delegate(done)

    def _onRasterMouseMove(self, rasterview, mouse_event):
        if self._task_delegate is not None:
            done = self._task_delegate.on_mouse_move(widget, mouse_event)
            self._update_delegate(done)

    def _onRasterMouseRelease(self, rasterview, mouse_event):
        '''
        When the display image is clicked on, this method gets invoked, and it
        translates the click event's coordinates into the location on the
        raster data set.
        '''
        if self._task_delegate is not None:
            done = self._task_delegate.on_mouse_release(widget, mouse_event)
            self._update_delegate(done)

        else:
            if mouse_event.button() == Qt.LeftButton:
                # Map the coordinate of the mouse-event to the actual raster-image
                # pixel that was clicked, then emit a signal.
                r_coord = rasterview.image_coord_to_raster_coord(mouse_event.localPos())

                rasterview_pos = self._get_rasterview_position(rasterview)
                self.click_pixel.emit(rasterview_pos, r_coord)


    def _onRasterKeyPress(self, rasterview, key_event):
        if self._task_delegate is not None:
            done = self._task_delegate.on_key_press(widget, key_event)
            self._update_delegate(done)

    def _onRasterKeyRelease(self, rasterview, key_event):
        if self._task_delegate is not None:
            done = self._task_delegate.on_key_release(widget, key_event)
            self._update_delegate(done)

    def _onRasterContextMenu(self, rasterview, context_menu_event):
        menu = QMenu(self)

        # TODO(donnie):  Set up handler for the action
        act = menu.addAction(self.tr('Annotate location'))

        # Calculate the coordinate of the click in dataset coordinates
        ds_coord = rasterview.image_coord_to_raster_coord(context_menu_event.pos())

        # Find Regions of Interest that include the click location.
        picked_rois = []
        for (name, roi) in self._app_state.get_rois().items():
            if roi.is_picked_by(ds_coord):
                picked_rois.append(roi)

        if len(picked_rois) > 0:
            # At least one region of interest was picked
            menu.addSeparator()

            for roi in picked_rois:
                roi_menu = menu.addMenu(roi.get_name())

                # TODO(donnie):  Set up handlers for the actions

                roi_menu.addAction(self.tr('Show spectrum'))
                roi_menu.addAction(self.tr('Edit geometry'))
                roi_menu.addAction(self.tr('Edit metadata...'))
                roi_menu.addAction(self.tr('Export spectrum...'))
                roi_menu.addSeparator()
                roi_menu.addAction(self.tr('Delete region...'))

        menu.exec_(context_menu_event.globalPos())

    def _afterRasterScroll(self, rasterview, dx, dy):
        '''
        This function is called when the raster-view's scrollbars are moved.

        It fires an event that the visible region of the raster-view has
        changed.

        If multiple raster-views are active, and scrolling is linked, this also
        propagates the scroll changes to the other raster-views.
        '''
        self._emit_viewport_change(self._get_rasterview_position(rasterview))


    def _update_delegate(self, done):
        if done:
            # selection = self._creator.get_selection()
            # print(f'TODO:  Store selection {selection} on application state')
            # TODO(donnie):  How to handle completion of task delegates???
            self._task_delegate.finish()
            self._task_delegate = None

        # TODO:  This needs to be adjusted for multi-views
        self.get_rasterview().update()


    def _emit_viewport_change(self, rasterview_pos=None):
        ''' A helper that emits the viewport-changed event. '''
        self.viewport_change.emit(rasterview_pos)


    def _on_zoom_cbox_activated(self, data):
        # print(f'Zoom combo-box activated:  {data}')
        pass


    def get_toolbar(self):
        return self._toolbar


    def get_current_dataset(self, rasterview_pos=(0, 0)) -> Optional[RasterDataSet]:
        '''
        Returns the current dataset being displayed in the specified view of the
        raster pane.

        TODO(donnie):  Eventually this will become obsolete, when a raster-view
            can display multiple images at a time.
        '''
        rasterview = self.get_rasterview(rasterview_pos)
        return rasterview.get_raster_data()


    def show_dataset(self, dataset, rasterview_pos=(0, 0)):
        '''
        Sets the dataset being displayed in the specified view of the raster
        pane.
        '''
        rasterview = self.get_rasterview(rasterview_pos)

        # If the rasterview is already showing the specified dataset, skip!
        if rasterview.get_raster_data() is dataset:
            return

        ds_id = dataset.get_id()
        bands = self._display_bands[ds_id]
        stretches = self._app_state.get_stretches(ds_id, bands)

        rasterview.set_raster_data(dataset, bands, stretches)


    def set_display_bands(self, ds_id: int, bands: Tuple):
        # TODO(donnie):  Verify the dataset ID?

        if len(bands) not in [1, 3]:
            raise ValueError(f'bands must be either a 1-tuple or 3-tuple; got {bands}')

        # print(f'Display-band information:  {self._display_bands}')

        self._display_bands[ds_id] = bands

        # If the specified data set is the one currently being displayed, update
        # the UI display.
        if ds_id == self._dataset_id:
            # Get the stretches at the same time, so that we only update the
            # raster-view once.
            stretches = self._app_state.get_stretches(self._dataset_id, bands)
            # TODO(donnie):  What to do for multi-views?
            self.get_rasterview().set_display_bands(bands, stretches=stretches)


    def make_point_visible(self, x, y, rasterview_pos=(0, 0)):
        if rasterview_pos is not None:
            self.get_rasterview(rasterview_pos).make_point_visible(x, y)

        else:
            for rv in self._rasterviews.values():
                rv.make_point_visible(x, y)


    def set_viewport_highlight(self, viewport):
        '''
        Sets the "viewport highlight" to be displayed in this raster-pane.  This
        is used to allow the Context Pane to show the Main View viewport, and
        the Main View can show the Zoom Pane viewport.

        TODO(donnie):  This will need to be updated for multiple viewports and
        linked windows.
        '''
        # print(f'{self}:  Setting viewport highlight to {viewport}')

        self._viewport_highlight = viewport

        # If the specified viewport highlight region is not entirely within this
        # raster-view's visible area, scroll such that the viewport highlight is
        # in the middle of the raster-view's visible area.

        # TODO(donnie):  Do we want to force all raster-views to switch to
        #     displaying the viewport?
        for rasterview in self._rasterviews.values():
            visible = rasterview.get_visible_region()
            if visible is None or viewport is None or isinstance(viewport, list):
                rasterview.update()
                continue

            if not visible.contains(viewport):
                center = viewport.center()
                rasterview.make_point_visible(center.x(), center.y())

            # Repaint raster-view
            rasterview.update()


    def set_pixel_highlight(self, pixel_sel: Optional[SinglePixelSelection],
                            recenter:RecenterMode=RecenterMode.ALWAYS):
        '''
        Sets the "pixel highlight" to be displayed in this raster-pane.  This is
        used by both the main image window and the zoom window, to indicate the
        pixel most recently selected by the user.
        '''
        self._pixel_highlight = pixel_sel

        coord = None
        dataset = None
        if pixel_sel is not None:
            coord = pixel_sel.get_pixel()
            dataset = pixel_sel.get_dataset()

        for rasterview in self._rasterviews.values():
            # Get the current dataset and visible region of the rasterview
            rv_dataset = rasterview.get_raster_data()
            visible = rasterview.get_visible_region()

            if rv_dataset is None or visible is None or pixel_sel is None:
                # Don't worry about recentering things, or displaying things -
                # we are missing something that is necessary to display a pixel
                # selection anyway.  Just update the rasterview to remove any
                # pixel selection, then move on.
                rasterview.update()
                continue

            if dataset is not None and rv_dataset is not dataset:
                # The selection's dataset was specified, and the rasterview is
                # showing a different dataset.  Update the rasterview to remove
                # any pixel selection, and move on.
                rasterview.update()
                continue

            # If we got here, we want to show the selection in this rasterview.
            # The only question is whether we need to recenter the rasterview to
            # ensure that the pixel selection is visible.
            do_recenter = False
            if recenter == RecenterMode.ALWAYS:
                do_recenter = True
            elif recenter == RecenterMode.IF_NOT_VISIBLE:
                do_recenter = not visible.contains(coord)

            if do_recenter:
                # Scroll the raster-view such that the pixel is in the middle of the
                # raster-view's visible area.
                rasterview.make_point_visible(coord.x(), coord.y())

            # Repaint raster-view
            rasterview.update()


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


    def _on_dataset_added(self, ds_id):
        '''
        This function handles "dataset added" events from the application state.
        It records the initial display bands to use for the dataset.  Also, if
        this is the first dataset loaded, the function shows it in all
        rasterviews.
        '''

        dataset = self._app_state.get_dataset(ds_id)
        bands = find_display_bands(dataset)
        self._display_bands[ds_id] = bands

        # print(f'on_dataset_added:  band info:  {self._display_bands}')

        if self._app_state.num_datasets() == 1:
            # We finally have a dataset!

            for rasterview in self._rasterviews.values():
                # Only do this when the raster dataset actually changes,
                # or the displayed bands change, etc.
                if dataset != rasterview.get_raster_data():
                    bands = self._display_bands[ds_id]
                    stretches = self._app_state.get_stretches(ds_id, bands)
                    rasterview.set_raster_data(dataset, bands, stretches)

                if self._initial_zoom is not None:
                    rasterview.scale_image(self._initial_zoom)

            self._update_zoom_widgets()


    def _on_dataset_removed(self, ds_id):
        '''
        This function handles "dataset removed" events from the application
        state.  It cleans up the locally-recorded display band info for the
        dataset.  Also, if any rasterviews are showing the dataset, the function
        switches them to a different dataset (if more than one is loaded).
        '''
        del self._display_bands[ds_id]

        # print(f'on_dataset_removed:  band info:  {self._display_bands}')

        for rasterview in self._rasterviews.values():
            rv_ds = rasterview.get_raster_data()
            if rv_ds is not None and rv_ds.get_id() == ds_id:
                rasterview.set_raster_data(None, None)


    def _on_dataset_changed(self, act):
        (rasterview_pos, ds_id) = act.data()
        dataset = self._app_state.get_dataset(ds_id)
        self.show_dataset(dataset, rasterview_pos)


    def _on_band_chooser(self, act):
        dataset = self.get_current_dataset()
        # TODO(donnie):  This needs to change with multi-views
        display_bands = self.get_rasterview().get_display_bands()

        dialog = BandChooserDialog(dataset, display_bands, parent=self)
        dialog.setModal(True)

        if dialog.exec_() == QDialog.Accepted:
            bands = dialog.get_display_bands()
            is_global = dialog.apply_globally()

            self.display_bands_change.emit(self._dataset_id, bands, is_global)

            # Only update our display bands if the change was not global, since
            # if it was, the main application controller will change everybody's
            # display bands.
            if not is_global:
                self.set_display_bands(self._dataset_id, bands)


    def _on_stretch_changed(self, ds_id, bands):
        # If we aren't displaying the dataset whose stretch was changed, ignore
        # the event.
        if ds_id != self._dataset_id:
            return

        # TODO(donnie):  What to do with multi-views?
        rasterview = self.get_rasterview()
        bands = self.rasterview.get_display_bands()
        stretches = self._app_state.get_stretches(self._dataset_id, bands)
        rasterview.set_stretches(stretches)


    def _on_zoom_in(self, evt):
        ''' Zoom in the zoom-view by one level. '''
        scale = self.get_scale()
        new_scale = self._zoom_in_scale(scale)

        if self._max_zoom_scale is None or new_scale <= self._max_zoom_scale:
            self.set_scale(new_scale)

        self._update_zoom_widgets()


    def _on_zoom_out(self, evt):
        ''' Zoom out the zoom-view by one level. '''
        scale = self.get_scale()
        new_scale = self._zoom_out_scale(scale)

        if self._min_zoom_scale is None or new_scale >= self._min_zoom_scale:
            self.set_scale(new_scale)

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
        ''' Zoom the display to the specified option in the zoom combo-box. '''
        self._cbox_zoom.lineEdit().clearFocus()
        self.set_scale(self._cbox_zoom.currentData())
        self._update_zoom_widgets()


    def _on_zoom_cbox_edit_text(self):
        '''
        Zoom the display to the amount that the user typed into the zoom
        combo-box's line edit.  The user may specify this as a percentage
        (e.g. "50.3%"), or they may specify it as a number (e.g. "50.3"); these
        values are all normalized to a whole number with a percent sign
        (e.g. "50%").
        '''
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
        self.set_scale(new_scale)
        self._update_zoom_widgets()


    def _update_zoom_widgets(self):
        scale = self.get_scale()

        # Enable / disable zoom buttons based on scale
        self._act_zoom_out.setEnabled(self._min_zoom_scale is None or
                                      scale >= self._min_zoom_scale)
        self._act_zoom_in.setEnabled(self._max_zoom_scale is None or
                                     scale <= self._max_zoom_scale)

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
            self._task_delegate = RectangleSelectionCreator(self._app_state, self._rasterview)
            # TODO:  Update status bar to indicate the creation of the rectangle
            #        selection.

        elif selection_type == SelectionType.POLYGON:
            self._task_delegate = PolygonSelectionCreator(self._app_state, self._rasterview)
            # TODO:  Update status bar to indicate the creation of the polygon
            #        selection.

        elif selection_type == SelectionType.MULTI_PIXEL:
            self._task_delegate = MultiPixelSelectionCreator(self._app_state, self._rasterview)
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



    # TODO(donnie):  Make this function take a QPainter argument???
    # TODO(donnie):  Only pass in the bounding rectangle from the paint event???
    def _afterRasterPaint(self, rasterview, widget, paint_event):
        '''
        This method may be implemented by subclasses to draw additional
        information on top of the raster data.

        The rasterview argument is the rasterview generating the after-paint
        event.

        The widget argument is the widget to draw into; a painter can be
        constructed like this:

            painter = QPainter(widget)
            ... # Draw stuff
            painter.end()

        The paint-event that prompted the call to this method is provided, so
        that data to draw may be clipped to the specified rectangle.
        '''

        # Draw regions of interest
        self._draw_regions_of_interest(rasterview, widget, paint_event)

        # Draw the viewport highlight, if there is one
        self._draw_viewport_highlight(rasterview, widget, paint_event)

        # Draw the pixel highlight, if there is one
        self._draw_pixel_highlight(rasterview, widget, paint_event)

        if self._task_delegate is not None:
            with get_painter(widget) as painter:
                self._task_delegate.draw_state(painter)


    def _draw_regions_of_interest(self, rasterview, widget, paint_event):
        # active_roi = self._app_state.get_active_roi()
        with get_painter(widget) as painter:
            for (name, roi) in self._app_state.get_rois().items():
                # print(f'{name}: {roi}')
                # TODO(donnie):  This needs to change for multi-views
                draw_roi(self.get_rasterview(), painter, roi)


    def _draw_viewport_highlight(self, rasterview, widget, paint_event):
        '''
        This helper function draws the viewport highlight in this raster-pane.
        The color to draw with is taken from the application state's config.

        If there is no viewport highlight, this is a no-op.
        '''

        if self._viewport_highlight is None:
            return

        # The viewport highlight will either be a box (QRect or QRectF), or it
        # will be a list of boxes in some circumstances.
        if self._viewport_highlight is not None and \
           not isinstance(self._viewport_highlight, list):
            highlights = [self._viewport_highlight]
        else:
            highlights = self._viewport_highlight

        # Draw the viewport highlight.
        with get_painter(widget) as painter:
            color = self._app_state.get_color_of('viewport-highlight')
            painter.setPen(QPen(color))

            for box in highlights:
                scale = self.get_scale()

                scaled = QRect(box.x() * scale, box.y() * scale,
                               box.width() * scale, box.height() * scale)

                if scaled.width() >= widget.width():
                    scaled.setWidth(widget.width() - 1)

                if scaled.height() >= widget.height():
                    scaled.setHeight(widget.height() - 1)

                painter.drawRect(scaled)


    def _draw_pixel_highlight(self, rasterview, widget, paint_event):
        '''
        This helper function draws the "currently selected pixel" highlight in
        this raster-pane.  The color to draw with is taken from the application
        state's config.

        If there is no "currently selected pixel" highlight, or if the highlight
        specifies a different dataset than the rasterview is showing, this is a
        no-op.
        '''

        if self._pixel_highlight is None:
            return

        dataset = self._pixel_highlight.get_dataset()
        if dataset is not None and rasterview.get_raster_data() is not dataset:
            return

        coord = self._pixel_highlight.get_pixel()

        with get_painter(widget) as painter:
            color = self._app_state.get_color_of('pixel-highlight')
            painter.setPen(QPen(color))

            # (ds_x, ds_y) is the coordinate within the data-set.
            ds_x = coord.x()
            ds_y = coord.y()

            # This is the size of individual data-set pixels in the display
            # coordinate system.
            scale = self.get_scale()

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

            elif reticle_type == PixelReticleType.SMALL_CROSS_BOX:
                painter.drawLine(screen_x - 15, screen_y, screen_x + 15, screen_y)
                painter.drawLine(screen_x, screen_y - 15, screen_x, screen_y + 15)

                # Draw a box around the highlighted pixel, but only if it's
                # larger than a certain scale.
                if scale >= 4:
                    # Compute the rectangle that will border the specified
                    # pixel.  Subtract 1 from the width and height to keep the
                    # rectangle from spilling into the neighboring pixel.
                    scaled = QRect(screen_x, screen_y, scale, scale)
                    painter.drawRect(scaled)

            else:
                raise ValueError(f'Unrecognized reticle-type {reticle_type}')
