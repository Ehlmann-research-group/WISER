import os
from enum import Enum
from typing import List, Optional, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .app_config import PixelReticleType
from .band_chooser import BandChooserDialog
from .dataset_chooser import DatasetChooser
from .roi_info_editor import ROIInfoEditor
from .rasterview import RasterView
from .util import add_toolbar_action, get_painter, make_filename
from .plugin_utils import add_plugin_context_menu_items

from wiser import plugins

from wiser.raster import roi_export
from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset import find_display_bands
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import SelectionType, SinglePixelSelection
from wiser.raster.spectra_export import export_roi_pixel_spectra
from wiser.raster.spectrum import ROIAverageSpectrum

from wiser.gui.app_state import ApplicationState

from .ui_roi import draw_roi, get_picked_roi_selections
from .ui_selection_rectangle import RectangleSelectionCreator, RectangleSelectionEditor
from .ui_selection_polygon import PolygonSelectionCreator, PolygonSelectionEditor
from .ui_selection_multi_pixel import MultiPixelSelectionCreator, MultiPixelSelectionEditor


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


class TiledRasterView(RasterView):
    '''
    This class subclasses the RasterView class for use in a tiled layout in the
    RasterPane.  It adds a toolbar and a position within the RasterPane.  The
    toolbar includes some basic tools, including a band chooser, a stretch
    builder, and a widget to choose the dataset being displayed.
    '''

    def __init__(self, rasterpane, position, app_state: ApplicationState, parent=None, forward=None):
        super().__init__(parent=parent, forward=forward, app_state=app_state)

        self._rasterpane = rasterpane
        self._position = position

        self._toolbar = QToolBar(
            self.tr('RasterView [{row}, {col}] Toolbar').format(row=position[0],
                                                                col=position[1]),
            parent=self)
        self._layout.setMenuBar(self._toolbar)

        self._toolbar.setIconSize(QSize(16, 16))
        # self._toolbar.setFloatable(False)
        # self._toolbar.setMovable(False)

        #====================
        # Dataset Chooser

        self._cbox_dataset_chooser = QComboBox()
        self._cbox_dataset_chooser.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._toolbar.addWidget(self._cbox_dataset_chooser)

        self._cbox_dataset_chooser.activated.connect(self._on_switch_to_dataset)

        # rv_dataset_chooser = DatasetChooser(None, self._app_state)
        # rv_toolbar.addWidget(rv_dataset_chooser)
        # TODO:  rv_dataset_chooser.triggered.connect(self._on_dataset_changed)

        #====================
        # Band Chooser

        self._act_band_chooser = add_toolbar_action(self._toolbar,
            ':/icons/choose-bands.svg', self.tr('Band chooser'), self)
        self._act_band_chooser.triggered.connect(
            lambda checked=False : self._rasterpane._on_band_chooser(
                checked, rasterview_pos=self._position))

        #====================
        # Stretch Builder

        self._act_stretch_builder = add_toolbar_action(self._toolbar,
            ':/icons/stretch-builder.svg', self.tr('Stretch builder'), self)
        self._act_stretch_builder.triggered.connect(
            lambda checked=False : self._rasterpane._on_stretch_builder(
                checked, rasterview_pos=self._position))

        self.update_toolbar_state()


    def show_toolbar(self):
        ''' Shows the toolbar on this raster-view. '''
        self._toolbar.show()

    def hide_toolbar(self):
        ''' Hides the toolbar on this raster-view. '''
        self._toolbar.hide()

    def update_toolbar_state(self):
        '''
        Updates the toolbar's state to match the current application state.
        This includes synchronizing the list of datasets in the dataset-chooser
        with the datasets in the application, and also updating the button
        enabled-state based on what tasks should be available to users.
        '''
        app_state = self._rasterpane.get_app_state()

        num_datasets = app_state.num_datasets()

        self._act_band_chooser.setEnabled(num_datasets > 0)
        self._act_stretch_builder.setEnabled(num_datasets > 0)

        current_index = self._cbox_dataset_chooser.currentIndex()
        current_ds_id = None
        if current_index != -1:
            current_ds_id = self._cbox_dataset_chooser.itemData(current_index)
        else:
            # This occurs initially, when the combobox is empty and has no
            # selection.  Make sure the "(no data)" option is selected by the
            # end of this process.
            current_index = 0
            current_ds_id = -1

        # print(f'update_toolbar_state(position={self._position}):  current_index = {current_index}, current_ds_id = {current_ds_id}')

        new_index = None
        self._cbox_dataset_chooser.clear()

        if num_datasets > 0:
            for (index, dataset) in enumerate(app_state.get_datasets()):
                id = dataset.get_id()
                name = dataset.get_name()

                self._cbox_dataset_chooser.addItem(name, id)
                if dataset.get_id() == current_ds_id:
                    new_index = index

            self._cbox_dataset_chooser.insertSeparator(num_datasets)
            self._cbox_dataset_chooser.addItem(self.tr('(no data)'), -1)
            if current_ds_id == -1:
                new_index = self._cbox_dataset_chooser.count() - 1
        else:
            # No datasets yet
            self._cbox_dataset_chooser.addItem(self.tr('(no data)'), -1)
            if current_ds_id == -1:
                new_index = 0

        # print(f'update_toolbar_state(position={self._position}):  new_index = {new_index}')

        if new_index is None:
            if num_datasets > 0:
                new_index = min(current_index, num_datasets - 1)
            else:
                new_index = 0

        self._cbox_dataset_chooser.setCurrentIndex(new_index)


    def _on_switch_to_dataset(self, index: int):
        '''
        This function handles "activated" events from the dataset-chooser
        widget.
        '''
        # print(f'_on_switch_to_dataset(position={self._position}):  index = {index}')

        ds_id = self._cbox_dataset_chooser.itemData(index)

        dataset = None
        if ds_id != -1:
            app_state = self._rasterpane.get_app_state()
            dataset = app_state.get_dataset(ds_id)

        self._rasterpane.show_dataset(dataset, self._position)


    def set_raster_data(self, raster_data, display_bands, stretches=None):
        '''
        Override the base-class implementation to also update the toolbar's
        dataset-chooser to match the dataset being displayed.
        '''
        # Let the base implementation do its thing first.
        super().set_raster_data(raster_data, display_bands, stretches=stretches)

        # Update the dataset-chooser to match this raster data.

        # print(f'set_raster_data(position={self._position})')

        if raster_data is None:
            index = self._cbox_dataset_chooser.findData(-1)
            assert index != -1, f'Missing the (no data) option!'

        else:
            ds_id = raster_data.get_id()
            index = self._cbox_dataset_chooser.findData(ds_id)
            assert index != -1, f'Tried to display an unrecognized dataset:  {ds_id}'

        self._cbox_dataset_chooser.setCurrentIndex(index)


class RasterPane(QWidget):
    '''
    This widget provides a raster-view with an associated toolbar.
    '''

    # Signal:  the raster pane was shown or hidden
    #   - The value is True for "visible", False for "invisible".
    visibility_change = Signal(bool)

    # Signal:  the number of views being shown in the rasterpane was updated
    #   - The value is a 2-tuple specifying the number of rows and columns in
    #     the raster pane's view layout.
    # TODO(donnie):  Maybe rename to view_layout_change ?  A bit confusing name
    #     with viewport_change signal below.
    views_changed = Signal(tuple)

    # Signal:  the display bands in this view were changed
    #   - The int is the numeric ID of the dataset whose display bands are
    #     changing
    #   - The tuple is either a 1-tuple or 3-tuple specifying the display bands
    #   - The object is an optional string colormap name, when the tuple has 1
    #     element; otherwise it is None (NOTE:  declaring the value as a str
    #     will cause Qt to convert a value of None to an empty string, which is
    #     not what we want)
    #   - The Boolean argument is True for "global change," or False for "this
    #     view only"
    display_bands_change = Signal(int, tuple, object, bool)


    # Signal:  for when the user selects a raster pixel.  The signal reports the
    # (row, col) coordinates of the raster-view that was clicked in, and also
    # the coordinates of the pixel in the raster image.
    click_pixel = Signal(tuple, QPoint)


    # Signal:  a Region of Interest's selection was added/removed/changed.  The
    # ROI (arg1) and the selection (arg2) are passed as arguments.  Note that
    # if a selection is deleted, then the second argument will be None.
    roi_selection_changed = Signal(object, object)


    # Signal:  one or more raster-views changed their display viewport.  The
    # signal reports one optional 2-tuple with the raster-view's position in the
    # pane, or the argument will be None if all raster-views changed their
    # display viewport.
    # TODO(donnie):  Maybe rename to raster_viewport_change ?  A bit confusing
    #     name with viewport_change signal below.
    viewport_change = Signal(tuple)


    def __init__(self, app_state: ApplicationState, parent=None, size_hint=None,
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
        self._app_state.mainview_dataset_changed.connect(self._on_dataset_added)
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
            ':/icons/choose-bands.svg', self.tr('Band chooser'), self)
        # TODO(donnie):  If we just pop up a widget...
        # self._band_chooser = BandChooser(self._app_state)
        # toolbar.addWidget(self._band_chooser)
        self._act_band_chooser.triggered.connect(self._on_band_chooser)
        self._act_band_chooser.setEnabled(False)


    def _init_zoom_tools(self):
        '''
        Initialize zoom toolbar buttons.
        '''

        # Zoom In
        self._act_zoom_in = add_toolbar_action(self._toolbar,
            ':/icons/zoom-in.svg', self.tr('Zoom in'), self, QKeySequence.ZoomIn)
        self._act_zoom_in.triggered.connect(self._on_zoom_in)

        # Zoom Out
        self._act_zoom_out = add_toolbar_action(self._toolbar,
            ':/icons/zoom-out.svg', self.tr('Zoom out'), self, QKeySequence.ZoomOut)
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
        Initialize the Region of Interest / Selection tools
        '''

        # Button to add new Regions of Interest.  This will pop up a dialog for
        # the user to enter a name and some metadata for the ROI.
        self._act_create_roi = add_toolbar_action(self._toolbar,
            ':/icons/add-roi.svg', self.tr('Add new Region of Interest'), self,
            QKeySequence.ZoomIn)
        self._act_create_roi.triggered.connect(self._on_create_roi)

        # Drop-down combobox for ROIs, so that new selections will go into the
        # currently highlighted ROI.
        self._cbox_current_roi = QComboBox()
        self._cbox_current_roi.setEditable(False)
        self._cbox_current_roi.setInsertPolicy(QComboBox.NoInsert)
        self._cbox_current_roi.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self._act_cbox_current_roi = self._toolbar.addWidget(self._cbox_current_roi)

        # The selection-tools chooser is a drop down menu of selection tools.
        # First, populate the menu of tools, then create the chooser button.

        chooser = QToolButton()
        chooser.setIcon(QIcon(':/icons/select.svg'))
        chooser.setToolTip(self.tr('Add selection to current ROI'))

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

        # act = menu.addAction(self.tr('Predicate selection'))
        # act.setData(SelectionType.PREDICATE)

        self._toolbar.addWidget(chooser)

        chooser.triggered.connect(self._on_add_selection_to_roi)
        self._roi_selection_chooser = chooser

        self._populate_roi_combobox()

        # Register for some ROI-specific events
        self._app_state.roi_added.connect(self._on_roi_added)
        self._app_state.roi_removed.connect(self._on_roi_removed)



    def _init_rasterviews(self, num_views: Tuple[int, int]=(1, 1)):
        '''
        Initialize the raster-pane to have MxN raster-views displayed in the
        pane.  The default is to have only one raster-view showing in the pane.
        '''

        if num_views[0] < 1 or num_views[1] < 1:
            raise ValueError(f'Minimum number of raster-view rows/cols is 1, got {num_views}')

        # If the current raster-view layout is the same as the requested
        # raster-view layout, ignore.
        if self._num_views == num_views:
            return

        forward = {
            # 'resizeEvent'       : self._onRasterResize,
            'mousePressEvent'   : self._onRasterMousePress,
            'mouseReleaseEvent' : self._onRasterMouseRelease,
            'mouseMoveEvent'    : self._onRasterMouseMove,
            'keyPressEvent'     : self._onRasterKeyPress,
            'keyReleaseEvent'   : self._onRasterKeyRelease,
            'contextMenuEvent'  : self._onRasterContextMenu,
            'paintEvent'        : self._afterRasterPaint,
            'scrollContentsBy'  : self._afterRasterScroll,
        }

        if len(self._rasterviews) != 0:
            # There are existing raster-views.  Clean up the ones that are now
            # "out of bounds" for the new layout.
            for (position, rasterview) in list(self._rasterviews.items()):
                (row, col) = position
                if row < num_views[0] and col < num_views[1]:
                    # Keep this RasterView and its frame.
                    continue

                # Remove the rasterview-frame from the layout, and close it.
                self._rasterview_layout.removeWidget(rasterview)
                rasterview.close()

                # Remove the entry from the rasterviews collection
                del self._rasterviews[position]

        # Create new raster-views as specified by the arguments

        multiviews = (num_views != (1, 1))

        (rows, cols) = num_views
        for row in range(rows):
            for col in range(cols):
                position = (row, col)

                rasterview = self._rasterviews.get(position)
                if rasterview is None:
                    rasterview = TiledRasterView(self, position, self._app_state, forward=forward)
                    rasterview.setContextMenuPolicy(Qt.DefaultContextMenu)

                    self._rasterviews[position] = rasterview
                    self._rasterview_layout.addWidget(rasterview, row, col)

                if multiviews:
                    rasterview.show_toolbar()
                else:
                    rasterview.hide_toolbar()

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


    def get_app_state(self):
        return self._app_state


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


    def get_rasterview(self, rasterview_pos=(0, 0)) -> RasterView:
        '''
        Returns the raster-view at the specified (row, column) position in the
        raster-pane.  The default (row, column) value is (0, 0), which can be
        used by panes that will only ever have one raster-view.
        '''
        return self._rasterviews[rasterview_pos]


    def get_rasterview_with_data(self) -> Optional[RasterView]:
        '''
        If there is a rasterview with data. This function returns the first it finds. It
        starts from (0, 0), then goes across columns then down rows.

        An example traversal for a 2x2 grid would be (0, 0) -> (0, 1) -> (1, 0) -> (1, 1)
        '''
        rows, cols = self.get_num_views()

        for col in range(cols):
            for row in range(rows):
                rv = self.get_rasterview((row, col))
                if rv.get_raster_data() is not None:
                    return rv

        return None


    def update_all_rasterviews(self):
        '''
        Cause all rasterviews in this pane to repaint themselves.
        '''
        for rv in self._rasterviews.values():
            rv.update()


    def _update_rasterview_toolbars(self):
        '''
        An internal helper function to update the toolbars of all tiled
        rasterviews being displayed.
        '''
        # Update the toolbar state of all rasterviews
        for rasterview in self._rasterviews.values():
            rasterview.update_toolbar_state()


    def get_all_visible_regions(self) -> List:
        '''
        Returns a list of all visible regions
        '''
        # Get the list of visible regions, but filter out empty regions.
        regions = []
        for rv in self._rasterviews.values():
            region = rv.get_visible_region()
            if region is not None:
                regions.append(region)

        return regions


    def get_app_state(self):
        return self._app_state


    def get_visible_datasets(self) -> List[RasterDataSet]:
        '''
        Returns all of the datasets that are displayed in all of the 
        RasterViews under this RasterPane
        '''
        visible_ds = []
        for rasterview in self._rasterviews.values():
            if rasterview._raster_data is not None:
                visible_ds.append(rasterview._raster_data)
        return visible_ds


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
        for rv in self._rasterviews.values():
            rv.scale_image(scale)


    def resizeEvent(self, event):
        '''
        Override the QtWidget resizeEvent() virtual method to fire an event that
        the visible region of the raster-view has changed.
        '''
        self._emit_viewport_change()


    # def _onRasterResize(self, rasterview, resize_event):
    #     self._emit_viewport_change()


    def _onRasterMousePress(self, rasterview, mouse_event):
        if self._has_delegate_for_rasterview(rasterview):
            done = self._task_delegate.on_mouse_press(mouse_event)
            self._update_delegate(done)

    def _onRasterMouseMove(self, rasterview, mouse_event):
        if self._has_delegate_for_rasterview(rasterview, user_input=False):
            done = self._task_delegate.on_mouse_move(mouse_event)
            self._update_delegate(done)

    def _onRasterMouseRelease(self, rasterview, mouse_event):
        '''
        When the display image is clicked on, this method gets invoked, and it
        translates the click event's coordinates into the location on the
        raster data set.
        '''
        if not isinstance(mouse_event, QMouseEvent):
            return

        # print(f'MouseEvent at pos={mouse_event.pos()}, localPos={mouse_event.localPos()}')

        if self._has_delegate_for_rasterview(rasterview):
            done = self._task_delegate.on_mouse_release(mouse_event)
            self._update_delegate(done)

        else:
            # Only handle left mouse buttons; right mouse clicks will result in
            # a context menu being generated.
            if mouse_event.button() == Qt.LeftButton:
                # Map the coordinate of the mouse-event to the actual raster-image
                # pixel that was clicked, then emit a signal.
                r_coord = rasterview.image_coord_to_raster_coord(mouse_event.localPos())
                if rasterview.is_raster_coord_in_bounds(r_coord):
                    rasterview_pos = self._get_rasterview_position(rasterview)
                    self.click_pixel.emit(rasterview_pos, r_coord)


    def _onRasterKeyPress(self, rasterview, key_event):
        if self._has_delegate_for_rasterview(rasterview):
            done = self._task_delegate.on_key_press(key_event)
            self._update_delegate(done)

    def _onRasterKeyRelease(self, rasterview, key_event):
        if self._has_delegate_for_rasterview(rasterview):
            done = self._task_delegate.on_key_release(key_event)
            self._update_delegate(done)

    def _onRasterContextMenu(self, rasterview, context_menu_event):
        # print(f'ContextMenuEvent at {context_menu_event.pos()}')
        menu = QMenu(self)

        self._build_context_menu(menu, rasterview, context_menu_event)

        # Only show the context menu if we have stuff to show.
        if not menu.isEmpty():
            menu.exec_(context_menu_event.globalPos())


    def _afterRasterScroll(self, widget, dx, dy):
        '''
        This function is called when the scroll-area moves around.  Fire an
        event that the visible region of the raster-view has changed.
        '''
        self._emit_viewport_change()


    def _build_context_menu(self, menu, rasterview, context_menu_event):
        if rasterview.get_raster_data() is None:
            return

        # Add any context-menu items that are independent of location
        self._context_menu_add_global_items(menu, rasterview)

        # Add any context-menu items that are specific to this location
        self._context_menu_add_local_items(menu, rasterview, context_menu_event)

        # Add any context-menu items that should go at the end
        self._context_menu_add_end_items(menu, rasterview)


    def _context_menu_add_global_items(self, menu, rasterview):
        '''
        This helper function adds "global" items to the context menu, that is,
        items that aren't specific to the location clicked in the window.

        The base implementation does nothing.
        '''
        # Let subclasses do things if they want.
        pass


    def _context_menu_add_local_items(self, menu, rasterview, context_menu_event):
        '''
        This helper function adds "local" items to the context menu, that is,
        items that are specific to the location clicked in the window.  The
        data-set coordinate of the click is passed into the function.

        The base implementation picks against regions of interest, to add
        various options for users to employ.
        '''
        # menu.addSeparator()

        # Calculate the coordinate of the click in dataset coordinates
        ds_coord = rasterview.image_coord_to_raster_coord(context_menu_event.pos())

        # TODO(donnie):  Set up handler for the action
        # act = menu.addAction(self.tr('Annotate location'))

        # Add plugin menu items
        add_plugin_context_menu_items(self._app_state,
            plugins.ContextMenuType.DATASET_PICK, menu,
            dataset=rasterview.get_raster_data(),
            display_bands=rasterview.get_display_bands(),
            ds_coord=ds_coord.toTuple())

        # Find Regions of Interest that include the click location.  This is a
        # complicated thing to do, since a ROI can consist of multiple
        # selections.  So, this variable is a list of 2-tuples, where each pair
        # is (the picked ROI, a list of the indexes of selections picked in the
        # ROI).  Clear as mud.
        picked_rois = []
        for roi in self._app_state.get_rois():
            picked_sels = get_picked_roi_selections(roi, ds_coord)
            if len(picked_sels) > 0:
                picked_rois.append( (roi, picked_sels) )

        if len(picked_rois) > 0:
            # At least one region of interest was picked
            if not menu.isEmpty():
                menu.addSeparator()

            for (roi, picked_sels) in picked_rois:
                roi_menu = menu.addMenu(roi.get_name())

                act = roi_menu.addAction(self.tr('Edit ROI information...'))
                act.triggered.connect(lambda checked : self._on_edit_roi_info(roi=roi))

                act = roi_menu.addAction(self.tr('Show ROI average spectrum'))
                act.triggered.connect(
                    lambda checked : self._on_show_roi_avg_spectrum(roi=roi, rasterview=rasterview))

                roi_menu.addSeparator()

                act = roi_menu.addAction(self.tr('Export ROI...'))
                act.triggered.connect(
                    lambda checked : self._on_export_region_of_interest(roi=roi, rasterview=rasterview))

                act = roi_menu.addAction(self.tr('Export all spectra in ROI...'))
                act.triggered.connect(
                    lambda checked : self._on_export_roi_pixel_spectra(roi=roi, rasterview=rasterview))

                # Add plugin menu items
                add_plugin_context_menu_items(self._app_state,
                    plugins.ContextMenuType.ROI_PICK, menu,
                    dataset=rasterview.get_raster_data(),
                    display_bands=rasterview.get_display_bands(),
                    roi=roi,
                    ds_coord=ds_coord.toTuple())

                for sel_index in picked_sels:
                    roi_menu.addSeparator()
                    act = roi_menu.addAction(self.tr(f'Edit selection {sel_index} geometry'))
                    act.triggered.connect(
                        lambda checked : self._on_edit_roi_selection_geometry(
                            roi=roi, sel_index=sel_index, rasterview=rasterview))

                    act = roi_menu.addAction(self.tr(f'Delete selection {sel_index} from ROI...'))
                    act.triggered.connect(
                        lambda checked : self._on_delete_roi_selection_geometry(
                            roi=roi, sel_index=sel_index))

                roi_menu.addSeparator()

                act = roi_menu.addAction(self.tr('Delete Region of Interest...'))
                act.triggered.connect(lambda checked : self._on_delete_roi(roi=roi))


    def _context_menu_add_end_items(self, menu, rasterview):
        '''
        This helper function adds items to the context menu that should be at
        the end of the menu.

        The base implementation does nothing.
        '''
        # Let subclasses do things if they want.
        pass


    def _afterRasterScroll(self, rasterview, dx, dy):
        '''
        This function is called when the raster-view's scrollbars are moved.

        It fires an event that the visible region of the raster-view has
        changed.

        If multiple raster-views are active, and scrolling is linked, this also
        propagates the scroll changes to the other raster-views.
        '''
        self._emit_viewport_change(self._get_rasterview_position(rasterview))


    def _has_delegate_for_rasterview(self, rasterview: RasterView,
                                     user_input: bool = True) -> bool:
        '''
        This helper function encapsulates the logic for checking whether a
        raster-view's events can be handled by the current task delegate (if a
        delegate is even active).  Task delegates record the raster-view they
        are handling events from, so it is straightforward to check if the
        delegate's raster-view matches the passed-in raster-view.

        The user_input flag indicates whether this function was called because
        of a "user-input event"; i.e. an intentional interaction with the
        raster-view.  User input is the only thing that should cause the
        task-delegate's raster-view to be set.  (Since the user interaction must
        be intentional, "user input" means actual key presses/releases and mouse
        button presses/releases.  Simple mouse motion doesn't indicate an
        intentional interaction, and is therefore excluded.)
        '''

        # If we don't even have a task delegate, return False.
        if self._task_delegate is None:
            return False

        # Retrieve the delegate's raster-view.  If the delegate's raster-view is
        # not set, and this call was made because of user input, set the
        # delegate's raster-view to the passed-in raster-view.
        td_rasterview = self._task_delegate.get_rasterview()
        if (td_rasterview is None) and user_input:
            self._task_delegate.set_rasterview(rasterview)
            td_rasterview = rasterview

        # If the task-delegate is taking input from a different raster-view
        # then don't forward this event.
        return (td_rasterview is rasterview)


    def _update_delegate(self, done: bool) -> None:
        '''
        This helper function handles common post-event operations with the task
        delegate and its associated raster-view.

        If the task delegate reported that it's done, the delegate's finish()
        function is called, and the delegate is cleared.

        The delegate's raster-view UI is always updated to reflect any info the
        delegate has to draw, or the fact that the delegate is going away.
        '''
        # Get out the raster-view first, since the delegate may be finished and
        # about to go away.
        td_rasterview = self._task_delegate.get_rasterview()

        if done:
            # selection = self._creator.get_selection()
            # print(f'TODO:  Store selection {selection} on application state')
            # TODO(donnie):  How to handle completion of task delegates???
            self._task_delegate.finish()
            self._task_delegate = None

        self.update_all_rasterviews()


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


    def show_dataset(self, dataset: RasterDataSet, rasterview_pos=(0, 0)):
        '''
        Sets the dataset being displayed in the specified view of the raster
        pane.
        '''
        rasterview = self.get_rasterview(rasterview_pos)

        # If the rasterview is already showing the specified dataset, skip!
        if rasterview.get_raster_data() is dataset:
            return

        bands = None
        stretches = None
        if dataset is not None:
            ds_id = dataset.get_id()
            bands = self._display_bands[ds_id]
            stretches = self._app_state.get_stretches(ds_id, bands)

        rasterview.set_raster_data(dataset, bands, stretches)
        if dataset is not None and self._num_views == (1, 1):
            self._dataset_chooser.check_dataset(dataset.get_id())

        # This is a check to see if this rasterpane is MainView
        if hasattr(self, "_link_view_scrolling"):
            self.on_rasterview_dataset_changed()


    def is_showing_dataset(self, dataset) -> Optional[Tuple[int, int]]:
        '''
        If some rasterview in this pane is displaying the dataset, this function
        returns the 2-tuple position of the rasterview in the pane.  If no
        rasterview is displaying the dataset then this function returns None.
        '''

        for r in range(self._num_views[0]):
            for c in range(self._num_views[1]):
                pos = (r, c)
                rv_dataset = self._rasterviews[pos].get_raster_data()

                if (dataset is None and rv_dataset is None):
                    return pos

                # At this point, at least one dataset is not None

                if (dataset is None or rv_dataset is None):
                    continue

                if dataset.get_id() == rv_dataset.get_id():
                    return pos

        return None

    def set_display_bands(self, ds_id: int, bands: Tuple, colormap: Optional[str] = None):
        # TODO(donnie):  Verify the dataset ID?

        if len(bands) not in [1, 3]:
            raise ValueError(f'bands must be either a 1-tuple or 3-tuple; got {bands}')

        # print(f'Display-band information:  {self._display_bands}')

        self._display_bands[ds_id] = bands

        # If the specified data set is the one currently being displayed, update
        # the UI display.
        for rv in self._rasterviews.values():
            rv_dataset = rv.get_raster_data()
            if rv_dataset is not None and ds_id == rv_dataset.get_id():
                # Get the stretches at the same time, so that we only update the
                # raster-view once.
                stretches = self._app_state.get_stretches(ds_id, bands)

                rv.set_display_bands(bands, stretches=stretches, colormap=colormap)


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
        for rv in self._rasterviews.values():
            visible = rv.get_visible_region()
            if visible is None or viewport is None or isinstance(viewport, list):
                rv.update()
                continue

            if not visible.contains(viewport):
                center = viewport.center()
                rv.make_point_visible(center.x(), center.y())

            # Repaint raster-view
            rv.update()


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
        self._view_dataset(ds_id)
    
    def _view_dataset(self, ds_id):
        '''
        This function changes the current raster pane's view to that of the dataset
        from ds_id.
        It records the initial display bands to use for the dataset.  Also, if
        this is the first dataset loaded, the function shows it in all
        rasterviews.
        '''
        dataset = self._app_state.get_dataset(ds_id)
        bands = find_display_bands(dataset)
        self._display_bands[ds_id] = bands

        self._update_rasterview_toolbars()

        # Search through the current set of RasterViews; if one is currently not
        # showing data, display the new dataset in that view.  If all are
        # showing data, just show it in the top-left view.

        display_pos = (0, 0)

        positions = [(r, c) for r in range(self._num_views[0])
                            for c in range(self._num_views[1])]
        for pos in positions:
            if self._rasterviews[pos].get_raster_data() is None:
                display_pos = pos
                break

        self.show_dataset(dataset, display_pos)

        # Always do this when we add a data set
        self._act_band_chooser.setEnabled(True)
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

        for rv in self._rasterviews.values():
            rv_ds = rv.get_raster_data()
            if rv_ds is not None and rv_ds.get_id() == ds_id:
                rv.set_raster_data(None, None)

        self._update_rasterview_toolbars()

        if self._app_state.num_datasets() == 0:
            self._act_band_chooser.setEnabled(False)


    def _on_dataset_changed(self, act):
        (rasterview_pos, ds_id) = act.data()
        dataset = self._app_state.get_dataset(ds_id)
        self.show_dataset(dataset, rasterview_pos)


    def _on_band_chooser(self, checked=False, rasterview_pos=(0,0)):
        # print(f'on_band_chooser invoked for position {rasterview_pos}')

        rasterview = self.get_rasterview(rasterview_pos)
        dataset = rasterview.get_raster_data()
        display_bands = rasterview.get_display_bands()
        colormap = rasterview.get_colormap()

        dialog = BandChooserDialog(self._app_state, dataset, display_bands,
            colormap=colormap, parent=self)
        dialog.setModal(True)

        if dialog.exec_() == QDialog.Accepted:
            bands = dialog.get_display_bands()
            is_global = dialog.apply_globally()
            colormap = dialog.get_colormap_name()

            self.display_bands_change.emit(dataset.get_id(), bands, colormap, is_global)

            # Only update our display bands if the change was not global, since
            # if it was, the main application controller will change everybody's
            # display bands.
            if not is_global:
                self.set_display_bands(dataset.get_id(), bands, colormap=colormap)


    def _on_stretch_builder(self, checked=False, rasterview_pos=(0, 0)):
        ''' Show the Stretch Builder on behalf of the specified raster-view. '''

        # print(f'on_stretch_builder invoked for position {rasterview_pos}')

        if self._stretch_builder is None:
            self._stretch_builder = StretchBuilderDialog(parent=self, app_state=self._app_state)

        rasterview = self.get_rasterview(rasterview_pos)
        self._stretch_builder.show(rasterview.get_raster_data(),
                                   rasterview.get_display_bands(),
                                   rasterview.get_stretches())


    def _on_stretch_changed(self, ds_id, bands):
        # Iterate through all rasterviews.  If any is displaying the dataset
        # that changed stretch, update its stretches.
        for rv in self._rasterviews.values():
            dataset = rv.get_raster_data()
            if dataset is None:
                continue

            if dataset.get_id() == ds_id:
                bands = rv.get_display_bands()
                stretches = self._app_state.get_stretches(ds_id, bands)
                rv.set_stretches(stretches)

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


    def _on_create_roi(self, act):
        '''
        Pop up a dialog allowing the user to create a new Region of Interest,
        then make that ROI the current ROI in the combo-box.
        '''
        dialog = ROIInfoEditor(self._app_state, parent=self)
        result = dialog.exec()
        if result == QDialog.Accepted:
            roi = RegionOfInterest()
            dialog.store_values(roi)
            self._app_state.add_roi(roi)


    def _on_roi_added(self, roi):
        self._populate_roi_combobox(choose_roi_id=roi.get_id())

    def _on_roi_removed(self, roi):
        self._populate_roi_combobox()
        # Force a repaint of all raster-views.
        self.update_all_rasterviews()

    def _populate_roi_combobox(self, choose_roi_id=None):
        if choose_roi_id is None:
            choose_roi_id = self._cbox_current_roi.currentData()
        self._cbox_current_roi.clear()

        rois = self._app_state.get_rois()
        if len(rois) > 0:
            for roi in self._app_state.get_rois():
                self._cbox_current_roi.addItem(roi.get_name(), roi.get_id())

            self._roi_selection_chooser.setEnabled(True)

        else:
            self._cbox_current_roi.addItem(self.tr('(no ROIs)'), None)
            self._roi_selection_chooser.setEnabled(False)

        # Figure out which item was previously selected.  If it is no longer
        # present, just select the first item.
        i = self._cbox_current_roi.findData(choose_roi_id)
        if i == -1:
            i = 0
        self._cbox_current_roi.setCurrentIndex(i)

    def get_current_roi(self) -> Optional[RegionOfInterest]:
        roi_id = self._cbox_current_roi.currentData()
        roi = None
        if roi_id is not None:
            roi = self._app_state.get_roi(id=roi_id)

        return roi


    def _on_edit_roi_info(self, roi):
        '''
        Pop up a dialog allowing the user to create a new Region of Interest,
        then make that ROI the current ROI in the combo-box.
        '''
        dialog = ROIInfoEditor(self._app_state, parent=self)
        dialog.configure_ui(roi)
        result = dialog.exec()
        if result == QDialog.Accepted:
            dialog.store_values(roi)
            # TODO(donnie):  Need to notify everyone that the ROI has been edited

            self._populate_roi_combobox()


    def _on_add_selection_to_roi(self, act):
        '''
        This helper function initiates the creation of a selection, which is
        then added to a Region of Interest.
        '''

        # TODO(donnie):  What to do if the task-delegate already exists?!

        selection_type = act.data()

        if selection_type == SelectionType.RECTANGLE:
            self._task_delegate = RectangleSelectionCreator(self)

        elif selection_type == SelectionType.POLYGON:
            self._task_delegate = PolygonSelectionCreator(self)

        elif selection_type == SelectionType.MULTI_PIXEL:
            self._task_delegate = MultiPixelSelectionCreator(self)

        elif selection_type == SelectionType.PREDICATE:
            ok, pred_text = QInputDialog.getText(self,
                self.tr('Create predicate selection'),
                self.tr('Enter a predicate specifying what pixels to select.'))

            if ok and pred_text:
                print(f'TODO:  Create selection from predicate {pred_text}')

        else:
            QMessageBox.warning(self, self.tr('Unsupported Feature'),
                f'WISER does not yet support selections of type {selection_type}')


    def _on_edit_roi_selection_geometry(self, roi, sel_index, rasterview) -> None:
        sel = roi.get_selections()[sel_index]
        selection_type = sel.get_type()

        if selection_type == SelectionType.RECTANGLE:
            self._task_delegate = \
                RectangleSelectionEditor(roi, sel, self, rasterview)

        elif selection_type == SelectionType.POLYGON:
            self._task_delegate = \
                PolygonSelectionEditor(roi, sel, self, rasterview)

        elif selection_type == SelectionType.MULTI_PIXEL:
            self._task_delegate = \
                MultiPixelSelectionEditor(roi, sel, self, rasterview)

        else:
            QMessageBox.warning(self, self.tr('Unsupported Feature'),
                f'WISER does not yet support editing selections of type {selection_type}')


    def _on_delete_roi_selection_geometry(self, roi, sel_index) -> None:
        name = roi.get_name()
        sel = roi.get_selections()[sel_index]

        result = QMessageBox.question(self,
            self.tr('Delete selection {0} from ROI "{1}"').format(sel_index, name),
            self.tr('Are you sure you wish to delete selection {0} ' +
                    'from the ROI "{1}"?  This cannot be undone.').format(sel_index, name))

        if result == QMessageBox.Yes:
            roi.del_selection(sel_index)

            # Signal that the ROI changed, so that everyone can be notified.
            self.roi_selection_changed.emit(roi, None)

            # Report to the user that the ROI was deleted.
            self._app_state.show_status_text(
                self.tr('Deleted selection {0} from Region of Interest "{1}"')
                    .format(sel_index, name), 5)


    def _on_show_roi_avg_spectrum(self, roi: RegionOfInterest, rasterview: RasterView):
        # TODO(donnie):  Need to get the default average mode from somewhere
        spectrum = ROIAverageSpectrum(rasterview.get_raster_data(), roi)
        self._app_state.set_active_spectrum(spectrum)


    def _on_export_region_of_interest(self, roi: RegionOfInterest, rasterview: RasterView) -> None:
        selected = QFileDialog.getSaveFileName(self,
            self.tr('Export Region of Interest:  {0}').format(roi.get_name()),
            self._app_state.get_current_dir(),
            self.tr('GeoJSON files (*.geojson);;All Files (*)'))

        if selected[0]:
            roi_export.export_roi_to_geojson_file(roi, selected[0])


    def _on_export_roi_pixel_spectra(self, roi: RegionOfInterest, rasterview: RasterView) -> None:

        # If the ROI has a lot of pixels, confirm with the user.
        pixels = roi.get_all_pixels()
        if len(pixels) > 200:
            result = QMessageBox.question(self,
                self.tr('Export ROI Pixel Spectra'),
                self.tr('This ROI has {0} pixels.  Are you sure you wish to ' +
                        'output the spectra of all pixels?').format(len(pixels)))

            if result == QMessageBox.No:
                return

        # Build up a candidate filename.
        filename = f'{make_filename(roi.get_name())}.txt'
        filename = os.path.join(self._app_state.get_current_dir(), filename)

        # Ask the user for a save filename.
        (filename, type) = QFileDialog.getSaveFileName(self,
            self.tr('Save all spectra from ROI {0}').format(roi.get_name()),
            filename,
            self.tr('Text files (*.txt);;All files (*)'))

        if len(filename) == 0:
            # User canceled out of the operation.
            return

        # Update app-state's current directory from the user-selected path
        self._app_state.update_cwd_from_path(filename)

        # Export the spectra of all pixels in the ROI
        export_roi_pixel_spectra(filename, rasterview.get_raster_data(), roi)


    def _on_delete_roi(self, roi: RegionOfInterest) -> None:
        name = roi.get_name()
        result = QMessageBox.question(self,
            self.tr('Delete ROI "{0}"').format(name),
            self.tr('Are you sure you wish to delete the ROI ' +
                    '"{0}"?  This cannot be undone.').format(name))

        if result == QMessageBox.Yes:
            self._app_state.remove_roi(roi.get_id())

            # Report to the user that the ROI was deleted.
            self._app_state.show_status_text(
                self.tr('Deleted Region of Interest "{0}"').format(name), 5)


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

        Or, to leverage automatic resource management, do this:

            with get_painter(widget) as painter:
                ... # Draw stuff

        The paint-event that prompted the call to this method is provided, so
        that data to draw may be clipped to the specified rectangle.
        '''

        # Draw regions of interest
        self._draw_regions_of_interest(rasterview, widget, paint_event)

        # Draw the viewport highlight, if there is one
        self._draw_viewport_highlight(rasterview, widget, paint_event)

        # Draw the pixel highlight, if there is one
        self._draw_pixel_highlight(rasterview, widget, paint_event)

        if self._has_delegate_for_rasterview(rasterview, user_input=False):
            # Let the task-delegate draw any state it needs to draw.
            with get_painter(widget) as painter:
                self._task_delegate.draw_state(painter)


    def _draw_regions_of_interest(self, rasterview, widget, paint_event):
        # active_roi = self._app_state.get_active_roi()
        with get_painter(widget) as painter:
            for roi in self._app_state.get_rois():
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
            color = self._app_state.get_config('raster.viewport_highlight_color')
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
            color = self._app_state.get_config('raster.pixel_cursor_color')
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

            reticle_type = self._app_state.get_config(
                'raster.pixel_cursor_type',
                default=PixelReticleType.SMALL_CROSS,
                as_type=lambda s : PixelReticleType[s])

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
                    scaled = QRect(ds_x * scale, ds_y * scale, scale, scale)
                    painter.drawRect(scaled)

            else:
                raise ValueError(f'Unrecognized pixel cursor type {reticle_type}')
