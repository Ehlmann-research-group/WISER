import os

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import gui.resources

from .toolbarmenu import ToolbarMenu
from .rasterpane import RasterPane
from .rasterview import RasterView
from .stretch_builder import StretchBuilderDialog
from .util import add_toolbar_action


class MainViewWidget(RasterPane):
    '''
    This widget provides the main raster-data view in the user interface.
    '''


    def __init__(self, app_state, parent=None):
        super().__init__(app_state=app_state, parent=parent, embed_toolbar=False,
            max_zoom_scale=16, zoom_options=[0.25, 0.5, 0.75, 1, 2, 4, 8, 16],
            initial_zoom=1)

        self._stretch_builder = StretchBuilderDialog(parent=self)
        self._link_view_scrolling = False


    def _init_toolbar(self):
        '''
        The main view initializes dataset tools, view tools, zoom tools, and
        selection tools.
        '''
        self._init_dataset_tools()
        self._toolbar.addSeparator()
        self._init_view_tools()
        self._toolbar.addSeparator()
        self._init_zoom_tools()
        self._toolbar.addSeparator()
        self._init_select_tools()


    def _init_dataset_tools(self):
        '''
        Override the default dataset-tools function to add a stretch-builder
        button to the dataset-tools part of the toolbar.
        '''
        super()._init_dataset_tools()

        self._act_stretch_builder = add_toolbar_action(self._toolbar,
            ':/icons/stretch-builder.svg', self.tr('Stretch builder'), self)
        self._act_stretch_builder.triggered.connect(self._on_stretch_builder)


    def _init_view_tools(self):
        '''
        Initialize view management tools.  This subclass has the ability to show
        multiple sub-views, all at the same zoom level, and possibly with linked
        scrolling.
        '''

        # The split-views chooser is a drop down menu of options the user can
        # choose from.  First populate the menu, then create the chooser button.

        chooser_items = [
            (self.tr('1 row x 1 column'  ), (1, 1)),
            (self.tr('1 row x 2 columns' ), (1, 2)),
            (self.tr('2 rows x 1 column' ), (2, 1)),
            (self.tr('2 rows x 2 columns'), (2, 2)),
        ]
        self._view_chooser = ToolbarMenu(icon=QIcon(':/icons/split-view.svg'),
            items=chooser_items)
        self._view_chooser.setToolTip(self.tr('Split/unsplit the main view'))
        self._toolbar.addWidget(self._view_chooser)
        self._view_chooser.triggered.connect(self._on_split_views)

        self._act_link_view_scroll = add_toolbar_action(self._toolbar,
            ':/icons/link-scroll.svg', self.tr('Link view scrolling'), self)
        self._act_link_view_scroll.setCheckable(True)
        self._act_link_view_scroll.triggered.connect(self._on_link_view_scroll)


    def _init_zoom_tools(self):
        '''
        Initialize zoom toolbar buttons.  This subclass adds a few extra zoom
        buttons to the zoom tools.
        '''
        super()._init_zoom_tools()

        # Zoom to Actual Size
        self._act_zoom_to_actual = add_toolbar_action(self._toolbar,
            ':/icons/zoom-to-actual.svg', self.tr('Zoom to actual size'), self,
            before=self._act_cbox_zoom)
        self._act_zoom_to_actual.triggered.connect(self._on_zoom_to_actual)

        # Zoom to Fit
        self._act_zoom_to_fit = add_toolbar_action(self._toolbar,
            ':/icons/zoom-to-fit.svg', self.tr('Zoom to fit'), self,
            before=self._act_cbox_zoom)
        self._act_zoom_to_fit.triggered.connect(self._on_zoom_to_fit)


    def _context_menu_add_global_items(self, menu, rasterview):
        '''
        This helper function adds "global" items to the context menu, that is,
        items that aren't specific to the location clicked in the window.
        '''
        act = menu.addAction(self.tr('Export to image file...'))
        # When hooking up the action, use a lambda to pass through the
        # rasterview that generated the event.
        act.triggered.connect(lambda checked, rv=rasterview : self._on_export_image(rv))


    def _on_export_image(self, rasterview):
        # TODO:  IMPLEMENT
        pass


    def get_stretch_builder(self):
        return self._stretch_builder


    def _on_stretch_builder(self, act):
        ''' Show the Stretch Builder. '''

        self._stretch_builder.show(self.get_current_dataset(),
                                   self.get_rasterview().get_display_bands(),
                                   self.get_rasterview().get_stretches())


    def _on_split_views(self, action):
        '''
        This function handles when the user requests a split-view layout of some
        structure.
        '''
        new_dim = action.data()
        if new_dim != self._num_views:
            msg = self.tr('Switching main view to {rows}x{cols} display')
            msg = msg.format(rows=new_dim[0], cols=new_dim[1])
            self._app_state.show_status_text(msg, 5)

        self._init_rasterviews(action.data())


    def is_scrolling_linked(self):
        return self._link_view_scrolling


    def _on_link_view_scroll(self, checked):
        '''
        This function handles when the user links or unlinks raster-view
        scrolling.  When raster-view scrolling is linked, all raster views must
        be updated to show the same coordinates as the top left raster view.
        '''

        self._link_view_scrolling = checked
        if checked:
            self._app_state.show_status_text(self.tr('Linked view scrolling is ON'), 5)

            # Sync scroll state of all raster views to the one in the top left.
            self._sync_scroll_state(self.get_rasterview())
        else:
            self._app_state.show_status_text(self.tr('Linked view scrolling is OFF'), 5)


    def _afterRasterScroll(self, rasterview, dx, dy):
        '''
        This function is called when the raster-view's scrollbars are moved.

        It fires an event that the visible region of the raster-view has
        changed.

        If multiple raster-views are active, and scrolling is linked, this also
        propagates the scroll changes to the other raster-views.
        '''
        # Invoke the superclass version of this operation to emit the
        # viewport-changed event.
        super()._afterRasterScroll(rasterview, dx, dy)
        self._sync_scroll_state(rasterview)


    def _sync_scroll_state(self, rasterview: RasterView) -> None:
        '''
        This helper function synchronizes the scroll state of all raster views
        in this pane.  The source rasterview to take the scroll state from is
        specified as the argument.
        '''
        sb_state = rasterview.get_scrollbar_state()
        if len(self._rasterviews) > 1 and self._link_view_scrolling:
            for rv in self._rasterviews.values():
                # Skip the rasterview that generated the scroll event
                if rv is rasterview:
                    continue

                rv.set_scrollbar_state(sb_state)


    def _on_zoom_to_actual(self, evt):
        ''' Zoom the view to 100% scale. '''

        self.set_scale(1.0)
        self._update_zoom_widgets()


    def _on_zoom_to_fit(self):
        ''' Zoom the view such that the entire image fits in the view. '''

        # Use the rasterview at (0, 0) to compute the scale for the image to fit
        rasterview = self.get_rasterview()
        rasterview.scale_image_to_fit()

        # If we are in a multi-view mode, propagate that scale to all views
        if self.is_multi_view():
            self.set_scale(rasterview.get_scale())

        self._update_zoom_widgets()
