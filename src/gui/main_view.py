import os

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .dataset_chooser import DatasetChooser
from .toolbarmenu import ToolbarMenu
from .rasterpane import RasterPane
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
            'resources/stretch-builder.svg', self.tr('Stretch builder'), self)
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
        self._view_chooser = ToolbarMenu(icon=QIcon('resources/split-view.svg'),
            items=chooser_items)
        self._view_chooser.setToolTip(self.tr('Split/unsplit the main view'))
        self._toolbar.addWidget(self._view_chooser)
        self._view_chooser.triggered.connect(self._on_split_views)

        self._act_link_view_scroll = add_toolbar_action(self._toolbar,
            'resources/link-scroll.svg', self.tr('Link view scrolling'), self)
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
            'resources/zoom-to-actual.svg', self.tr('Zoom to actual size'), self,
            before=self._act_cbox_zoom)
        self._act_zoom_to_actual.triggered.connect(self._on_zoom_to_actual)

        # Zoom to Fit
        self._act_zoom_to_fit = add_toolbar_action(self._toolbar,
            'resources/zoom-to-fit.svg', self.tr('Zoom to fit'), self,
            before=self._act_cbox_zoom)
        self._act_zoom_to_fit.triggered.connect(self._on_zoom_to_fit)


    def get_stretch_builder(self):
        return self._stretch_builder


    def _on_stretch_builder(self, act):
        ''' Show the Stretch Builder. '''

        self._stretch_builder.show(self.get_current_dataset(),
                                   self._rasterview.get_display_bands(),
                                   self._rasterview.get_stretches())


    def _on_split_views(self, action):
        '''
        This function handles when the user requests a split-view layout of some
        structure.
        '''
        self._init_rasterviews(action.data())


    def _on_link_view_scroll(self, checked):
        '''
        This function handles when the user links or unlinks raster-view
        scrolling.  When raster-view scrolling is linked, all raster views must
        be updated to show the same coordinates as the top left raster view.
        '''

        self._link_view_scrolling = checked
        if checked:
            self._app_state.show_status_text(self.tr('Linked view scrolling is ON'), 5)

            # TODO(donnie):  Update scroll state of all raster views
        else:
            self._app_state.show_status_text(self.tr('Linked view scrolling is OFF'), 5)



    def _on_zoom_to_actual(self, evt):
        ''' Zoom the view to 100% scale. '''

        self.get_rasterview().scale_image(1.0)
        self._update_zoom_widgets()


    def _on_zoom_to_fit(self):
        ''' Zoom the view such that the entire image fits in the view. '''
        self.get_rasterview().scale_image_to_fit()
        self._update_zoom_widgets()
