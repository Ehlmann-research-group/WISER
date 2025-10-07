from typing import List, Union, Dict

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import wiser.gui.generated.resources

from wiser.raster.dataset import RasterDataSet

from .rasterview import ScaleToFitMode, RasterView
from .rasterpane import RasterPane
from .dataset_chooser import DatasetChooser
from .util import add_toolbar_action
from .app_state import ApplicationState
from wiser.raster.dataset import find_display_bands

class ContextPaneDatasetChooser(DatasetChooser):
    '''
    A customized subclass of the DatasetChooser toolbar-button/widget for the
    Context Pane window to use to "lock" the dataset that the Context Pane
    can show.
    '''
    def __init__(self, raster_pane: RasterPane, app_state: ApplicationState):
        super().__init__(raster_pane, app_state)


    def _add_dataset_menu_items(self, menu, rasterview_pos=(0, 0)):
        '''
        Override the parent-class implementation to add a "use clicked dataset"
        option.
        '''
        # Find the action that is currently selected (if any)
        current_data = None
        for act in menu.actions():
            if act.isChecked():
                current_data = act.data()

        # Remove all existing actions
        menu.clear()

        actDefault: QAction = menu.addAction(self.tr('Use clicked dataset'))
        actDefault.setCheckable(True)
        actDefault.setChecked(True)
        actDefault.setData( (None, -1) )
        menu.addSeparator()

        # Add an action for each dataset
        for dataset in self._app_state.get_datasets():
            # TODO(donnie):  Eventually, include the path if the name isn't unique.
            act = QAction(dataset.get_name(), menu)
            act.setCheckable(True)
            act_data = (rasterview_pos, dataset.get_id())
            act.setData(act_data)
            if act_data == current_data:
                act.setChecked(True)
                actDefault.setChecked(False)

            menu.addAction(act)

class ContextPane(RasterPane):
    '''
    This widget provides a context pane in the user interface, along with a
    toolbar of overview/context related tools.  The pane shows the raster image
    zoomed out, with highlights such as the area visible in the main image
    window, annotation locations, etc.
    '''

    def __init__(self, app_state, parent=None):
        super().__init__(app_state=app_state, parent=parent,
            size_hint=QSize(200, 200))


    def _init_toolbar(self):
        '''
        The Context Pane only initializes dataset tools and zoom tools.
        '''
        self._init_dataset_tools()
        self._toolbar.addSeparator()
        self._init_zoom_tools()

    def _init_dataset_tools(self):
        '''
        Initialize dataset-management toolbar buttons.
        '''

        # Dataset / Band Tools

        self._dataset_chooser = ContextPaneDatasetChooser(self, self._app_state)
        self._toolbar.addWidget(self._dataset_chooser)
        self._dataset_chooser.triggered.connect(self._on_dataset_changed)
        self._ds_id = -1

        self._act_band_chooser = add_toolbar_action(self._toolbar,
            ':/icons/choose-bands.svg', self.tr('Band chooser'), self)
        # TODO(donnie):  If we just pop up a widget...
        # self._band_chooser = BandChooser(self._app_state)
        # toolbar.addWidget(self._band_chooser)
        self._act_band_chooser.triggered.connect(self._on_band_chooser)
        self._act_band_chooser.setEnabled(False)


    def _init_zoom_tools(self):
        '''
        Initialize zoom toolbar buttons.  This method replaces the superclass
        method, since the context pane only needs to show one zoom button.
        '''

        self._act_fit_to_window = self._toolbar.addAction(
            QIcon(':/icons/zoom-to-fit.svg'),
            self.tr('Fit image to window'))
        self._act_fit_to_window.setCheckable(True)
        self._act_fit_to_window.setChecked(True)

        self._act_fit_to_window.triggered.connect(self._on_toggle_fit_to_window)


    def _update_zoom_widgets(self):
        '''
        This method replaces the superclass method, since the context pane
        doesn't provide typical zoom operations.
        '''
        pass


    def resizeEvent(self, event):
        '''
        The context pane updates the image scaling when it is resized, so that
        the required part of the dataset stays in view when the window
        dimensions change.
        '''
        self._update_image_scale()


    def show_dataset(self, dataset: RasterDataSet, rasterview_pos=(0, 0)):
        '''
        Sets the dataset being displayed in the specified view of the raster
        pane.
        '''
        if self._ds_id == -1 or self._ds_id == dataset.get_id():
            rasterview = self.get_rasterview(rasterview_pos)

            # If the rasterview is already showing the specified dataset, skip!
            if rasterview.get_raster_data() is dataset:
                return

            bands = None
            stretches = None
            if dataset is not None:
                ds_id = dataset.get_id()
                if ds_id not in self._display_bands:
                    display_bands = find_display_bands(dataset)
                    self._display_bands[ds_id] = display_bands
                bands = self._display_bands[ds_id]
                stretches = self._app_state.get_stretches(ds_id, bands)

            rasterview.set_raster_data(dataset, bands, stretches)


    def _on_dataset_changed(self, act):
        (rasterview_pos, ds_id) = act.data()
        self._ds_id = ds_id
        if ds_id != -1:
            dataset = self._app_state.get_dataset(ds_id)
            self.show_dataset(dataset, rasterview_pos)


    def _on_dataset_added(self, ds_id, view_dataset: bool = True):
        '''
        Override the base-class implementation so we can also update the
        image scaling.
        '''
        super()._on_dataset_added(ds_id, view_dataset)
        self._update_image_scale()


    def _on_dataset_removed(self, ds_id):
        '''
        Override the base-class implementation so we can also update the
        image scaling.
        '''
        super()._on_dataset_removed(ds_id)
        self._update_image_scale()
        if ds_id == self._ds_id:
            self._ds_id = -1


    def _on_toggle_fit_to_window(self):
        '''
        Update the raster-view image when the "fit to window" button is toggled.
        '''
        self._update_image_scale()


    def _update_image_scale(self):
        '''
        Scale the raster-view image based on the image size, and the state of
        the "fit to window" button.
        '''

        # Handle window-scaling changes
        if self._act_fit_to_window.isChecked():
            # The entire image needs to fit in the summary view.
            self.get_rasterview().scale_image_to_fit(
                mode=ScaleToFitMode.FIT_BOTH_DIMENSIONS)
        else:
            # Just zoom such that one of the dimensions fits.
            self.get_rasterview().scale_image_to_fit(
                mode=ScaleToFitMode.FIT_ONE_DIMENSION)

            
    def set_viewport_highlight(self, viewports: List[Union[QRect, QRectF]], \
                               rasterviews: List[RasterView]):
        '''
        Sets the "viewport highlight" to be displayed in this raster-pane.  This
        is used to allow the Context Pane to show the Main View viewport. This 
        function is triggered by a scroll event in the mainview. Although we collect all
        rasterviews from the mainview, we only want those rasterviews that are compatible
        with the rasterview displayed in the context pane to display.
        '''
        self.create_viewport_highlight_dictionary(viewports, rasterviews)

        # If the specified viewport highlight region is not entirely within this
        # raster-view's visible area, scroll such that the viewport highlight is
        # in the middle of the raster-view's visible area. This is only done if 
        # the context pane has one item in the viewport because it doesn't make
        # sense to make two points be in the middle of the raster-view's visible
        # area.

        # This will update on all the rasterviews in Context Pane although, as of 
        # 02/20/2025, there is only one rasterview in Context Pane. 

        # The call to update causes the highlight boxes to be drawn in Context
        # Pane's rasterview. However, the rasterview will only draw the boxes
        # that have the same dataset id as the rasterview.

        for rv in self._rasterviews.values():
            visible = rv.get_visible_region()
            if visible is None or viewports is None or len(viewports) > 1:
                rv.update()
                continue
            
            # Extra Note: Its not possible for the first element in viewports to be none 
            # if viewports just has one element. This situation would arise
            # if the context pane displayed a dataset but none of the 
            # rasterviews did, but in that case, there is no viewport_highlight
            # to set so this function would never be called. 
            if not visible.contains(viewports[0]):
                center = viewports[0].center()
                # TODO (Joshua G-K): Make this compatible with geographic linking
                rv.make_point_visible(center.x(), center.y(), reference_rasterview=rasterviews[0])

            # Repaint raster-view
            rv.update()
