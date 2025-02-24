from typing import List, Union, Dict

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import wiser.gui.generated.resources

from .rasterview import ScaleToFitMode, RasterView
from .rasterpane import RasterPane

from .util import get_painter


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
    
        self._viewport_highlight: Dict[int, List[Union[QRect, QRectF]]] = None


    def _init_toolbar(self):
        '''
        The Context Pane only initializes dataset tools and zoom tools.
        '''
        self._init_dataset_tools()
        self._toolbar.addSeparator()
        self._init_zoom_tools()


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


    def _on_dataset_added(self, ds_id):
        '''
        Override the base-class implementation so we can also update the
        image scaling.
        '''
        super()._on_dataset_added(ds_id)
        self._update_image_scale()


    def _on_dataset_removed(self, ds_id):
        '''
        Override the base-class implementation so we can also update the
        image scaling.
        '''
        super()._on_dataset_removed(ds_id)
        self._update_image_scale()


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
                # print(f"Updating context pane")
                rv.update()
                continue
            
            # TLDR: Its not possible for the first element in viewports to be none 
            # if viewports just has one element. This situation would arise
            # if the context pane displayed a dataset but none of the 
            # rasterviews did, but in that case, there is no viewport_highlight
            # to set so this function would never be called. 
            if not visible.contains(viewports[0]):
                center = viewports[0].center()
                # TODO (Joshua G-K): Make this compatible with geographic linking
                rv.make_point_visible(center.x(), center.y(), reference_rasterview=rasterviews[0])

            # print(f"Updating context pane")
            # Repaint raster-view
            rv.update()
