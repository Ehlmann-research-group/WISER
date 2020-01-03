from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .rasterview import ScaleToFitMode
from .rasterpane import RasterPane


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


    def _init_zoom_tools(self):
        '''
        Initialize zoom toolbar buttons.  This method replaces the superclass
        method, since the context pane only needs to show one zoom button.
        '''

        self._act_fit_to_window = self._toolbar.addAction(
            QIcon('resources/zoom-to-fit.svg'),
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
        self._update_image()


    def _on_toggle_fit_to_window(self):
        '''
        Update the raster-view image when the "fit to window" button is toggled.
        '''
        self._update_image()


    def _on_view_attr_changed(self, attr_name):
        if attr_name == 'image.visible_area':
            self._rasterview.update()


    def _update_image(self):
        '''
        Scale the raster-view image based on the image size, and the state of
        the "fit to window" button.
        '''

        dataset = None
        if self._app_state.num_datasets() > 0:
            dataset = self._app_state.get_dataset(self._dataset_index)

        # TODO(donnie):  Only do this when the raster dataset actually changes,
        #     or the displayed bands change, etc.
        if dataset != self._rasterview.get_raster_data():
            self._rasterview.set_raster_data(dataset)

        if dataset is None:
            return

        # Handle window-scaling changes
        if self._act_fit_to_window.isChecked():
            # The entire image needs to fit in the summary view.
            self._rasterview.scale_image_to_fit(
                mode=ScaleToFitMode.FIT_BOTH_DIMENSIONS)
        else:
            # Just zoom such that one of the dimensions fits.
            self._rasterview.scale_image_to_fit(
                mode=ScaleToFitMode.FIT_ONE_DIMENSION)
