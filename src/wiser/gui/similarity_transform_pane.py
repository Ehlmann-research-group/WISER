from typing import TYPE_CHECKING, Tuple

import logging

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .band_chooser import BandChooserDialog
from .rasterview import RasterView, ImageWidget, ImageColors, ScaleToFitMode, \
                        make_channel_image, make_rgb_image, make_grayscale_image
from .rasterpane import RasterPane, TiledRasterView, RecenterMode
from .app_state import ApplicationState
from wiser.raster.selection import SinglePixelSelection
from .util import pillow_rotate_scale_expand, cv2_rotate_scale_expand

if TYPE_CHECKING:
    from .geo_reference_task_delegate import GeoReferencerTaskDelegate

import numpy as np
from PIL import Image
import cv2
import time

logger = logging.getLogger(__name__)

WIDTH_INDEX = 1
HEIGHT_INDEX = 0

class SimilarityTransformImageWidget(ImageWidget):

    def set_dataset_info(self, array, scale):
        # TODO(donnie):  Do something
        if array is not None:
            width = array[WIDTH_INDEX]
            height = array[HEIGHT_INDEX]
            self._scaled_size = QSize(int(width * scale), int(height * scale))
        else:
            self._scaled_size = None

        # Inform the parent widget/layout that the geometry may have changed.
        self.setFixedSize(self._get_size_of_contents())

        # Request a repaint, since this function is called when any details
        # about the dataset are modified (including stretch adjustments, etc.)
        self.update()

class SimilarityTransformRasterView(TiledRasterView):

    def rotate_pixmap(self, pixmap: QPixmap, angle_deg: float) -> QPixmap:
        center = pixmap.rect().center()

        tr = (QTransform()
            .translate(center.x(), center.y())
            .rotate(angle_deg)
            .translate(-center.x(), -center.y()))

        return pixmap.transformed(tr, Qt.SmoothTransformation)

    def scale_pixmap(self, pixmap: QPixmap, factor: float) -> QPixmap:
        return pixmap.scaled(pixmap.size() * factor,
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)

    def update_display_image(self, rotation: float = 0.0, scale: float = 1.0, colors=ImageColors.RGB):
        '''
        Overrides RasterViews version of this function to allow for rotation (degrees) and scaling.
        '''
        img_data = None
        if self._raster_data is None:
            # No raster data to display
            self._image_widget.set_dataset_info(None, self._scale_factor)
            return

        # Only generate (or regenerate) each color plane if we don't already
        # have data for it, and if we aren't told to explicitly regenerate it.

        assert len(self._display_bands) in [1, 3]

        time_1 = time.perf_counter()
        if len(self._display_bands) == 3:
            # Check each color band to see if we need to update it.
            color_indexes = [ImageColors.RED, ImageColors.GREEN, ImageColors.BLUE]
            for i in range(len(self._display_bands)):
                if self._display_data[i] is None or color_indexes[i] in colors:
                    # Compute the contents of this color channel.
            
                    arr = self._raster_data.get_band_data_normalized(self._display_bands[i])

                    band_data = arr
                    band_mask = None
                    if isinstance(arr, np.ma.masked_array):
                        band_data = arr.data
                        band_mask = arr.mask
                    stretches = [None, None]
                    if self._stretches[i]:
                        stretches = self._stretches[i].get_stretches()
                    new_data = make_channel_image(band_data, stretches[0], stretches[1])

                    new_arr = new_data
                    if isinstance(arr, np.ma.masked_array):
                        new_arr = np.ma.masked_array(new_data, mask=band_mask)
                        new_arr.data[band_mask] = 0
                    new_arr = cv2_rotate_scale_expand(new_arr, angle=rotation, scale=scale, mask_fill_value=0)

                    self._display_data[i] = new_arr

            time_2 = time.perf_counter()

            if isinstance(self._display_data[0], np.ma.masked_array):
                band_masks = []
                for data in self._display_data:
                    band_masks.append(data.mask)
                img_data = make_rgb_image(self._display_data[0].data, self._display_data[1].data, self._display_data[2].data)
            
                if not img_data.flags['C_CONTIGUOUS']:
                    img_data = np.ascontiguousarray(img_data)
                
                mask = np.zeros(img_data.shape, dtype=bool)
                img_data = np.ma.masked_array(img_data, mask)
            else:
                img_data = make_rgb_image(self._display_data[0], self._display_data[1], self._display_data[2])

        else:
            # This is a grayscale image.
            if colors != ImageColors.NONE or self._display_data[0] is None:
                # Regenerate the image.  Since all color bands are the same,
                # generate the first one, then duplicate it for the other two
                # bands.

                arr = self._raster_data.get_band_data_normalized(self._display_bands[0])

                band_data = arr
                band_mask = None
                if isinstance(arr, np.ma.masked_array):
                    band_data = arr.data
                    band_mask = arr.mask

                stretches = [None, None]
                if self._stretches[0]:
                    stretches = self._stretches[0].get_stretches()
                new_data  = make_channel_image(band_data, stretches[0], stretches[1])

                new_arr = new_data
                if isinstance(arr, np.ma.masked_array):
                    new_arr = np.ma.masked_array(new_data, mask=band_mask)
                    new_arr.data[band_mask] = 0
                
                self._display_data[0] = new_arr
                self._display_data[1] = self._display_data[0]
                self._display_data[2] = self._display_data[0]

            time_2 = time.perf_counter()

            # Combine our individual color channel(s) into a single RGB image.
            img_data = make_grayscale_image(self._display_data[0], self._colormap)

        # This is necessary because the QImage doesn't take ownership of the
        # data we pass it, and if we drop this reference to the data then Python
        # will reclaim the memory and Qt will start to display garbage.

        # assert isinstance(img_data, (np.ndarray, np.ma.masked_array))
        # img_data = self.pillow_rotate_scale_expand(img_data, 45)

        self._img_data = img_data
        self._img_data.flags.writeable = False

        time_3 = time.perf_counter()

        # This is the 100% scale QImage of the data.
        # self._image = QImage(img_data,
        #     self._raster_data.get_width(), self._raster_data.get_height(),
        #     QImage.Format_RGB32)
        self._image = QImage(img_data,
           img_data.shape[1], img_data.shape[0],
            QImage.Format_RGB32)
        

        self._image_pixmap = QPixmap.fromImage(self._image)
        # self._image_pixmap = self.rotate_pixmap(self._image_pixmap, 45)
        # self._image_pixmap = self.scale_pixmap(self._image_pixmap, 2)

        time_4 = time.perf_counter()

        logger.debug(f'update_display_image(colors={colors}) update times:  ' +
                     f'channels = {time_2 - time_1:0.02f}s ' +
                     f'image = {time_3 - time_2:0.02f}s ' +
                     f'qt = {time_4 - time_3:0.02f}s')

        self._update_scaled_image(update_by_dataset=False)


class SimilarityTransformPane(RasterPane):

    # The first parameter is the dataset, the second is the point
    pixel_selected_for_translation = Signal(object, tuple)

    dataset_changed = Signal(int)

    def __init__(self, app_state, translation=False, parent=None):
        super().__init__(app_state=app_state, parent=parent,
            max_zoom_scale=64, zoom_options=[0.25, 0.5, 0.75, 1, 2, 4, 8, 16, 24, 32],
            initial_zoom=1)
        
        self._translation = translation
        self.click_pixel.connect(self._on_similarity_transform_raster_pixel_select)

    def _init_rasterviews(self, num_views: Tuple[int, int]=(1, 1), rasterview_class: TiledRasterView = TiledRasterView):
        rasterview_class = SimilarityTransformRasterView
        return super()._init_rasterviews(num_views, rasterview_class)

    def rotate_and_scale_rasterview(self, rotation:float, scale: float):
        rv: SimilarityTransformRasterView = self.get_rasterview()
        rv.update_display_image(rotation=rotation, scale=scale)

    def _init_select_tools(self):
        '''
        We don't want this to initialize any of the select tools.
        The select tools currently are just the ROI tools
        '''
        return

    def _init_zoom_tools(self):
        '''
        Initialize zoom toolbar buttons.  This method replaces the superclass
        method, since the context pane only needs to show one zoom button.
        '''
        super()._init_zoom_tools()
        self._act_fit_to_window = self._toolbar.addAction(
            QIcon(':/icons/zoom-to-fit.svg'),
            self.tr('Fit image to window'))
        self._act_fit_to_window.setCheckable(True)
        self._act_fit_to_window.setChecked(False)

        self._act_fit_to_window.triggered.connect(self._on_toggle_fit_to_window)

    
    def _on_toggle_fit_to_window(self):
        '''
        Update the raster-view image when the "fit to window" button is toggled.
        '''
        self._update_image_scale()

    def _on_dataset_added(self, ds_id):
        pass

    # def _on_dataset_changed(self, act):
    #     super()._on_dataset_changed(act)
    #     (rasterview_pos, ds_id) = act.data()
    #     self.dataset_changed.emit(ds_id)

    def _on_dataset_changed(self, act):
        (rasterview_pos, ds_id) = act.data()
        dataset = self._app_state.get_dataset(ds_id)
        if dataset.has_geographic_info() or not self._translation:
            self.show_dataset(dataset, rasterview_pos)
            self.dataset_changed.emit(ds_id)
        else:
            # Only show this if we do not have geographic information and we are
            # the translation pane
            QMessageBox.information(self,
                                    self.tr("No Geographic Information"),
                                    self.tr("Selected dataset must have geographic information\n" \
                                    "to use this feature."))


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


    def _on_band_chooser(self, checked=False, rasterview_pos=(0,0)):
        rasterview = self.get_rasterview(rasterview_pos)
        dataset = rasterview.get_raster_data()
        display_bands = rasterview.get_display_bands()
        colormap = rasterview.get_colormap()

        dialog = BandChooserDialog(self._app_state, dataset, display_bands,
            colormap=colormap, can_apply_global=False, parent=self)
        dialog.setModal(True)

        if dialog.exec_() == QDialog.Accepted:
            bands = dialog.get_display_bands()
            colormap = dialog.get_colormap_name()

            # For the geo referencer, the change shouldn't be global
            self.set_display_bands(dataset.get_id(), bands, colormap=colormap)

    def _on_similarity_transform_raster_pixel_select(self, rasterview_position, ds_point):
        print(f"_on_similarity_transform_raster_pixel_select called!\n raster pos: {rasterview_position}, ds_point: {ds_point}")
        # Get the dataset of the main view.  If no dataset is being displayed or
        # this is not the translation pane, then this is a no-op
        if self._translation:
            ds = self.get_current_dataset(rasterview_position)
            if ds is None:
                # The clicked-on rasterview has no dataset loaded; ignore.
                return
            sel = SinglePixelSelection(ds_point, ds)
            self.set_pixel_highlight(sel, recenter=RecenterMode.NEVER)

            self.pixel_selected_for_translation.emit(ds, ds_point)
