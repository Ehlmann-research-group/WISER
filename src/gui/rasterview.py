import sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np

from .constants import ImageColors

from raster.dataset import RasterDataSet


from raster.units import find_band_near_wavelength, RED_WAVELENGTH, GREEN_WAVELENGTH, BLUE_WAVELENGTH


class RasterImage(QLabel):
    '''
    A simple subclass of QLabel used for displaying a raster image.  The main
    reason we subclass QLabel is simply to forward mouse-click events from the
    image to the RasterView, which then can scale and interpret them properly.
    '''

    # Signal for when pixels are clicked in the raster image data.
    pixel_clicked = Signal(QMouseEvent)

    def __init__(self, text, parent=None):
        '''
        Initialize the raster-image widget with the specified text and parent.
        '''
        super().__init__(text, parent=parent)

    def mouseReleaseEvent(self, mouse_event):
        '''
        When a mouse-button is released within the widget, this method emits a
        "pixel clicked" signal.  It is up to the recipient to scale this event's
        coordinates appropriately.
        '''
        self.pixel_clicked.emit(mouse_event)


class RasterView(QWidget):
    '''
    A general-purpose widget for viewing raster data at varying zoom levels,
    possibly with overlay information drawn onto the image, such as annotations,
    regions of interest, etc.
    '''

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # The image that the raster view uses to display its data
        self._lbl_image = RasterImage('(no data)')
        self._lbl_image.setBackgroundRole(QPalette.Base)
        self._lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._lbl_image.setScaledContents(False)
        self._lbl_image.pixel_clicked.connect(self.raster_pixel_clicked)

        # The scroll area used to handle images larger than the widget size
        self._scroll_area = QScrollArea()
        self._scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scroll_area.setBackgroundRole(QPalette.Dark)
        self._scroll_area.setWidget(self._lbl_image)
        self._scroll_area.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        # Set up the layout

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        # layout.addWidget(self.image_toolbar)
        # layout.setMenuBar(self.image_toolbar)
        layout.addWidget(self._scroll_area)

        self.setLayout(layout)

        # Initialize fields in the object
        self.clear_members()


    def clear_members(self):
        self._raster_data = None
        self._scale_factor = 1.0

        # TODO(donnie):  This will likely need to migrate into the GUI's info
        #     for each raster data-set.
        self._red_band = 0
        self._green_band = 0
        self._blue_band = 0

        # These members are for storing the components of the raster data, so
        # that assembling the image is faster when only one color's band
        # changes.

        self._red_data = None
        self._green_data = None
        self._blue_data = None
        self._img_data = None

        # The unscaled image, which is always the same dimensions as the raw
        # raster data.  It is scaled for final display.
        self._unscaled_image = None

        # The scaled image, onto which overlay data may be drawn.
        self._scaled_image = None


    # TODO(donnie):  This code likely needs to migrate into the model.  The
    #     raster-view probably shouldn't be this smart.
    def find_display_bands(self, raster_data):
        # TODO(donnie):  See if the raster data specifies display bands, and if
        #     so, use them.

        # Try to find bands based on what is close to visible spectral bands
        bands = raster_data.band_list()
        red_band   = find_band_near_wavelength(bands, RED_WAVELENGTH)
        green_band = find_band_near_wavelength(bands, GREEN_WAVELENGTH)
        blue_band  = find_band_near_wavelength(bands, BLUE_WAVELENGTH)

        # If that didn't work, just choose first, middle and last bands
        if red_band is None or green_band is None or blue_band is None:
            red_band   = 0
            green_band = max(0, raster_data.num_bands() // 2 - 1)
            blue_band  = max(0, raster_data.num_bands() - 1)

        return (red_band, green_band, blue_band)


    def set_raster_data(self, raster_data):
        '''
        Specify a raster data-set to display in the raster-view widget.  A value
        of None causes the raster-view to display nothing.

        The scale factor is reset to 1.0 when this method is called.
        '''
        if raster_data is not None and not isinstance(raster_data, RasterDataSet):
            raise ValueError('raster_data must be a RasterDataSet object')

        self.clear_members()

        self._raster_data = raster_data
        self._scale_factor = 1.0

        if raster_data is not None:
            rgb_bands = self.find_display_bands(raster_data)
            self._red_band   = rgb_bands[0]
            self._green_band = rgb_bands[1]
            self._blue_band  = rgb_bands[2]

        self.update_display_image()


    def extract_band_for_display(self, band_index):
        '''
        Extracts the specified band of raster data for display in an RGB image
        in the user interface.  This operation is done with numpy so that it can
        be completed as efficiently as possible.

        The function returns a numpy array with np.uint32 elements in the range
        of 0..255.
        '''
        band_data = self._raster_data.get_band_data(band_index)

        # TODO(donnie):  Almost certainly, we will need a much more
        #     sophisticated way of specifying how the band data is transformed.
        band_data = (band_data * 255 + 30).clip(0, 255).astype(np.uint32)

        return band_data


    def update_display_image(self, colors=ImageColors.RGB):
        if self._raster_data is None:
            # No raster data to display - clear the view.
            self._lbl_image.clear()
            self._lbl_image.setText('(no data)')
            # self.lbl_image.adjustSize()
            return

        # print("Extracting raw band data")

        # Only generate (or regenerate) each color plane if we don't already
        # have data for it, and if we aren't told to explicitly regenerate it.

        if self._red_data is None or ImageColors.RED in colors:
            # print('Regenerating red data')
            self._red_data = self.extract_band_for_display(self._red_band)

        if self._green_data is None or ImageColors.GREEN in colors:
            # print('Regenerating green data')
            self._green_data = self.extract_band_for_display(self._green_band)

        if self._blue_data is None or ImageColors.BLUE in colors:
            # print('Regenerating blue data')
            self._blue_data = self.extract_band_for_display(self._blue_band)

        # print("Converting raw data to RGB color data")

        # TODO(donnie):  I don't know why the tostring() is required here, but
        #     it seems to be required for making the QImage when we use GDAL.
        img_data = (self._red_data << 16 | self._green_data << 8 | self._blue_data) | 0xff000000
        img_data = img_data.tostring()
        # This is necessary because the QImage doesn't take ownership of the
        # data we pass it, and if we drop this reference to the data then Python
        # will reclaim the memory and Qt will start to display garbage.
        self._img_data = img_data

        # This is the 100% scale QImage of the data.
        self._unscaled_image = QImage(img_data,
            self._raster_data.get_width(), self._raster_data.get_height(),
            QImage.Format_RGB32)

        self.update_scaled_image()


    def update_scaled_image(self):
        scaled_size = self._unscaled_image.size() * self._scale_factor
        self._scaled_image = self._unscaled_image.scaled(scaled_size)

        # print("QImage is complete")

        # print("Displaying image.")
        self._lbl_image.setPixmap(QPixmap.fromImage(self._scaled_image))
        self._lbl_image.adjustSize()
        self._scroll_area.setVisible(True)
        # print("Done.")


    def scale_image(self, factor):
        self._scale_factor = factor
        self.update_scaled_image()


    def resizeEvent(self, event):
        # TODO:  Fire an event that the visible region of the raster-view has
        #        changed.
        self.get_visible_region()


    def get_visible_region(self):
        h_scroll = self._scroll_area.horizontalScrollBar()
        v_scroll = self._scroll_area.verticalScrollBar()

        # print('horz value=%d, min=%d, max=%d' % (h_scroll.value(), h_scroll.minimum(), h_scroll.maximum()))
        # print('vert value=%d, min=%d, max=%d' % (v_scroll.value(), v_scroll.minimum(), v_scroll.maximum()))

        return (0, 0, 0, 0)


    @Slot()
    def zoom_in(self, evt):
        if self._raster_data is None:
            return

        self.scale_image(self._scale_factor * 1.25)
        # TODO:  Disable zoom-in if too zoomed

    @Slot()
    def zoom_out(self, evt):
        if self._raster_data is None:
            return

        self.scale_image(self._scale_factor * 0.8)
        # TODO:  Disable zoom-out if too un-zoomed

    @Slot()
    def zoom_to_actual(self, evt):
        if self._raster_data is None:
            return

        self.scale_image(1.0)

    @Slot()
    def zoom_to_fit(self, evt):
        if self._raster_data is None:
            return

        # Figure out the appropriate scale factor, then do it
        new_factor = min(
            self._scroll_area.width() / self._raster_data.get_width(),
            self._scroll_area.height() / self._raster_data.get_height()
        )
        self.scale_image(new_factor)

    # @Slot()
    # def choose_colors(self, evt):
    #     if not self.rgb_selector.isVisible():
    #         self.rgb_selector.set_slider_ranges(self.raster_data.num_bands())
    #         self.rgb_selector.set_slider_values(self.red_band, self.green_band, self.blue_band)
    #         self.rgb_selector.setVisible(True)
    #
    #     else:
    #         self.rgb_selector.setVisible(False)

    # TODO(donnie):  Should be Slot(ImageColors, int), but causes PySide2 to
    #     crash at startup.
    @Slot(int, int)
    def rgb_band_changed(self, color, band_index):
        # print(f'Color:  {color}\tNew band:  {band_index}')

        # TODO(donnie):  See above TODO
        color = ImageColors(color)

        if color == ImageColors.RED:
            self._red_band = band_index

        elif color == ImageColors.GREEN:
            self._green_band = band_index

        elif color == ImageColors.BLUE:
            self._blue_band = band_index

        else:
            print(f'WARNING:  Unrecognized color # {color}')

        self.update_display_image(colors=color)

    # TODO(donnie):  Change the name of this to e.g. view_pixel_clicked()
    @Slot(QMouseEvent)
    def raster_pixel_clicked(self, mouse_event):
        # Map the coordinate of the mouse-event to the actual raster-image
        # pixel that was clicked.
        # TODO(donnie):  I could move this calculation inside the if statement,
        #     but I have a feeling we may want to do other things when raster
        #     pixels are clicked.
        raster_point = (mouse_event.localPos() / self._scale_factor).toPoint()
        (raster_x, raster_y) = raster_point.toTuple()
        # print(f'Raster pixel clicked:  {raster_point}')

        # if self.spectrum_plot.isVisible():
        #     # TODO(donnie):  Extract the spectrum at the specified pixel, then
        #     #     display it in the spectrum plot.
        #     spectrum = self.raster_data.get_all_bands_at(raster_x, raster_y)
        #     self.spectrum_plot.add_spectrum(spectrum)


    # @Slot()
    # def show_spectrum(self, evt):
    #     if not self.spectrum_plot.isVisible():
    #         self.spectrum_plot.clear()
    #         self.spectrum_plot.setVisible(True)
    #
    #     else:
    #         self.rgb_selector.setVisible(False)
