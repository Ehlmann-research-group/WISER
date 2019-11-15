import sys
from enum import Enum

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np

from .constants import ImageColors

from raster.dataset import RasterDataSet


from raster.units import find_band_near_wavelength, RED_WAVELENGTH, GREEN_WAVELENGTH, BLUE_WAVELENGTH


class ZoomToFitMode(Enum):
    '''
    The "zoom to fit" operation can be customized in several ways, depending on
    how the image needs to fit.  This enumeration specifies exactly how the zoom
    operation is to be done.
    '''

    FIT_HORIZONTAL      = 1
    FIT_VERTICAL        = 2
    FIT_ONE_DIMENSION   = 3
    FIT_BOTH_DIMENSIONS = 4


class ImageWidget(QLabel):
    '''
    A simple subclass of QLabel used for displaying an image.  The main reason
    we subclass QLabel is simply to forward mouse-click events from the image
    to the RasterView, which then can scale and interpret them properly.
    '''

    # Signal for when the mouse is clicked in the image widget.
    mouse_click = Signal(QMouseEvent)

    # Signal for when the mouse is moved in the image widget.
    mouse_move = Signal(QMouseEvent)

    def __init__(self, text, parent=None):
        '''
        Initialize the raster-image widget with the specified text and parent.
        '''
        super().__init__(text, parent=parent)

    def mouseReleaseEvent(self, mouse_event):
        '''
        When a mouse-button is released within the widget, this method emits a
        "mouse click" signal.  It is up to the recipient to scale this event's
        coordinates appropriately.
        '''
        self.mouse_click.emit(mouse_event)

    def mouseMoveEvent(self, mouse_event):
        self.mouse_move.emit(mouse_event)


class RasterView(QWidget):
    '''
    A general-purpose widget for viewing raster data at varying zoom levels,
    possibly with overlay information drawn onto the image, such as annotations,
    regions of interest, etc.
    '''

    # Signal for when a mouse click occurs.  The coordinates of the pixel in
    # the raster image are included in the arguments.
    raster_mouse_click = Signal( (int, int), QMouseEvent)

    # Signal for when the mouse moves.  The coordinates of the pixel in
    # the raster image are included in the arguments.  Note that
    # enable_mouse_move must be set to True when this object is initialized, for
    # mouse-move events to be generated.
    raster_mouse_move = Signal( (int, int), QMouseEvent)

    # Signal for when the raster display-area changes.  The rectangle of the new
    # display area is reported to the signal handler, using raster dataset
    # coordinates:  (x, y, width, height).
    raster_viewport_change = Signal(QRect)


    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # The image that the raster view uses to display its data

        self._lbl_image = ImageWidget('(no data)')
        self._lbl_image.setBackgroundRole(QPalette.Base)
        self._lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._lbl_image.setScaledContents(False)
        self._lbl_image.mouse_click.connect(self._mouse_click)
        self._lbl_image.mouse_move.connect(self._mouse_move)

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
        self._clear_members()


    def _clear_members(self):
        '''
        A helper function to clear all raster dataset members when the dataset
        changes.  This way we don't accidentally leave anything out.
        '''

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

        self._clear_members()

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
        '''
        Fire an event that the visible region of the raster-view has changed.
        '''
        self.raster_viewport_change.emit(self.get_visible_region())


    def get_visible_region(self):
        if self._raster_data is None:
            return (0, 0, 0, 0)

        h_start = int(self._scroll_area.horizontalScrollBar().value() / self._scale_factor)
        v_start = int(self._scroll_area.verticalScrollBar().value() / self._scale_factor)

        h_size = int(self._scroll_area.viewport().width() / self._scale_factor)
        v_size = int(self._scroll_area.viewport().height() / self._scale_factor)

        h_size = min(h_size, self._raster_data.get_width())
        v_size = min(v_size, self._raster_data.get_height())

        # print(f'Raster data is {self._raster_data.get_width()} x {self._raster_data.get_height()} pixels')

        visible_region = QRect(h_start, v_start, h_size, v_size)

        # print(f'Visible region = {visible_region}')

        return visible_region


    @Slot()
    def zoom_in(self, evt):
        '''
        If the raster-view widget has image data, the view is zoomed in by 20%.

        If there is no image data, this is a no-op.
        '''
        if self._raster_data is None:
            return

        self.scale_image(self._scale_factor * 1.25)
        # TODO:  Disable zoom-in if too zoomed

    @Slot()
    def zoom_out(self, evt):
        '''
        If the raster-view widget has image data, the view is zoomed out by 20%.

        If there is no image data, this is a no-op.
        '''
        if self._raster_data is None:
            return

        self.scale_image(self._scale_factor * 0.8)
        # TODO:  Disable zoom-out if too un-zoomed

    @Slot()
    def zoom_to_actual(self, evt):
        '''
        If the raster-view widget has image data, the view is zoomed to the
        actual size of the data; that is, each screen pixel corresponds to one
        pixel in the raster data.

        If there is no image data, this is a no-op.
        '''
        if self._raster_data is None:
            return

        self.scale_image(1.0)

    @Slot()
    def zoom_to_fit(self, mode=ZoomToFitMode.FIT_BOTH_DIMENSIONS):

        '''
        If the raster-view widget has image data, the view is zoomed such that
        the entire raster image data is visible within the view.  No scroll bars
        are visible after this operation.

        If there is no image data, this is a no-op.
        '''
        if self._raster_data is None:
            return

        # Figure out the appropriate scale factor for if no scrollbars were
        # needed, then apply that scale.
        area_size = self._scroll_area.maximumViewportSize()

        # TODO(donnie):  All options other than "fit both dimensions" are buggy!
        #     When fitting in one dimension, need to determine whether the
        #     other dimension's scrollbar is actually needed or not.  Then, zoom
        #     appropriately.

        if mode == ZoomToFitMode.FIT_HORIZONTAL:
            new_factor = area_size.width() / self._raster_data.get_width()

        elif mode == ZoomToFitMode.FIT_VERTICAL:
            new_factor = area_size.height() / self._raster_data.get_height()

        elif mode == ZoomToFitMode.FIT_ONE_DIMENSION:
            new_factor = max(
                area_size.width() / self._raster_data.get_width(),
                area_size.height() / self._raster_data.get_height()
            )

        elif mode == ZoomToFitMode.FIT_BOTH_DIMENSIONS:
            new_factor = min(
                area_size.width() / self._raster_data.get_width(),
                area_size.height() / self._raster_data.get_height()
            )

        else:
            raise ValueError(f'Unrecognized mode value {mode}')

        self.scale_image(new_factor)

        # print('ZOOM TO FIT:')
        # print(f'Max viewport size = {area_size.width()} x {area_size.height()}')
        # print(f'Scale factor = {new_factor}')
        # print(f'Scaled image size = {self._scaled_image.width()} x {self._scaled_image.height()}')


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


    def _image_coord_to_raster_coord(self, position):
        '''
        Takes a position in screen space as a QPointF object, and translates it
        into a 2-tuple containing the (X, Y) coordinates of the position within
        the raster data set.
        '''
        raster_point = (position / self._scale_factor).toPoint()
        (raster_x, raster_y) = raster_point.toTuple()


    @Slot(QMouseEvent)
    def _mouse_click(self, mouse_event):
        '''
        When the display image is clicked on, this method gets invoked, and it
        translates the click event's coordinates into the location on the
        raster data set.
        '''
        # Map the coordinate of the mouse-event to the actual raster-image
        # pixel that was clicked, then emit a signal.
        r_coord = self._image_coord_to_raster_coord(mouse_event.localPos())
        self.raster_mouse_click.emit(r_coord, mouse_event)

    @Slot(QMouseEvent)
    def _mouse_move(self, mouse_event):
        '''
        When the mouse is moved on the display image, this method gets invoked,
        and it translates the move event's coordinates into the location on the
        raster data set.
        '''
        # Map the coordinate of the mouse-event to the actual raster-image
        # pixel that was clicked, then emit a signal.
        r_coord = self._image_coord_to_raster_coord(mouse_event.localPos())
        self.raster_mouse_move.emit(r_coord, mouse_event)
