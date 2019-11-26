import sys
from enum import Enum

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np

from .constants import ImageColors

from raster.dataset import RasterDataSet, find_display_bands


class ScaleToFitMode(Enum):
    '''
    The "scale to fit" operation can be performed in several ways, depending
    on how the image needs to fit into its viewing area.  This enumeration
    specifies exactly how the image scaling operation is to be done.
    '''

    # Fit the horizontal dimension entirely into the viewing area.
    FIT_HORIZONTAL      = 1

    # Fit the vertical dimension entirely into the viewing area.
    FIT_VERTICAL        = 2

    # Fit either the horizontal or the vertical dimension entirely into the
    # viewing area.  Which one is dependent on the dimensions of the image and
    # the dimensions of the viewing area.
    FIT_ONE_DIMENSION   = 3

    # Fit both dimensions of the image entirely into the viewing area.
    FIT_BOTH_DIMENSIONS = 4


class ImageWidget(QLabel):
    '''
    A subclass of QLabel used for displaying an image.  Since Qt provides events
    via virtual functions to be overloaded, this class forwards a number of
    important events to the enclosing widget.
    '''

    def __init__(self, text, parent=None, **kwargs):
        '''
        Initialize the image widget with the specified text and parent.  Store
        the object we are to forward relevant events to.
        '''
        super().__init__(text, parent=parent)
        self._forward = kwargs

    def mouseReleaseEvent(self, mouse_event):
        if 'mouseReleaseEvent' in self._forward:
            self._forward['mouseReleaseEvent'](self, mouse_event)

    def mouseMoveEvent(self, mouse_event):
        if 'mouseMoveEvent' in self._forward:
            self._forward['mouseMoveEvent'](self, mouse_event)

    def paintEvent(self, paint_event):
        super().paintEvent(paint_event)

        if 'paintEvent' in self._forward:
            self._forward['paintEvent'](self, paint_event)


class ImageScrollArea(QScrollArea):
    '''
    A simple subclass of QScrollArea used for displaying an image that is
    potentially larger than the available display-area.  The main reason we
    subclass QScrollArea is simply to forward viewport-scroll events from the
    scroll-area to the RasterView, which can then act accordingly.
    '''

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self._forward = kwargs

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)

        if 'scrollContentsBy' in self._forward:
            self._forward['scrollContentsBy'](self, dx, dy)


class RasterView(QWidget):
    '''
    A general-purpose widget for viewing raster datasets at varying zoom levels,
    possibly with overlay information drawn onto the image, such as annotations,
    regions of interest, etc.

    TODO(donnie):  Probably want to provide a mouse-move event at some point in
                   the future, so that we can have hover-over tooltip type
                   displays of annotations, regions of interest, etc.
    '''

    # Signal for when a mouse click occurs.  The coordinates of the pixel in
    # the raster image are included in the arguments.
    mouse_click = Signal(QPoint, QMouseEvent)

    # Signal for when the mouse moves.  The coordinates of the pixel in
    # the raster image are included in the arguments.  Note that
    # enable_mouse_move must be set to True when this object is initialized, for
    # mouse-move events to be generated.
    mouse_move = Signal(QPoint, QMouseEvent)

    # Signal for when the raster display-area changes.  The rectangle of the new
    # display area is reported to the signal handler, using raster dataset
    # coordinates:  (x, y, width, height).
    viewport_change = Signal(QRect)


    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # The QLabel widget used to display the image data

        self._lbl_image = ImageWidget('(no data)',
            mouseReleaseEvent=self._onRasterMouseClick,
            paintEvent=self._afterRasterPaint)

        self._lbl_image.setBackgroundRole(QPalette.Base)
        self._lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._lbl_image.setScaledContents(True)

        # The scroll area used to handle images larger than the widget size

        self._scroll_area = ImageScrollArea(scrollContentsBy=self._afterRasterScroll)

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

        # The image generated from the raw raster data.
        self._image = None


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

        if raster_data is not None:
            rgb_bands = find_display_bands(raster_data)
            if len(rgb_bands) == 1:
                rgb_bands = rgb_bands * 3
            elif len(rgb_bands) != 3:
                raise ValueError(f'Raster data has an unexpected number of '
                                  'display bands:  {rgb_bands}')

            self._red_band   = rgb_bands[0]
            self._green_band = rgb_bands[1]
            self._blue_band  = rgb_bands[2]

        self.update_display_image()


    def _extract_band_for_display(self, band_index):
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
            self._red_data = self._extract_band_for_display(self._red_band)

        if self._green_data is None or ImageColors.GREEN in colors:
            # print('Regenerating green data')
            self._green_data = self._extract_band_for_display(self._green_band)

        if self._blue_data is None or ImageColors.BLUE in colors:
            # print('Regenerating blue data')
            self._blue_data = self._extract_band_for_display(self._blue_band)

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
        self._image = QImage(img_data,
            self._raster_data.get_width(), self._raster_data.get_height(),
            QImage.Format_RGB32)

        self._lbl_image.setPixmap(QPixmap.fromImage(self._image))
        self._lbl_image.adjustSize()
        self._scroll_area.setVisible(True)

    def get_scale(self):
        ''' Returns the current scale factor for the raster image. '''
        return self._scale_factor

    def scale_image(self, factor):
        '''
        Scales the raster image by the specified factor.  Note that this is an
        absolute operation, not an incremental operation; repeatedly calling
        this function with a factor of 0.5 will not halve the size of the image
        each call.  Rather, the image will simply be set to 0.5 of its original
        size.

        If there is no image data, this is a no-op.
        '''

        if self._raster_data is None:
            return

        # Only scale the image if the scale-factor is changing.
        if factor != self._scale_factor:
            self._scale_factor = factor
            self._lbl_image.resize(self._image.size() * self._scale_factor)
            self._emit_viewport_change()

    def scale_image_to_fit(self, mode=ScaleToFitMode.FIT_BOTH_DIMENSIONS):
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
        # print(f'area_size = {area_size}')

        # TODO(donnie):  The first time fitting one dimension seems to be buggy,
        #     since scrollbar sizes haven't been finalized.  Workaround is to
        #     do this multiple times, but what a drag.
        sb_width = self._scroll_area.verticalScrollBar().size().width()
        sb_height = self._scroll_area.horizontalScrollBar().size().height()
        # print(f'sb_width = {sb_width}')
        # print(f'sb_height = {sb_height}')

        # TODO(donnie):  Still some buggy behavior when the widget size allows
        #     the entire image to fit, but specific scale options are chosen.
        #     We may want to have another "max_size=True" kind of keyword arg
        #     for this function.

        if mode == ScaleToFitMode.FIT_HORIZONTAL:
            # Calculate new scale factor for fitting the image horizontally,
            # based on the maximum viewport size.
            new_factor = area_size.width() / self._raster_data.get_width()

            if self._raster_data.get_height() * new_factor > area_size.height():
                # At the proposed scale, the data won't fit vertically, so we
                # need to recalculate the scale factor to account for the
                # vertical scrollbar that will show up.
                new_factor = (area_size.width() - sb_width) / self._raster_data.get_width()

        elif mode == ScaleToFitMode.FIT_VERTICAL:
            # Calculate new scale factor for fitting the image vertically,
            # based on the maximum viewport size.
            new_factor = area_size.height() / self._raster_data.get_height()

            if self._raster_data.get_width() * new_factor > area_size.width():
                # At the proposed scale, the data won't fit horizontally, so we
                # need to recalculate the scale factor to account for the
                # horizontal scrollbar that will show up.
                new_factor = (area_size.height() - sb_height) / self._raster_data.get_height()

        elif mode == ScaleToFitMode.FIT_ONE_DIMENSION:
            # Unless the image is the exact same aspect ratio as the viewing
            # area, one scrollbar will be visible.
            r_aspectratio = self._raster_data.get_width() / self._raster_data.get_height()
            a_aspectratio = area_size.width() / area_size.height()

            if r_aspectratio == a_aspectratio:
                # Can use either width or height to do the calculation.
                new_factor = area_size.width() / self._raster_data.get_width()

            else:
                new_factor = max(
                    (area_size.width() - sb_width) / self._raster_data.get_width(),
                    (area_size.height() - sb_height) / self._raster_data.get_height()
                )

        elif mode == ScaleToFitMode.FIT_BOTH_DIMENSIONS:
            # The image will fit in both dimensions, so both scrollbars will be
            # hidden after scaling.
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


    def _emit_viewport_change(self):
        ''' A helper that emits the viewport-changed event. '''
        self.viewport_change.emit(self.get_visible_region())


    def resizeEvent(self, event):
        '''
        Override the QtWidget resizeEvent() virtual method to fire an event that
        the visible region of the raster-view has changed.
        '''
        self._emit_viewport_change()

    def _afterRasterScroll(self, widget, dx, dy):
        '''
        This function is called when the scroll-area moves around.  Fire an
        event that the visible region of the raster-view has changed.
        '''
        self._emit_viewport_change()

    def get_visible_region(self):
        '''
        This method reports the visible region of the raster data-set, in raster
        data-set coordinates.  The returned value is a Qt QRect, with the
        integer (x, y, width, height) values indicating the visible region.

        If the raster-view has no data set then None is returned.
        '''

        if self._raster_data is None:
            return None

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

    def make_point_visible(self, x, y):
        # Scroll the scroll-area to make the specified point visible.  The point
        # also needs scaled based on the current scale factor.  Finally, specify
        # a margin that's half the viewing area, so that the point will be in
        # the center of the area, if possible.
        self._scroll_area.ensureVisible(
            x * self._scale_factor, y * self._scale_factor,
            self._scroll_area.viewport().width() / 2,
            self._scroll_area.viewport().height() / 2
        )


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
        return (position / self._scale_factor).toPoint()


    def _onRasterMouseClick(self, widget, mouse_event):
        '''
        When the display image is clicked on, this method gets invoked, and it
        translates the click event's coordinates into the location on the
        raster data set.
        '''
        # Map the coordinate of the mouse-event to the actual raster-image
        # pixel that was clicked, then emit a signal.
        r_coord = self._image_coord_to_raster_coord(mouse_event.localPos())
        self.mouse_click.emit(r_coord, mouse_event)


    def _afterRasterPaint(self, widget, paint_event):
        # For subclasses to override, if they wish.
        pass
