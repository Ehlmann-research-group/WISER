import sys
from enum import Enum, IntFlag

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np

from raster.dataset import RasterDataSet, find_display_bands

from stretch import StretchBase, StretchLinear
from gui.stretch_builder import StretchBuilder
from gui.stretch_builder_ui import Ui_Dialog_stretchBuilder


class ImageColors(IntFlag):
    NONE = 0

    RED = 1
    GREEN = 2
    BLUE = 4

    RGB = 7


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
        self.setMouseTracking(True)

    def mousePressEvent(self, mouse_event):
        if 'mousePressEvent' in self._forward:
            self._forward['mousePressEvent'](self, mouse_event)

    def mouseReleaseEvent(self, mouse_event):
        if 'mouseReleaseEvent' in self._forward:
            self._forward['mouseReleaseEvent'](self, mouse_event)

    def mouseMoveEvent(self, mouse_event):
        if 'mouseMoveEvent' in self._forward:
            self._forward['mouseMoveEvent'](self, mouse_event)

    def keyPressEvent(self, key_event):
        if 'keyPressEvent' in self._forward:
            self._forward['keyPressEvent'](self, key_event)

    def keyReleaseEvent(self, key_event):
        if 'keyReleaseEvent' in self._forward:
            self._forward['keyReleaseEvent'](self, key_event)

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
    '''

    _stretchBuilderButton = None
    _stretchBuilder = None
    _stretches = [None, None, None]

    def __init__(self, parent=None, forward=None):
        super().__init__(parent=parent)

        if forward == None:
            forward = {}

        # Initialize fields in the object
        self._clear_members()
        self._scale_factor = 1.0

        # The widget used to display the image data

        self._image_widget = ImageWidget('(no data)', **forward)
        self._image_widget.setBackgroundRole(QPalette.Base)
        self._image_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._image_widget.setScaledContents(True)
        self._image_widget.setFocusPolicy(Qt.ClickFocus)

        # The scroll area used to handle images larger than the widget size

        self._scroll_area = ImageScrollArea(**forward)
        self._scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scroll_area.setBackgroundRole(QPalette.Dark)
        self._scroll_area.setWidget(self._image_widget)
        self._scroll_area.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        # Set up the layout

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        # layout.addWidget(self.image_toolbar)
        # layout.setMenuBar(self.image_toolbar)
        layout.addWidget(self._scroll_area)

        # Hack in a button to launch a StretchBuilder
        self._stretchBuilderButton = QPushButton("Stretch Builder")
        self._stretchBuilderButton.clicked.connect(self._show_stretchBuilder)
        layout.addWidget(self._stretchBuilderButton)
        self.setLayout(layout)

        self._stretchBuilder = StretchBuilder(self)
        self._stretchBuilder.stretchChanged.connect(self.set_stretches)

    @Slot()
    def _show_stretchBuilder(self):
        self._stretchBuilder.show()

    @Slot()
    def _hide_stretchBuilder(self):
        self._stretchBuilder.hide()

    def get_stretches(self):
        return self._stretches
    
    @Slot(StretchBase)
    def set_stretches(self, stretches: list):
        self._stretches = stretches
        self.update_display_image()

    def _clear_members(self):
        '''
        A helper function to clear all raster dataset members when the dataset
        changes.  This way we don't accidentally leave anything out.
        '''

        self._raster_data = None
        self._display_bands = None

        # These members are for storing the components of the raster data, so
        # that assembling the image is faster when only one color's band
        # changes.

        self._red_data = None
        self._green_data = None
        self._blue_data = None
        self._img_data = None

        # The image generated from the raw raster data.
        self._image = None


    def set_raster_data(self, raster_data, display_bands):
        '''
        Specify a raster data-set to display in the raster-view widget.  A value
        of None causes the raster-view to display nothing.
        '''
        if raster_data is not None and not isinstance(raster_data, RasterDataSet):
            raise ValueError('raster_data must be a RasterDataSet object')

        self._clear_members()

        self._raster_data = raster_data
        self._display_bands = display_bands

        if raster_data is not None:
            assert len(self._display_bands) in [1, 3], \
                f'Raster data has an unsupported number of display bands:  {rgb_bands}'

        self.update_display_image()


    def get_raster_data(self):
        '''
        Returns the current raster dataset that is being displayed, or None if
        no dataset is currently being displayed.
        '''
        return self._raster_data


    def get_display_bands(self):
        '''
        Returns a copy of the display-band list, which will have one element for
        grayscale, or three elements for RGB.
        '''
        return list(self._display_bands)


    def set_display_bands(self, display_bands):
        if len(display_bands) not in [1, 3]:
            raise ValueError('display_bands must be a list of 1 or 3 ints')

        # Figure out what colors changed, so that we only have to update the
        # parts of the image that are required.
        changed = ImageColors.NONE

        if len(self._display_bands) != len(display_bands):
            # Assume all 3 colors changed
            changed = ImageColors.RGB

        elif len(self._display_bands) == 1:
            if self._display_bands[0] != display_bands[0]:
                changed = ImageColors.RGB

        else:
            assert len(self._display_bands) == 3

            if self._display_bands[0] != display_bands[0]:
                changed |= ImageColors.RED

            if self._display_bands[1] != display_bands[1]:
                changed |= ImageColors.GREEN

            if self._display_bands[2] != display_bands[2]:
                changed |= ImageColors.BLUE

        self._display_bands = display_bands
        self.update_display_image(colors=changed)


    def extract_band_for_display(self, band_index):
        '''
        Extracts the specified band of raster data for display in an RGB image
        in the user interface.  This operation is done with numpy so that it can
        be completed as efficiently as possible.

        The function returns a numpy array with np.float32 elements in the range
        of 0.0 .. 1.0, unless the input is already np.float64, in which case the
        type is left as np.float64.
        '''
        band_data = self._raster_data.get_band_data(band_index)

        # TODO(donnie):  Almost certainly, we will need a much more
        #     sophisticated way of specifying how the band data is transformed.
        #     But for now, handle all but complex data linearly
        if band_data.dtype == np.float32 or band_data.dtype == np.float64:
            np.clip(band_data, 0., 1., out=band_data)

        elif band_data.dtype == np.uint32 or band_data.dtype == np.int32:
            # fake a linear stretch by simply ignoring the low bytes
            band_data = (band_data >> 24).astype(np.float32) / 255.

        elif band_data.dtype == np.uint16 or band_data.dtype == np.int16:
            # fake a linear stretch by simply ignoring the low byte
            band_data = (band_data >> 8).astype(np.uint32) / 255.

        elif band_data.dtype == np.uint8 or band_data.dtype == np.int8:
            band_data = band_data.astype(np.uint32) / 255.
            
        else:
            print("Data type {} not currently supported".format(band_data.dtype))
            raise NotImplementedError

        return band_data


    def update_display_image(self, colors=ImageColors.RGB):
        if self._raster_data is None:
            # No raster data to display - clear the view.
            self._image_widget.clear()
            self._image_widget.setText('(no data)')
            # self.lbl_image.adjustSize()
            return

        # print("Extracting raw band data")

        # Only generate (or regenerate) each color plane if we don't already
        # have data for it, and if we aren't told to explicitly regenerate it.

        if len(self._display_bands) == 3:
            if self._red_data is None or ImageColors.RED in colors:
                self._red_data = self.extract_band_for_display(self._display_bands[0])
                if self._stretches[0]:
                    self._red_data = self._red_data.copy()
                    self._red_data = self._stretches[0].apply(self._red_data)
                self._red_data = (self._red_data * 255.).astype(np.uint32)

            if self._green_data is None or ImageColors.GREEN in colors:
                self._green_data = self.extract_band_for_display(self._display_bands[1])
                if self._stretches[1]:
                    self._green_data = self._green_data.copy()
                    self._green_data = self._stretches[1].apply(self._green_data)
                self._green_data = (self._green_data * 255.).astype(np.uint32)

            if self._blue_data is None or ImageColors.BLUE in colors:
                self._blue_data = self.extract_band_for_display(self._display_bands[2])
                if self._stretches[2]:
                    self._blue_data = self._blue_data.copy()
                    self._blue_data = self._stretches[2].apply(self._blue_data)
                self._blue_data = (self._blue_data * 255.).astype(np.uint32)

        else:
            assert len(self._display_bands) == 1

            # Grayscale:  We can extract the band data once, and use it for all
            # three colors.
            data = self.extract_band_for_display(self._display_bands[0])
            if self._stretches[0]:
                data = data.copy()
                data = self._stretch.apply(data)
            data = (data * 255.).astype(np.uint32)

            if self._red_data is None or ImageColors.RED in colors:
                self._red_data = data

            if self._green_data is None or ImageColors.GREEN in colors:
                self._green_data = data

            if self._blue_data is None or ImageColors.BLUE in colors:
                self._blue_data = data

        # print("Converting raw data to RGB color data")

        img_data = (self._red_data << 16 | self._green_data << 8 | self._blue_data) | 0xff000000
        if isinstance(img_data, np.ma.MaskedArray):
            img_data.fill_value = 0xff000000

        # TODO(donnie):  I don't know why the tostring() is required here, but
        #     it seems to be required for making the QImage when we use GDAL.
        #     Note - may be because of the numpy MaskedArray...
        img_data = img_data.tostring()
        # This is necessary because the QImage doesn't take ownership of the
        # data we pass it, and if we drop this reference to the data then Python
        # will reclaim the memory and Qt will start to display garbage.
        self._img_data = img_data
        # print('stored:')
        # print(self._img_data)

        # This is the 100% scale QImage of the data.
        self._image = QImage(img_data,
            self._raster_data.get_width(), self._raster_data.get_height(),
            QImage.Format_RGB32)

        self._update_scaled_image()


    def _update_scaled_image(self, old_scale_factor=None):
        # Update the scaled version of the image.
        scaled_image = self._image.scaled(
            self._raster_data.get_width() * self._scale_factor,
            self._raster_data.get_height() * self._scale_factor,
            Qt.IgnoreAspectRatio, Qt.FastTransformation)

        # Update the image that the label is displaying.
        pixmap = QPixmap.fromImage(scaled_image)
        self._image_widget.setPixmap(pixmap)
        self._image_widget.adjustSize()
        self._scroll_area.setVisible(True)

        # Need to process queued events now, since the image-widget has changed
        # size, and it needs to report a resize-event before the scrollbars will
        # update to the new size.
        QCoreApplication.processEvents()

        if old_scale_factor is not None and old_scale_factor != self._scale_factor:
            # The scale is changing, so update the scrollbars to ensure that the
            # image stays centered in the viewport area.

            scale_change = self._scale_factor / old_scale_factor

            self._update_scrollbar(self._scroll_area.horizontalScrollBar(),
                self._scroll_area.viewport().width(), scale_change)

            self._update_scrollbar(self._scroll_area.verticalScrollBar(),
                self._scroll_area.viewport().height(), scale_change)


    def _update_scrollbar(self, scrollbar, view_size, scale_change):
        # The scrollbar's value will be scaled by the scale_change value.  For
        # example, if the original scale was 100% and the new scale is 200%,
        # the scale_change value will be 200%/100%, or 2.  To keep the same area
        # within the scroll-area's viewport, the scrollbar's value needs to be
        # multiplied by the scale_change.

        # That said, the

        view_diff = view_size * (scale_change - 1)
        scrollbar.setValue(scrollbar.value() * scale_change + view_diff / 2)


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
            old_factor = self._scale_factor
            self._scale_factor = factor
            self._update_scaled_image(old_scale_factor=old_factor)

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


    def update(self):
        '''
        Override the QWidget update() function to make sure that the internal
        widgets are updated.  This is necessary since the raster-view is
        comprised of multiple widgets.
        '''
        super().update()
        self._image_widget.update()


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


    def make_point_visible(self, x, y, margin=0.5):
        '''
        Make the specified (x, y) coordinate of the raster dataset visible in
        the view.

        The optional margin argument controls where the pixel will appear in the
        view after the operation, and may range between 0 and 0.5.  The default
        value is 0.5.  The value corresponds to the percentage of the view's
        size that will be used as a margin around the specified coordinate.  Of
        course, the requested margin is not always achievable, for example, if
        the pixel to display is on the edge of the raster image.  But, the view
        widget will do the best that it can.

        *   A value of 0.5 will cause the pixel to appear in the center of the
            view, if possible; in other words, 50% of the view's size will be
            used as a margin.

        *   A value of 0 will allow the pixel to appear on the edge of the view,
            because the requested margin size is 0% of the view's size.
        '''

        if margin < 0 or margin > 0.5:
            raise ValueError(f'margin must be in the range [0, 0.5, got {margin}]')

        # Scroll the scroll-area to make the specified point visible.  The point
        # also needs scaled based on the current scale factor.  Finally, specify
        # a margin that's half the viewing area, so that the point will be in
        # the center of the area, if possible.
        self._scroll_area.ensureVisible(
            x * self._scale_factor, y * self._scale_factor,
            self._scroll_area.viewport().width() * margin,
            self._scroll_area.viewport().height() * margin
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


    def image_coord_to_raster_coord(self, position):
        '''
        Takes a position in screen space as a QPointF object, and translates it
        into a 2-tuple containing the (X, Y) coordinates of the position within
        the raster data set.
        '''
        # Scale the screen position into the dataset's coordinate system.
        scaled = position / self._scale_factor

        # Convert to an integer coordinate.  Can't use QPointF.toPoint() because
        # it rounds to the nearest point, and we just want truncation/floor.
        return QPoint(int(scaled.x()), int(scaled.y()))
