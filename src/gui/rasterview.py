import sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np

from .constants import ImageColors
from .rgb_selector import RGBSelector
from .spectrum_plot import SpectrumPlot


def add_toolbar_action(toolbar, icon_path, text, parent, shortcut=None):
    '''
    A helper function to set up a toolbar action using the common configuration
    used for these actions.
    '''
    act = QAction(QIcon(icon_path), text, parent)

    if shortcut is not None:
        act.setShortcuts(shortcut)

    toolbar.addAction(act)
    return act


class RasterImage(QLabel):
    '''
    A simple subclass of QLabel used for displaying a raster image.  The main
    reason we subclass QLabel is simply to forward mouse-click events from the
    image to the RasterView, which then can scale and interpret them properly.
    '''
    pixel_clicked = Signal(QMouseEvent)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def mouseReleaseEvent(self, mouse_event):
        self.pixel_clicked.emit(mouse_event)


class RasterView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # A toolbar for the raster view
        self.init_toolbar()
        # self.image_toolbar.

        # The image that the raster view uses to display its data
        self.lbl_image = RasterImage()
        self.lbl_image.setBackgroundRole(QPalette.Base)
        self.lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_image.setScaledContents(True)
        self.lbl_image.pixel_clicked.connect(self.raster_pixel_clicked)

        # The scroll area used to handle images larger than the widget size
        self.scroll_area = QScrollArea()
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area.setBackgroundRole(QPalette.Dark)
        self.scroll_area.setWidget(self.lbl_image)
        self.scroll_area.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        # Set up the layout

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        # layout.addWidget(self.image_toolbar)
        layout.setMenuBar(self.image_toolbar)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

        # Initialize fields in the object

        self.raster_data = None
        self.scale_factor = 1.0

        self.red_band = 0
        self.green_band = 0
        self.blue_band = 0

        self.red_data = None
        self.green_data = None
        self.blue_data = None

        # Initialize some helper widgets

        self.rgb_selector = RGBSelector(self)
        self.rgb_selector.band_changed.connect(self.rgb_band_changed)

        self.spectrum_plot = SpectrumPlot(self)


    def init_toolbar(self):
        print('Initializing toolbar.')
        self.image_toolbar = QToolBar(self.tr('Image'), self)

        # Zoom In
        self.act_zoom_in = add_toolbar_action(self.image_toolbar,
            'resources/zoom-in.svg', self.tr('Zoom In'), self, QKeySequence.ZoomIn)
        self.act_zoom_in.triggered.connect(self.zoom_in)

        # Zoom Out
        self.act_zoom_out = add_toolbar_action(self.image_toolbar,
            'resources/zoom-out.svg', self.tr('Zoom Out'), self, QKeySequence.ZoomOut)
        self.act_zoom_out.triggered.connect(self.zoom_out)

        # Zoom to Actual Size
        self.act_zoom_to_actual = add_toolbar_action(self.image_toolbar,
            'resources/zoom-to-actual.svg', self.tr('Zoom to Actual Size'), self, None)
        self.act_zoom_to_actual.triggered.connect(self.zoom_to_actual)

        # Zoom to Fit
        self.act_zoom_to_fit = add_toolbar_action(self.image_toolbar,
            'resources/zoom-to-fit.svg', self.tr('Zoom to Fit'), self, None)
        self.act_zoom_to_fit.triggered.connect(self.zoom_to_fit)

        self.image_toolbar.addSeparator()

        # Choose RGB Bands
        self.act_choose_rgb_bands = add_toolbar_action(self.image_toolbar,
            'resources/choose-colors.svg', self.tr('Choose RGB Bands'), self, None)
        self.act_choose_rgb_bands.triggered.connect(self.choose_colors)

        self.image_toolbar.addSeparator()

        # Show spectral data for a specific band
        self.act_show_spectrum = add_toolbar_action(self.image_toolbar,
            'resources/target.svg', self.tr('Show Pixel Spectrum'), self, None)
        self.act_show_spectrum.triggered.connect(self.show_spectrum)


    def set_raster_data(self, raster_data, rgb_bands=(0, 0, 0)):
        self.raster_data = raster_data
        self.scale_factor = 1.0

        self.red_band   = rgb_bands[0]
        self.green_band = rgb_bands[1]
        self.blue_band  = rgb_bands[2]

        self.update_display_image()


    def extract_band_for_display(self, band_index):
        '''
        Extracts the specified band of raster data for display in an RGB image
        in the user interface.  This operation is done with numpy so that it can
        be completed as efficiently as possible.

        The function returns a numpy array with np.uint32 elements in the range
        of 0..255.
        '''
        band_data = self.raster_data.get_band_data(band_index)

        # TODO(donnie):  Almost certainly, we will need a much more
        #     sophisticated way of specifying how the band data is transformed.
        band_data = (band_data * 255 + 30).clip(0, 255).astype(np.uint32)

        return band_data


    def update_display_image(self, colors=ImageColors.RGB):
        if self.raster_data is None:
            # No raster data to display - clear the view.
            self.lbl_image.clear()
            # self.lbl_image.adjustSize()
            return

        # print("Extracting raw band data")

        # Only generate (or regenerate) each color plane if we don't already
        # have data for it, and if we aren't told to explicitly regenerate it.

        if self.red_data is None or ImageColors.RED in colors:
            # print('Regenerating red data')
            self.red_data = self.extract_band_for_display(self.red_band)

        if self.green_data is None or ImageColors.GREEN in colors:
            # print('Regenerating green data')
            self.green_data = self.extract_band_for_display(self.green_band)

        if self.blue_data is None or ImageColors.BLUE in colors:
            # print('Regenerating blue data')
            self.blue_data = self.extract_band_for_display(self.blue_band)

        # print("Converting raw data to RGB color data")

        # TODO(donnie):  I don't know why the tostring() is required here, but
        #     it seems to be required for making the QImage when we use GDAL.
        img_data = (self.red_data << 16 | self.green_data << 8 | self.blue_data) | 0xff000000
        img_data = img_data.tostring()
        # This is necessary because the QImage doesn't take ownership of the
        # data we pass it, and if we drop this reference to the data then Python
        # will reclaim the memory and Qt will start to display garbage.
        self.img_data = img_data

        # print("Making QImage")
        image = QImage(img_data,
            self.raster_data.get_width(), self.raster_data.get_height(),
            QImage.Format_RGB32)

        # print("QImage is complete")

        # print("Displaying image.")
        self.lbl_image.setPixmap(QPixmap.fromImage(image))
        self.lbl_image.adjustSize()
        self.scroll_area.setVisible(True)
        # print("Done.")


    def scale_image(self, factor):
        self.scale_factor = factor
        self.lbl_image.resize(factor * self.raster_data.get_width(),
                              factor * self.raster_data.get_height())


    @Slot()
    def zoom_in(self, evt):
        self.scale_image(self.scale_factor * 1.25)
        # TODO:  Disable zoom-in if too zoomed

    @Slot()
    def zoom_out(self, evt):
        self.scale_image(self.scale_factor * 0.8)
        # TODO:  Disable zoom-out if too un-zoomed

    @Slot()
    def zoom_to_actual(self, evt):
        self.scale_image(1.0)

    @Slot()
    def zoom_to_fit(self, evt):
        # Figure out the appropriate scale factor, then do it
        new_factor = min(
            self.scroll_area.width() / self.raster_data.get_width(),
            self.scroll_area.height() / self.raster_data.get_height()
        )
        self.scale_image(new_factor)

    @Slot()
    def choose_colors(self, evt):
        if not self.rgb_selector.isVisible():
            self.rgb_selector.set_slider_ranges(self.raster_data.num_bands())
            self.rgb_selector.set_slider_values(self.red_band, self.green_band, self.blue_band)
            self.rgb_selector.setVisible(True)

        else:
            self.rgb_selector.setVisible(False)

    # TODO(donnie):  Should be Slot(ImageColors, int), but causes PySide2 to
    #     crash at startup.
    @Slot(int, int)
    def rgb_band_changed(self, color, band_index):
        # print(f'Color:  {color}\tNew band:  {band_index}')

        # TODO(donnie):  See above TODO
        color = ImageColors(color)

        if color == ImageColors.RED:
            self.red_band = band_index

        elif color == ImageColors.GREEN:
            self.green_band = band_index

        elif color == ImageColors.BLUE:
            self.blue_band = band_index

        else:
            print(f'WARNING:  Unrecognized color # {color}')

        self.update_display_image(colors=color)

    @Slot(QMouseEvent)
    def raster_pixel_clicked(self, mouse_event):
        # Map the coordinate of the mouse-event to the actual raster-image
        # pixel that was clicked.
        # TODO(donnie):  I could move this calculation inside the if statement,
        #     but I have a feeling we may want to do other things when raster
        #     pixels are clicked.
        raster_point = (mouse_event.localPos() / self.scale_factor).toPoint()
        (raster_x, raster_y) = raster_point.toTuple()
        # print(f'Raster pixel clicked:  {raster_point}')

        if self.spectrum_plot.isVisible():
            # TODO(donnie):  Extract the spectrum at the specified pixel, then
            #     display it in the spectrum plot.
            spectrum = self.raster_data.get_all_bands_at(raster_x, raster_y)
            self.spectrum_plot.add_spectrum(spectrum)


    @Slot()
    def show_spectrum(self, evt):
        if not self.spectrum_plot.isVisible():
            self.spectrum_plot.clear()
            self.spectrum_plot.setVisible(True)

        else:
            self.rgb_selector.setVisible(False)
