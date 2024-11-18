import enum
import logging
import sys
import time

from typing import Dict, List, Optional, Tuple, Union

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np
from matplotlib import cm

from .util import get_painter, scale_qpoint_by_float

from wiser.raster.dataset import RasterDataSet, find_display_bands
from wiser.raster.stretch import StretchBase, StretchLinear
from wiser.raster.utils import normalize_ndarray

from wiser.gui.app_state import ApplicationState
from wiser.gui.rasterview_metadata import RasterViewMetaData

logger = logging.getLogger(__name__)

def get_new_display_band_data(raster_data: RasterDataSet,
                              display_bands_raw: Dict[int, np.ndarray], 
                              display_bands: Tuple[int],
                              sample_factor: int = 4):
    """
    Updates the display_bands_raw dict based on the comparison between display_bands
    and the keys of display_bands_raw. For matching tuples, it keeps the existing data. 
    It fetches the band data from raster_data if there is no array data or no matches

    Parameters:
        raster_data (RasterDataSet): The raster dataset object to fetch band data.
        display_bands_raw (list): A list of the band index with its correspoding numpy array
        display_bands (tuple): The current display bands.

    Returns:
        dict: Updated display_bands_raw dict that contains possibly new arrays
    
    Edge cases: 
        - display_bands_raw is [None, None, None]
        - current_display_bands is None
    """
    # Initialize the updated bands dictionary
    updated_bands = {}

    # If display_bands is None, return an empty updated_bands dictionary
    if display_bands is None:
        return updated_bands

    # Loop through the current bands in display_bands
    for curr_band in display_bands:
        if display_bands_raw is not None and curr_band in display_bands_raw:
            # Check if the corresponding value in display_bands_raw is None
            if display_bands_raw[curr_band] is None:
                # Fetch the band data using raster_data
                # updated_bands[curr_band] = raster_data.get_band_data(curr_band)
                print(f"curr_band: {curr_band}")
                arr = raster_data.sample_band_data(curr_band, sample_factor)
                print(f"type(arr): {type(arr)}")
                arr = arr.copy()
                print(f"copy arr type: {type(arr)}")
                print(f"copy arr: {arr}")
                updated_bands[curr_band] = arr
            else:
                # Use the existing data from display_bands_raw
                print(f"Getting past bands: type: {type(display_bands_raw[curr_band])}")
                print(f"Getting past bands: {display_bands_raw[curr_band]}")
                updated_bands[curr_band] = display_bands_raw[curr_band]
        else:
            # Fetch the band data if the band is not in display_bands_raw
            # updated_bands[curr_band] = raster_data.get_band_data(curr_band)
            print(f"curr_band2: {curr_band}")
            arr = raster_data.sample_band_data(curr_band, sample_factor)
            print(f"type(arr): {type(arr)}")
            print(f"copy arr type: {type(arr)}")
            arr = arr.copy()
            print(f"copy arr: {arr}")
            updated_bands[curr_band] = raster_data.sample_band_data(curr_band, sample_factor).copy()

    # Return the updated bands dictionary
    return updated_bands

    # Initialize the updated bands dict

    # Loop through the current bands in display_bands
        # If one of these bands is in display_bands_raw we, we check to see if display_bands_raw
        # is none. If it is none we get the array data using raster_data.get_band_data(curr_band)

        # If its not none we add the array to the updated dict

        # If one of the bends is not in display_bands, we just get
        # the band with raster_data.get_band_data(curr_band)
    
    # We return the updated bands dict

    # # Initialize a new list to hold the updated band data
    # updated_bands_raw = []
    
    # for new_index, new_band in enumerate(new_display_bands):
    #     # Check if the current new_band exists in the current_display_bands tuple
    #     if new_band in current_display_bands:
    #         # Find the index of the matching band in current_display_bands
    #         current_index = current_display_bands.index(new_band)
    #         # Use the corresponding data from display_bands_raw
    #         if display_bands_raw[current_index] is None:
    #             print(f"display_bands_raw[current_index] is none: {display_bands_raw[current_index]}")
    #             updated_bands_raw.append(raster_data.get_band_data(new_band))
    #         else:
    #             updated_bands_raw.append(display_bands_raw[current_index])
    #     else:
    #         print(f"New band not in current display bands: new band: {new_band}, curr_display: {current_display_bands}")
    #         # Fetch new data from raster_data for the non-matching band
    #         updated_bands_raw.append(raster_data.get_band_data(new_band))
    
    # return updated_bands_raw


def make_channel_image_with_band(band: np.ndarray, stretch: StretchBase = None) -> np.ndarray:
    '''
    Given a raster data set, band index, and optional contrast stretch object,
    this function generates color channel data into a NumPy array. Elements in
    the output array will be in the range [0, 255].
    '''
    # Extract the raw band data and associated statistics from the data set.
    start_time = time.perf_counter()
    temp_data = band.copy()
    print(f"Temp data shape: {temp_data.shape}")
    end_time = time.perf_counter()
    print(f"Time to copy band data: {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    temp_data = temp_data.astype(np.float32, copy=False)
    end_time = time.perf_counter()
    print(f"Time to cast data to float32: {end_time - start_time:.6f} seconds")

    # If a stretch is specified for the channel, apply it to the normalized band data.
    if stretch is not None:
        start_time = time.perf_counter()
        stretch.apply(temp_data)
        end_time = time.perf_counter()
        print(f"Time to apply stretch: {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    finite_vals = temp_data[np.isfinite(temp_data)]
    end_time = time.perf_counter()
    print(f"Time to extract finite values: {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    normalize_ndarray(temp_data, minval=finite_vals.min(), maxval=finite_vals.max(), in_place=True)
    end_time = time.perf_counter()
    print(f"Time to normalize array: {end_time - start_time:.6f} seconds")

    # Clip the data to be in the range [0.0, 1.0]. This should not remove NaNs.
    start_time = time.perf_counter()
    np.clip(temp_data, 0.0, 1.0, out=temp_data)
    end_time = time.perf_counter()
    print(f"Time to clip data: {end_time - start_time:.6f} seconds")
    
    # Finally, convert the normalized (and possibly stretched) band data into a color channel with values in the range [0, 255].
    start_time = time.perf_counter()
    temp_data = (temp_data * 255.0)
    end_time = time.perf_counter()
    print(f"Time to scale data to [0, 255]: {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    temp_data = temp_data.astype(np.uint8, copy=False)
    end_time = time.perf_counter()
    print(f"Time to cast data to uint8: {end_time - start_time:.6f} seconds")

    return temp_data

def make_channel_image(dataset: RasterDataSet, band: int, stretch: StretchBase = None) -> np.ndarray:
    '''
    Given a raster data set, band index, and optional contrast stretch object,
    this function generates color channel data into a NumPy array. Elements in
    the output array will be in the range [0, 255].
    '''
    # Extract the raw band data and associated statistics from the data set.
    start_time = time.perf_counter()
    temp_data = dataset.sample_band_data(band, 2)
    print(f"!!!!!!!!!!nan min before copy: {np.nanmin(temp_data)}")
    temp_data = temp_data.copy()
    print(f"!!!!!!!!!!nan min after copy: {np.nanmin(temp_data)}")
    end_time = time.perf_counter()
    print(f"Time to copy band data: {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    temp_data = temp_data.astype(np.float32, copy=False)
    print(f"!!!!!!!!!!nan min after astype: {np.nanmin(temp_data)}")
    end_time = time.perf_counter()
    print(f"Time to cast data to float32: {end_time - start_time:.6f} seconds")

    # If a stretch is specified for the channel, apply it to the normalized band data.
    if stretch is not None:
        start_time = time.perf_counter()
        stretch.apply(temp_data)
        end_time = time.perf_counter()
        print(f"Time to apply stretch: {end_time - start_time:.6f} seconds")
    print(f"!!!!!!!!!!nan min after stretch: {np.nanmin(temp_data)}")

    start_time = time.perf_counter()
    finite_vals = temp_data[np.isfinite(temp_data)]
    end_time = time.perf_counter()
    print(f"Time to extract finite values: {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    normalize_ndarray(temp_data, minval=finite_vals.min(), maxval=finite_vals.max(), in_place=True)
    print(f"!!!!!!!!!!!!!nan min after normalize_ndarray: {np.nanmin(temp_data)}")
    end_time = time.perf_counter()
    print(f"Time to normalize array: {end_time - start_time:.6f} seconds")

    # Clip the data to be in the range [0.0, 1.0]. This should not remove NaNs.
    start_time = time.perf_counter()
    np.clip(temp_data, 0.0, 1.0, out=temp_data)
    end_time = time.perf_counter()
    print(f"Time to clip data: {end_time - start_time:.6f} seconds")
    
    # Finally, convert the normalized (and possibly stretched) band data into a color channel with values in the range [0, 255].
    start_time = time.perf_counter()
    temp_data = (temp_data * 255.0)
    end_time = time.perf_counter()
    print(f"Time to scale data to [0, 255]: {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    temp_data = temp_data.astype(np.uint8, copy=False)
    end_time = time.perf_counter()
    print(f"Time to cast data to uint8: {end_time - start_time:.6f} seconds")

    return temp_data

# def make_channel_image(dataset: RasterDataSet, band: int, stretch: StretchBase = None) -> np.ndarray:
#     '''
#     Given a raster data set, band index, and optional contrast stretch object,
#     this function generates color channel data into a NumPy array. Elements in
#     the output array will be in the range [0, 255].
#     '''
#     # Extract the raw band data and associated statistics from the data set.
#     temp_data = dataset.get_band_data(band).copy()
#     temp_data = temp_data.astype(np.float32, copy=False)
#     # If a stretch is specified for the channel, apply it to the normalized band data.
#     if stretch is not None:
#         stretch.apply(temp_data)
#     finite_vals = temp_data[np.isfinite(temp_data)]
#     normalize_ndarray(temp_data, minval=finite_vals.min(), maxval=finite_vals.max(), in_place=True)

#     # Clip the data to be in the range [0.0, 1.0]. This should not remove NaNs.
#     np.clip(temp_data, 0.0, 1.0, out=temp_data)
    
#     # Finally, convert the normalized (and possibly stretched) band data into a color channel with values in the range [0, 255].
#     temp_data = (temp_data * 255.0)
#     temp_data = temp_data.astype(np.uint8, copy=False)
#     return temp_data


def make_rgb_image(channels: List[np.ndarray]) -> np.ndarray:
    '''
    Given a list of 3 color channels of the same dimensions, this function
    combines them together into an RGB image.  The first, second and third
    channels are correspondingly used for the red, green and blue channels of
    the resulting image.

    An exception is raised if:
    *   A number of channels is specified other than 3.
    *   Any channel is not of type ``np.uint8``, ``np.uint16``, or ``np.uint32``.
    *   The channels do not all have the same dimensions (shape).

    If optimizations are not enabled, the function also verifies that all
    channel values are in the range [0, 255]; since this is an expensive check,
    it is disabled if optimizations are turned on.
    '''

    # Make sure the correct number of channels were specified
    if len(channels) != 3:
        raise ValueError(f'Must specify 3 channels; got {len(channels)}')

    # Make sure that all color channels have unsigned integer data
    for c in channels:
        if c.dtype not in [np.uint8, np.uint16, np.uint32]:
            raise ValueError(f'All channels must be of type uint8, uint16, or uint32; got {c.dtype}')

    # Make sure that all color channels have the same dimensions
    for i in range(1, len(channels)):
        if channels[i].shape != channels[0].shape:
            raise ValueError(f'All channels must have the same dimensions')

    # Expensive sanity checks:
    print(f"np.nanmin(c): {np.nanmin(c)}")
    print(f"np.nanmax(c): {np.nanmax(c)}")
    if __debug__:
        assert (0 <= np.nanmin(c) <= 255) and (0 <= np.nanmax(c) <= 255), \
            'Channel may only contain values in range 0..255, and no NaNs'
    
    assert channels[0].dtype == np.uint8 and \
            channels[1].dtype == np.uint8 and \
            channels[2].dtype == np.uint8

    if isinstance(channels[0], np.ma.MaskedArray):
        # Create a masked array of zeros with the same shape as the channels
        rgb_data = np.ma.zeros(channels[0].shape, dtype=np.uint32)
        rgb_data.fill_value = 0xff000000  # Set the fill value for the masked array
    else:
        # Create a regular array of zeros
        rgb_data = np.zeros(channels[0].shape, dtype=np.uint32)

    rgb_data |= channels[0]
    rgb_data = rgb_data << 8
    rgb_data |= channels[1]
    rgb_data = rgb_data << 8
    rgb_data |= channels[2]
    rgb_data |= 0xff000000

    # Qt5/PySide2 complains if the array is not contiguous.
    if not rgb_data.flags['C_CONTIGUOUS']:
        rgb_data = np.ascontiguousarray(rgb_data)

    return rgb_data


def make_grayscale_image(channel: np.ndarray, colormap: Optional[str] = None) -> np.ndarray:
    '''
    Given a single image channel, this function generates a grayscale image, or,
    if a colormap name is specified, a color-mapped image from the single
    channel's data.

    An exception is raised if:
    *   The channel is not of type ``np.uint8``, ``np.uint16``, or ``np.uint32``.

    If optimizations are not enabled, the function also verifies that all
    channel values are in the range [0, 255]; since this is an expensive check,
    it is disabled if optimizations are turned on.
    '''
    def make_colormap_array(cmap):
        result = []
        for v in range(256):
            rgba = cmap(v, bytes=True)
            elem = np.uint32(0)
            elem |= rgba[0]
            elem = elem << 8
            elem |= rgba[1]
            elem = elem << 8
            elem |= rgba[2]
            elem |= 0xff000000

            # result.append(rgba[0] << 16 | rgba[1] << 8 | rgba[2] | 0xff000000)
            result.append(elem)

        return np.array(result, np.uint32)

    # Make sure that the color channel has unsigned integer data
    if channel.dtype not in [np.uint8, np.uint16, np.uint32]:
        raise ValueError(f'All channels must be of type uint8, uint16, or uint32; got {channel.dtype}')

    # Expensive sanity checks:
    if __debug__:
        assert (0 <= np.amin(channel) <= 255) and (0 <= np.amax(channel) <= 255), \
            'Channel may only contain values in range 0..255, and no NaNs'

    if isinstance(channel, np.ma.MaskedArray):
        # Create a masked array of zeros with the same shape as the channels
        rgb_data = np.ma.zeros(channel.shape, dtype=np.uint32)
        rgb_data.fill_value = 0xff000000  # Set the fill value for the masked array
    else:
        # Create a regular array of zeros
        rgb_data = np.zeros(channel.shape, dtype=np.uint32)
    
    if colormap is None:
        # Use the channel data to generate various gray RGB values.
        # rgb_data = (channel << 16 | channel << 8 | channel) | 0xff000000
        
        rgb_data |= channel
        rgb_data = rgb_data << 8
        rgb_data |= channel
        rgb_data = rgb_data << 8
        rgb_data |= channel
        rgb_data |= 0xff000000

    else:
        # Map the channel data to RGB colors using the colormap.
        cmap = cm.get_cmap(colormap, 256)
        cmap_arr = make_colormap_array(cmap)
        rgb_data = cmap_arr[channel]

    if isinstance(rgb_data, np.ma.MaskedArray):
        rgb_data.fill_value = 0xff000000

    # Qt5/PySide2 complains if the array is not contiguous.
    if not rgb_data.flags['C_CONTIGUOUS']:
        rgb_data = np.ascontiguousarray(rgb_data)

    return rgb_data


class ImageColors(enum.IntFlag):
    '''
    This enumeration is used to specify one or more color band, that may need to
    be regenerated within the rasterview.
    '''

    NONE = 0

    RED = 1
    GREEN = 2
    BLUE = 4

    RGB = 7


class ScaleToFitMode(enum.Enum):
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


class ImageWidget(QWidget):
    '''
    A subclass of QLabel used for displaying an image.  Since Qt provides events
    via virtual functions to be overloaded, this class forwards a number of
    important events to the enclosing widget.
    '''

    def __init__(self, rasterview: 'RasterView', forward: Dict):
        '''
        Initialize the image widget with the specified text and parent.  Store
        the object we are to forward relevant events to.
        '''
        super().__init__(parent=rasterview)
        self._rasterview = rasterview
        self._forward: Dict = forward

        self._scaled_size: Optional[QSize] = None

        self.setMouseTracking(True)

    def set_dataset_info(self, dataset, scale):
        # TODO(donnie):  Do something
        if dataset is not None:
            width = dataset.get_width()
            height = dataset.get_height()
            self._scaled_size = QSize(int(width * scale), int(height * scale))

        else:
            self._scaled_size = None

        # Inform the parent widget/layout that the geometry may have changed.
        self.setFixedSize(self._get_size_of_contents())

        # Request a repaint, since this function is called when any details
        # about the dataset are modified (including stretch adjustments, etc.)
        self.update()

    def set_dataset_info2(self, band, scale):
        # TODO(donnie):  Do something
        print(f"||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        if band is not None:
            width = band.shape[1]
            height = band.shape[0]
            print(f"width: {width}")
            print(f"height: {height}")
            self._scaled_size = QSize(int(width * scale), int(height * scale))
            print(f"Scaled size: {self._scaled_size}")

        else:
            print(f"Is none")
            self._scaled_size = None

        # Inform the parent widget/layout that the geometry may have changed.
        self.setFixedSize(self._get_size_of_contents())

        # Request a repaint, since this function is called when any details
        # about the dataset are modified (including stretch adjustments, etc.)
        self.update()


    def _get_size_of_contents(self):
        '''
        This helper function returns the size of the widget's scaled dataset,
        or a fixed size if the widget has no dataset.
        '''
        if self._scaled_size is not None:
            return self._scaled_size

        else:
            # TODO(donnie):  Do something more intelligent about this.
            return QSize(100, 100)


    def mousePressEvent(self, mouse_event):
        if 'mousePressEvent' in self._forward:
            self._forward['mousePressEvent'](self._rasterview, mouse_event)

    def mouseReleaseEvent(self, mouse_event):
        if 'mouseReleaseEvent' in self._forward:
            self._forward['mouseReleaseEvent'](self._rasterview, mouse_event)

    def mouseMoveEvent(self, mouse_event):
        if 'mouseMoveEvent' in self._forward:
            self._forward['mouseMoveEvent'](self._rasterview, mouse_event)

    def keyPressEvent(self, key_event):
        if 'keyPressEvent' in self._forward:
            self._forward['keyPressEvent'](self._rasterview, key_event)

    def keyReleaseEvent(self, key_event):
        if 'keyReleaseEvent' in self._forward:
            self._forward['keyReleaseEvent'](self._rasterview, key_event)

    def contextMenuEvent(self, context_menu_event):
        if 'contextMenuEvent' in self._forward:
            self._forward['contextMenuEvent'](self._rasterview, context_menu_event)

    def paintEvent(self, paint_event):
        with get_painter(self) as painter:
            if self._scaled_size is not None:
                # We have an image, so draw it, then forward on the repaint to
                # other code that cares.

                # Get the unscaled version of the pixmap.
                pixmap = self._rasterview.get_unscaled_pixmap()

                # Draw the scaled version of the pixmap.
                print(f"Before draw pixmap, width: {self._scaled_size.width()}, height: {self._scaled_size.height()}")
                painter.drawPixmap(0, 0,
                    self._scaled_size.width(), self._scaled_size.height(), pixmap,
                    0, 0, 0, 0)
                print(f"After drawPixel Map")
                if 'paintEvent' in self._forward:
                    print(f"PaintEvent forward")
                    self._forward['paintEvent'](self._rasterview, self, paint_event)
                print(f"After if statement")

            else:
                # Draw a note that there is no data to display.
                painter.setPen(Qt.black)
                painter.drawText(self.rect(), Qt.AlignCenter, '(no data)')


class ImageScrollArea(QScrollArea):
    '''
    A simple subclass of QScrollArea used for displaying an image that is
    potentially larger than the available display-area.  The main reason we
    subclass QScrollArea is simply to forward viewport-scroll events from the
    scroll-area to the RasterView, which can then act accordingly.
    '''

    def __init__(self, rasterview, forward, parent=None):
        super().__init__(parent)
        self._rasterview = rasterview
        self._forward = forward

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)

        if 'scrollContentsBy' in self._forward:
            self._forward['scrollContentsBy'](self._rasterview, dx, dy)

class RasterView(QWidget):
    '''
    A general-purpose widget for viewing raster datasets at varying zoom levels,
    possibly with overlay information drawn onto the image, such as annotations,
    regions of interest, etc.
    '''

    def __init__(self, parent=None, forward=None, app_state: ApplicationState = None):
        super().__init__(parent=parent)

        if forward == None:
            forward = {}

        self._app_state = app_state

        # Initialize fields in the object
        self._clear_members()
        self._scale_factor = 1.0

        # The widget used to display the image data

        self._image_widget = ImageWidget(self, forward)
        self._image_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._image_widget.setFocusPolicy(Qt.ClickFocus)

        # The scroll area used to handle images larger than the widget size

        self._scroll_area = ImageScrollArea(self, forward)
        self._scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scroll_area.setBackgroundRole(QPalette.Dark)
        self._scroll_area.setWidget(self._image_widget)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        # Set up the layout

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(QMargins(0, 0, 0, 0))

        # layout.addWidget(self.image_toolbar)
        # layout.setMenuBar(self.image_toolbar)
        self._layout.addWidget(self._scroll_area)
        self.setLayout(self._layout)

        self._display_bands_raw_data: List[Tuple[int, np.ndarray]] = [None, None, None]
        print("NEWWWWWWWWWWWWWWWWW \n RAAAAAAAAAAAAAAASSSSSSSSSSTTTTTTTTEEEEERRRRR \n VIEW")

    def get_stretches(self):
        return self._stretches

    @Slot(StretchBase)
    def set_stretches(self, stretches: List):
        import time
        print(f" raster view set stretches")
        self._stretches = stretches
        # Timing with time.time
        start_time = time.time()
        self.update_display_image()
        end_time = time.time()
        print(f"Time taken by update_display_image(): {end_time - start_time:.6f} seconds")

    def _clear_members(self):
        '''
        A helper function to clear all raster dataset members when the dataset
        changes.  This way we don't accidentally leave anything out.
        '''
        print(f"Clearing members")
        self._raster_data = None
        self._display_bands = None
        self._colormap: Optional[str] = None
        self._stretches = None

        # These members are for storing the components of the raster data, so
        # that assembling the image is faster when only one color's band
        # changes.

        self._display_data = [None, None, None]
        self._img_data = None

        # The image generated from the raw raster data.
        self._image = None
        self._image_pixmap = None


    def set_raster_data(self, raster_data, display_bands, stretches=None):
        '''
        Specify a raster data-set to display in the raster-view widget.  A value
        of None causes the raster-view to display nothing.
        '''
        print(f"Rasterview, set_raster_data: \n \
              raster_data {raster_data}  \
              \n display_bands: {display_bands} \
              \n stretches: {stretches}")
        if raster_data is not None and not isinstance(raster_data, RasterDataSet):
            raise ValueError('raster_data must be a RasterDataSet object')

        if raster_data is not None and len(display_bands) not in [1, 3]:
            raise ValueError(f'Unsupported number of display_bands:  {display_bands}')

        if stretches is not None and len(display_bands) != len(stretches):
            raise ValueError('display_bands and stretches must be the same length')

        self._clear_members()

        self._raster_data = raster_data

        if raster_data is not None:
            self._display_bands = display_bands
            print(f"self._display_bands_raw_data: {self._display_bands_raw_data}")
            # self._display_bands_raw_data = get_new_display_band_data(self._raster_data,
            #                                 self._display_bands_raw_data,
            #                                 self._display_bands)

            if stretches is not None:
                self._stretches = stretches
            else:
                # Default to no stretches.
                self._stretches = [None] * len(self._display_bands)
        else:
            print(f"Setting _display_bands_raw_datato none")
            self._display_bands = None
            self._display_bands_raw_data = [None, None, None]
            self._stretches = None

        self.update_display_image()

    def get_raster_data(self) -> Optional[RasterDataSet]:
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
        if self._display_bands is not None:
            return tuple(self._display_bands)
        else:
            return None


    def set_display_bands(self, display_bands: Tuple, stretches: List = None,
                          colormap: Optional[str] = None):
        print(f"RasterView, set_display_bands")
        if len(display_bands) not in [1, 3]:
            raise ValueError('display_bands must be a list of 1 or 3 ints')

        if stretches is not None and len(display_bands) != len(stretches):
            raise ValueError('display_bands and stretches must be same length')

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
        # self._display_bands_raw_data = get_new_display_band_data(self._raster_data,
        #                                                          self._display_bands_raw_data,
        #                                                          self._display_bands)
        self._colormap = colormap
        self.update_display_image(colors=changed)


    def get_colormap(self) -> Optional[str]:
        return self._colormap


    def update_display_image(self, colors=ImageColors.RGB):
        print(f"RasterView, update_display_image ")
        if self._raster_data is None:
            # No raster data to display
            self._image_widget.set_dataset_info(None, self._scale_factor)
            return

        # Only generate (or regenerate) each color plane if we don't already
        # have data for it, and if we aren't told to explicitly regenerate it.

        assert len(self._display_bands) in [1, 3]

        time_1 = time.perf_counter()

        ds_id = self._raster_data.get_id()

        last_added_raster_display = self._app_state.get_last_added_raster_display()
        current_added_raster_display = RasterViewMetaData(ds_id, self._display_bands, self._stretches, \
                                                          colormap=self._colormap)
        # TODO (Joshua G-K): Make this logic cleaner or move to another function
        if last_added_raster_display is not None and \
            last_added_raster_display.is_fully_initialized() and \
            last_added_raster_display == current_added_raster_display:
            img_data = self._app_state.get_last_added_raster_display().get_image_data()
            band_width = list(self._app_state._last_added_raster_display.get_raw_bands().values())[0].shape[1]
            band_height = list(self._app_state._last_added_raster_display.get_raw_bands().values())[0].shape[0]
            # band_width = None
            # band_height = None
            print(f"Skipping!")
            time_2 = time.perf_counter()
        else:
            display_bands_raw_data = get_new_display_band_data(self._raster_data,
                                                                 last_added_raster_display.get_raw_bands(),
                                                                 self._display_bands)
            band_width = None
            band_height = None
            if len(self._display_bands) == 3:
                # Check each color band to see if we need to update it.
                color_indexes = [ImageColors.RED, ImageColors.GREEN, ImageColors.BLUE]
                start1 = time.perf_counter()
                # for i in range(len(self._display_bands)):
                for i in [2, 1, 0]:
                    if self._display_data[i] is None or color_indexes[i] in colors:
                        # Start the timer
                        start_time = time.perf_counter()
                        band = self._display_bands[i]
                        # Compute the contents of this color channel.
                        self._display_data[i] = make_channel_image_with_band(display_bands_raw_data[band],
                                                    self._stretches[i])
                        
                        # self._display_data[i] = make_channel_image(self._raster_data, band, self._stretches[i])
                        band_width = display_bands_raw_data[band].shape[1]
                        band_height = display_bands_raw_data[band].shape[0]
                        print(f"Display data type: {type(self._display_data[i])}")
                        print(f"Display data: {self._display_data[i]}")
                        # End the timer
                        end_time = time.perf_counter()

                        # Print the time taken
                        print(f"Time taken for make_channel_image: {end_time - start_time:.6f} seconds")
                end1 = time.perf_counter()
                # Print the time taken
                print(f"Time taken for FOR loop: {end1 - start1:.6f} seconds")

                time_2 = time.perf_counter()
                # Start the timer
                start_time = time.perf_counter()
                # Combine our individual color channel(s) into a single RGB image.
                print(f"self._display_data: {self._display_data}")
                print(f"self._display_data[0]: {np.nanmin(self._display_data[0])}")
                print(f"self._display_data[1]: {np.nanmin(self._display_data[1])}")
                print(f"self._display_data[2]: {np.nanmin(self._display_data[2])}")
                img_data = make_rgb_image(self._display_data)
                # End the timer
                end_time = time.perf_counter()

                # Print the time taken
                print(f"Time taken for make_rgb_image: {end_time - start_time:.6f} seconds")

            else:
                # This is a grayscale image.
                if colors != ImageColors.NONE:
                    # Regenerate the image.  Since all color bands are the same,
                    # generate the first one, then duplicate it for the other two
                    # bands.

                    self._display_data[0] = make_channel_image(self._raster_data,
                        self._display_bands[0], self._stretches[0])

                    self._display_data[1] = self._display_data[0]
                    self._display_data[2] = self._display_data[0]

                time_2 = time.perf_counter()

                # Combine our individual color channel(s) into a single RGB image.
                img_data = make_grayscale_image(self._display_data[0], self._colormap)
            current_added_raster_display.set_image_data(img_data)
            current_added_raster_display.set_raw_band_data(display_bands_raw_data)
            self._app_state.set_last_added_raster_display(current_added_raster_display)
        self._display_data = [None, None, None]
        # This is necessary because the QImage doesn't take ownership of the
        # data we pass it, and if we drop this reference to the data then Python
        # will reclaim the memory and Qt will start to display garbage.
        self._img_data = img_data
        self._img_data.flags.writeable = False

        time_3 = time.perf_counter()

        start_time = time.perf_counter()
        # This is the 100% scale QImage of the data.
        if band_width is not None:
            print(f"band_width: {band_width}")
            print(f"band_height: {band_height}")
            self._image = QImage(img_data,
                band_width, band_height,
                QImage.Format_RGB32)
        else:
            print(f"self._raster_data.get_width(): {self._raster_data.get_width()}")
            print(f"self._raster_data.get_height(): {self._raster_data.get_height()}")
            self._image = QImage(img_data,
                self._raster_data.get_width(), self._raster_data.get_height(),
                QImage.Format_RGB32)
        # End the timer
        end_time = time.perf_counter()

        # Print the time taken
        print(f"Time taken for QImage: {end_time - start_time:.6f} seconds")

        start_time = time.perf_counter()
        self._image_pixmap = QPixmap.fromImage(self._image)
        # End the timer
        end_time = time.perf_counter()

        # Print the time taken
        print(f"Time taken for QPixmap: {end_time - start_time:.6f} seconds")

        time_4 = time.perf_counter()

        logger.debug(f'update_display_image(colors={colors}) update times:  ' +
                     f'channels = {time_2 - time_1:0.02f}s ' +
                     f'image = {time_3 - time_2:0.02f}s ' +
                     f'qt = {time_4 - time_3:0.02f}s')

        self._update_scaled_image()

    def get_unscaled_pixmap(self) -> QPixmap:
        '''
        Returns the unscaled QPixmap being displayed by this raster-view, along
        with contrast-stretch applied.
        '''
        return self._image_pixmap


    def get_image_data(self) -> Optional[np.ndarray]:
        '''
        Returns the raw image data being displayed by this raster-view, along
        with contrast-stretch applied.

        The result is a read-only NumPy array, where each pixel is a 32-bit
        unsigned integer stored in ARGB format, with 8 bits per color channel.
        '''
        return self._img_data


    def _update_scaled_image(self, old_scale_factor=None):
        self._image_widget.set_dataset_info2(list(self._app_state._last_added_raster_display.get_raw_bands().values())[0], self._scale_factor)
        # self._scroll_area.setVisible(True)

        # Need to process queued events now, since the image-widget has changed
        # size, and it needs to report a resize-event before the scrollbars will
        # update to the new size.
        # QCoreApplication.processEvents()

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


    def get_scrollbar_state(self) -> Tuple[int, int]:
        '''
        Returns the current state of the horizontal and vertical scrollbars.
        The state is returned as a 2-tuple of (horizontal scrollbar value,
        vertical scrollbar value).
        '''
        return (self._scroll_area.horizontalScrollBar().value(),
                self._scroll_area.verticalScrollBar().value())


    def set_scrollbar_state(self, state: Tuple[int, int]):
        '''
        Sets the state of the horizontal and vertical scrollbars to the
        specified values.  The state value must be a 2-tuple of (horizontal
        scrollbar value, vertical scrollbar value), as returned by
        get_scrollbar_state().
        '''
        self._scroll_area.horizontalScrollBar().setValue(state[0])
        self._scroll_area.verticalScrollBar().setValue(state[1])


    def get_visible_region(self) -> Optional[QRect]:
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
        print(f"Rasterview rgb_band_changed")
        self.update_display_image(colors=color)


    def image_coord_to_raster_coord(self, position: Union[QPoint, QPointF]) -> QPointF:
        '''
        Takes a position in screen space as a QPointF object, and translates it
        into a 2-tuple containing the (X, Y) coordinates of the position within
        the raster data set.
        '''
        if isinstance(position, QPoint):
            position = QPointF(position)
        elif not isinstance(position, QPointF):
            raise TypeError('This function requires a QPoint or QPointF ' +
                            f'argument; got {type(position)}')

        # Scale the screen position into the dataset's coordinate system.
        scaled = position / self._scale_factor

        # Convert to an integer coordinate.  Can't use QPointF.toPoint() because
        # it rounds to the nearest point, and we just want truncation/floor.
        return QPoint(int(scaled.x()), int(scaled.y()))
