import abc
import enum
import os
from typing import List, Optional, Tuple, Union, Dict

from collections import deque

from PySide2.QtCore import *

import numpy as np
from astropy import units as u


from wiser.gui.util import get_random_matplotlib_color, get_color_icon

from wiser.raster.dataset import RasterDataSet, SpectralMetadata
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import SelectionType
from wiser.raster.serializable import Serializable, SerializedForm


# ============================================================================
# SPECTRAL CALCULATIONS


class SpectrumAverageMode(enum.Enum):
    """
    This enumeration specifies the calculation mode when a spectrum is computed
    over multiple pixels of a raster data set.
    """

    # Compute the mean (average) spectrum over multiple spatial pixels
    MEAN = 1

    # Compute the median spectrum over multiple spatial pixels
    MEDIAN = 2


AVG_MODE_NAMES = {
    SpectrumAverageMode.MEAN: "Mean",
    SpectrumAverageMode.MEDIAN: "Median",
}


def find_rectangles_in_row(row: np.ndarray, y: int) -> List[np.ndarray]:
    rectangles = []
    start = None

    for x in range(len(row)):
        if row[x] == 1 and start is None:
            start = x  # Start of a new rectangle
        elif row[x] == 0 and start is not None:
            rectangles.append(np.array([start, x - 1, y, y]))  # End of rectangle
            start = None

    # If the row ends and a rectangle was still open
    if start is not None:
        rectangles.append(np.array([start, len(row) - 1, y, y]))

    return rectangles


def raster_to_combined_rectangles_x_axis(raster):
    rectangles = []
    previous_row_rectangles = deque()
    # prev_row_rect_to_keep = []

    for y in range(raster.shape[0]):  # For each row (y-coordinate)
        row = raster[y]
        current_row_rectangles = find_rectangles_in_row(row, y)

        # Get all of the previous rectangles (from previous row or continued on from farther back rows)
        # Since we haven't yet added the previous row's rectangles to our final set of rectangles, if there
        # are no matches with a current rectangle then it is safe to add it in with the final set of rects
        for i in range(len(previous_row_rectangles)):
            prev_rect = previous_row_rectangles.pop()
            prev_x_start, prev_x_end, prev_y_start, prev_y_end = prev_rect

            merged = False
            # Get all of the rectangles in the current row, we will compare each previous rectangle
            # with all the rectangles in the current row. If there is a match in x and y values
            # between the previous rectangle and a current row rectangle, then we update the current
            # row rectangle's size to expand. Then when we add this rectangle to previous row rectangles
            # it will have carried over.
            # If there isn't a merge, we add the previous rect to rectangles.
            for curr_rect in current_row_rectangles:
                x_start, x_end, y_start, y_end = curr_rect

                # If the current rectangle does match with a previous rectangle
                if prev_x_start == x_start and prev_x_end == x_end and prev_y_end == y - 1:
                    # Merge the current rectangle with the previous one
                    curr_rect[-2] = prev_y_start
                    merged = True
                    break

            # If the previous rectangle here does not continue from a current rows, we
            # immediately add it to rectangles which we can do because we know it
            # won't show up again
            if not merged:
                rectangles.append(np.array(prev_rect))

        # We make the previous row rectangles the be the current row to "move on" from this row
        # The current rectangles are updated to get merged into the previous ones and nothing is doubly added
        previous_row_rectangles = current_row_rectangles

    # For the last row it is never treated as a previous row (which would let it be added to
    # rectangles), so we just add it to rectangles
    rectangles += list(previous_row_rectangles)  # Accumulate merged rectangles

    return np.array(rectangles)


def raster_to_combined_rectangles_y_axis(raster_y: np.ndarray):
    raster_x = raster_y.T
    rectangles_x = raster_to_combined_rectangles_x_axis(raster_x)
    rectangles_y = rectangles_x[:, [0, 1, 2, 3]] = rectangles_x[:, [2, 3, 0, 1]]
    return rectangles_y


def array_to_qrects(array):
    qrects = []
    for row in array:
        x1, x2, y1, y2 = row
        # QRect takes (x, y, width, height), so calculate width and height
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        qrects.append(QRect(x1, y1, width, height))
    return qrects


def create_raster_from_roi(roi: RegionOfInterest) -> np.ndarray:
    bbox = roi.get_bounding_box()
    pixels = roi.get_all_pixels()

    xmin = bbox.topLeft().x()
    ymin = bbox.topLeft().y()

    raster = np.zeros((bbox.height(), bbox.width()), dtype=np.uint8)

    for pixel in pixels:
        pixel_x, pixel_y = pixel[0], pixel[1]
        pixel_index_x, pixel_index_y = pixel_x - xmin, pixel_y - ymin
        raster[pixel_index_y][pixel_index_x] = 1
    return raster


def calc_spectrum_fast(dataset: RasterDataSet, roi: RegionOfInterest, mode=SpectrumAverageMode.MEAN):
    """
    Calculate a spectrum over a region of interest from the specified dataset.
    The calculation mode can be specified with the mode argument.
    """
    spectra = []

    # We make a raster out of all of the pixels in the ROI
    raster = create_raster_from_roi(roi)

    # We do a variant of the Run Line Encoding (RLE) algorithm in the x direction
    # and the y direction
    rect_x_axis = raster_to_combined_rectangles_x_axis(raster)
    rect_y_axis = raster_to_combined_rectangles_y_axis(raster)
    bbox = roi.get_bounding_box()

    rects = None
    if len(rect_x_axis) < len(rect_y_axis):
        rects = rect_x_axis
    else:
        rects = rect_y_axis

    # We need to make the rectangles we got from the 'RLE' algorithm
    # be in the image coordinate system
    rects[:, :2] += bbox.left()
    rects[:, 2:] += bbox.top()

    # Accessing by rectangular blocks is faster than accessing point by point
    qrects = array_to_qrects(rects)
    for qrect in qrects:
        try:
            s = dataset.get_all_bands_at_rect(qrect.left(), qrect.top(), qrect.width(), qrect.height())
        except BaseException:
            # TODO (Joshua G-K): Make this cleaner. Either check in impl or don't let user create
            # ROIs that go out of bounds.
            arr = np.full((dataset.num_bands(),), np.nan)
            return arr
        ndim = s.ndim
        if ndim == 2:
            for i in range(s.shape[1]):
                spectra.append(s[:, i])
        elif ndim == 3:
            for i in range(s.shape[1]):
                for j in range(s.shape[2]):
                    spectra.append(s[:, i, j])
        else:
            raise TypeError(f"Expected 2 or 3 dimensions in rectangular aray, but got {s.ndim}")

    if len(spectra) > 1:
        spectra = np.asarray(spectra)
        # Need to compute mean/median/... of the collection of spectra
        if mode == SpectrumAverageMode.MEAN:
            spectrum = np.nanmean(spectra, axis=0)
        elif mode == SpectrumAverageMode.MEDIAN:
            spectrum = np.nanmedian(spectra, axis=0)
        else:
            raise ValueError(f"Unrecognized average type {mode}")

    else:
        # Only one spectrum, don't need to compute mean/median
        spectrum = spectra[0]

    return spectrum


def calc_rect_spectrum(dataset: RasterDataSet, rect: QRect, mode=SpectrumAverageMode.MEAN):
    """
    Calculate a spectrum over a rectangular area of the specified dataset.
    The calculation mode can be specified with the mode argument.

    The rect argument is expected to be a QRect object.
    """
    points = [(rect.left() + dx, rect.top() + dy) for dx, dy in np.ndindex(rect.width(), rect.height())]

    return calc_spectrum(dataset, points, mode)


def calc_spectrum(dataset: RasterDataSet, points: List[QPoint], mode=SpectrumAverageMode.MEAN):
    """
    Calculate a spectrum over a collection of points from the specified dataset.
    The calculation mode can be specified with the mode argument.

    The points argument can be any iterable that produces coordinates for this
    function to use.
    """

    n = 0
    spectra = []

    # Collect the spectra that we need for the calculation
    for p in points:
        n += 1
        s = dataset.get_all_bands_at(p[0], p[1])
        spectra.append(s)

    if len(spectra) > 1:
        # Need to compute mean/median/... of the collection of spectra
        if mode == SpectrumAverageMode.MEAN:
            spectrum = np.mean(spectra, axis=0)

        elif mode == SpectrumAverageMode.MEDIAN:
            spectrum = np.median(spectra, axis=0)

        else:
            raise ValueError(f"Unrecognized average type {mode}")

    else:
        # Only one spectrum, don't need to compute mean/median
        spectrum = spectra[0]

    return spectrum


def get_all_spectra_in_roi(
    dataset: RasterDataSet, roi: RegionOfInterest
) -> List[Tuple[Tuple[int, int], np.ndarray]]:
    """
    Given a raster data set and a region of interest, this function returns an
    array of 2-tuples, where each pair is comprised of:

    *   The pixel's (x, y) integer coordinates as a 2-tuple
    *   A NumPy ndarray object containing the spectrum at that coordinate.

    Note that the spectral data will include NaNs for any value from a bad band,
    or that was set to the "data ignore value".
    """
    # Generate the set of all pixels in the ROI.  Turn it into a list so we can
    # sort it.
    all_pixels = list(roi.get_all_pixels())
    all_pixels.sort()

    # Generate the collection of spectra at all of those pixels.  Each element
    # in the list is the pixel, plus its NumPy
    all_spectra = [(p, dataset.get_all_bands_at(x=p[0], y=p[1])) for p in all_pixels]

    return all_spectra


def calc_roi_spectrum(dataset: RasterDataSet, roi: RegionOfInterest, mode=SpectrumAverageMode.MEAN):
    """
    Calculate a spectrum over a Region of Interest from the specified dataset.
    The calculation mode can be specified with the mode argument.
    """
    return calc_spectrum_fast(dataset, roi, mode)


# ============================================================================
# CLASSES TO REPRESENT SPECTRA


class Spectrum(abc.ABC, Serializable):
    """
    The base class for representing spectra of interest to the user of the
    application.
    """

    def __init__(self):
        self._id: Optional[int] = None
        self._name: Optional[str] = None

        self._color = None

        # TODO(donnie):  Should these be in this class, or in display code?
        self._icon = None
        self._visible = True

    def get_id(self) -> Optional[int]:
        return self._id

    def set_id(self, id: int) -> None:
        self._id = id

    def get_name(self) -> str:
        raise NotImplementedError("Must be implemented in subclass")

    def set_name(self, name: str):
        raise NotImplementedError("Must be implemented in subclass")

    def get_source_name(self) -> str:
        """
        Returns the name of the spectrum's source.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def num_bands(self) -> int:
        """Returns the number of spectral bands in the spectrum."""
        pass

    def get_bad_bands(self) -> np.ndarray:
        """
        Returns a boolean numpy array indicating which bands are bad (1 for bands
        to keep, 0 for bands to removed).  If no bad bands are defined, returns None.
        """
        return np.array([1] * self.num_bands(), dtype=np.bool_)

    def get_shape(self) -> Tuple[int]:
        """
        Returns the shape of the spectrum.  This is always simply
        ``(num_bands)``.
        """
        return (self.num_bands(),)

    def get_elem_type(self) -> np.dtype:
        """
        Returns the element-type of the spectrum.
        """
        pass

    def has_wavelengths(self) -> bool:
        """
        Returns True if this spectrum has wavelength units for all bands, False
        otherwise.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def get_wavelengths(self) -> List[u.Quantity]:
        """
        Returns a list of wavelength values corresponding to each band.  The
        individual values are astropy values-with-units.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def get_wavelength_units(self) -> Optional[u.Unit]:
        """
        Returns the astropy unit corresponding to the wavelength.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def get_spectrum(self) -> np.ndarray:
        """
        Return the spectrum data as a 1D NumPy array.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def get_color(self) -> Optional[str]:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color
        self._icon = None

    def is_editable(self) -> bool:
        # By default, spectra are editable.
        return True

    def is_discardable(self) -> bool:
        # By default, spectra are discardable.
        return True

    def get_serialized_form(self) -> SerializedForm:
        """
        This should return all of the information needed to recreate this object.
        The first element is this class, so we can get the deserialize_into_class function
        The second element is a string that represents the file path to the dataset, or a numpy array
        that represents the data in the dataset. The third element is a dictionary that represents
        the metadata needed to recreate this object.
        """
        spectrum_arr = self.get_spectrum()
        metadata = {
            "name": self.get_name(),
            "source_name": self.get_source_name(),
            "id": self.get_id(),
            "elem_type": self.get_elem_type(),
            "wavelengths": self.get_wavelengths(),
            "wavelength_units": self.get_wavelength_units(),
            "editable": self.is_editable(),
            "discardable": self.is_discardable(),
        }
        return SerializedForm(self.__class__, spectrum_arr, metadata)

    def get_spectral_metadata(self) -> SpectralMetadata:
        spectral_metadata = SpectralMetadata(
            band_info=None,
            bad_bands=None,
            default_display_bands=None,
            num_bands=self.num_bands(),
            data_ignore_value=None,
            has_wavelengths=self.has_wavelengths(),
            wavelengths=self.get_wavelengths(),
            wavelength_units=self.get_wavelength_units(),
        )
        return spectral_metadata

    @staticmethod
    def deserialize_into_class(spectrum_arr: Union[str, np.ndarray], metadata: Dict) -> "NumPyArraySpectrum":
        """
        This should recreate the object from the serialized form that is obtained from the
        get_serialized_form method.
        """
        name = metadata["name"]
        source_name = metadata["source_name"]
        id = metadata["id"]
        wavelengths = metadata["wavelengths"]
        editable = metadata["editable"]
        discardable = metadata["discardable"]
        spectrum = NumPyArraySpectrum(spectrum_arr, name, source_name, wavelengths, editable, discardable)
        spectrum.set_id(id)
        return spectrum


# ===============================================================================
# NUMPY ARRAY SPECTRA
# ===============================================================================


class NumPyArraySpectrum(Spectrum):
    """
    This class represents a spectrum that wraps a simple 1D NumPy array.  This
    is generally used for computed spectra.
    """

    def __init__(
        self,
        arr: np.ndarray,
        name: Optional[str] = None,
        source_name: Optional[str] = None,
        wavelengths: Optional[List[u.Quantity]] = None,
        editable=True,
        discardable=True,
    ):
        super().__init__()

        self._arr = arr

        self._name = name
        self._source_name = "unknown"
        if source_name:
            self._source_name = source_name

        self._wavelengths = wavelengths

        self._editable = editable
        self._discardable = discardable

        self._bad_bands: np.ndarray = np.array([1] * arr.shape[0], dtype=np.bool_)

    def get_name(self) -> Optional[str]:
        """
        Returns the current name of the spectrum, or ``None`` if no name has
        been assigned.
        """
        return self._name

    def set_name(self, name: Optional[str]):
        """
        Sets the name of the spectrum.  ``None`` may be specified if the
        spectrum is to be unnamed.
        """
        self._name = name

    def get_source_name(self) -> Optional[str]:
        """
        Returns the name of the spectrum's source, or ``None`` if no source
        name has been specified.
        """
        return self._source_name

    def set_source_name(self, name: str):
        self._source_name = name

    def get_elem_type(self) -> np.dtype:
        """
        Returns the element-type of the spectrum.
        """
        return self._arr.dtype

    def num_bands(self) -> int:
        """Returns the number of spectral bands in the spectrum."""
        return self._arr.shape[0]

    def set_bad_bands(self, bad_bands: np.ndarray):
        """Sets the bad bands array for this spectrum. 1 is keep, 0 is ignore."""
        assert bad_bands.ndim == 1 and bad_bands.shape[0] == self.num_bands(), (
            "Passed in bad bands either doesn't have 1 dimension or doesn't have"
            "same number of bands as this spectrum!"
        )
        self._bad_bands = bad_bands

    def get_bad_bands(self):
        return self._bad_bands

    def has_wavelengths(self) -> bool:
        """
        Returns True if this spectrum has wavelength units for all bands, False
        otherwise.
        """
        if self._wavelengths is None:
            return False
        return isinstance(self._wavelengths[0], u.Quantity)

    def get_wavelengths(self) -> List[u.Quantity]:
        """
        Returns a list of wavelength values corresponding to each band.  The
        individual values are astropy values-with-units.
        """
        if self._wavelengths is None:
            raise KeyError("Spectrum doesn't have wavelengths")

        return self._wavelengths

    def set_wavelengths(self, wavelengths: Optional[List[u.Quantity]]):
        """
        Sets the wavelength values that correspond to each band.  The argument
        is a list of astropy values-with-units.  Alternately, this method may
        be used to clear the wavelength information, by passing in ``None`` as
        the argument.
        """
        if wavelengths is not None:
            if len(wavelengths) != self.num_bands():
                raise ValueError(
                    f"Spectrum has {self.num_bands()} bands, but "
                    + f"{len(wavelengths)} wavelengths were specified"
                )

            # Make a copy of the incoming list
            wavelengths = list(wavelengths)

        self._wavelengths = wavelengths

    def get_wavelength_units(self) -> Optional[u.Unit]:
        if self.has_wavelengths():
            if isinstance(self._wavelengths[0], u.Quantity):
                return self._wavelengths[0].unit
        return None

    def copy_spectral_metadata(self, source: SpectralMetadata):
        assert source.get_wavelengths(), "SpectralMetadata has no wavelengths"
        self.set_wavelengths(source.get_wavelengths())

    def get_spectrum(self) -> np.ndarray:
        """
        Return the spectrum data as a 1D NumPy array.
        """
        return self._arr

    def is_editable(self):
        return self._editable

    def is_discardable(self):
        return self._discardable

    def __eq__(self, other: "NumPyArraySpectrum") -> bool:
        return (
            self.get_spectrum() == other.get_spectrum()
            and self.get_elem_type() == other.get_elem_type()
            and self.get_wavelengths() == other.get_wavelengths()
            and self.get_wavelength_units() == other.get_wavelength_units()
        )


# ===============================================================================
# RASTER DATA-SET SPECTRA
# ===============================================================================


class RasterDataSetSpectrum(Spectrum):
    def __init__(self, dataset):
        super().__init__()
        self._dataset: RasterDataSet = dataset
        self._avg_mode = SpectrumAverageMode.MEAN

        self._name = None
        self._use_generated_name = True

        # This field holds the spectrum data.  It is generated lazily, so it
        # won't be set until it is requested.  Additionally, it may be set back
        # to None if the details of how the spectrum is generated are changed.
        self._spectrum = None

    def __str__(self) -> str:
        return f"RasterDataSetSpectrum[{self.get_source_name()}, " + f"name={self.get_name()}]"

    def _reset_internal_state(self):
        """
        This internal helper function should be called when important details
        of this object change, possibly necessitating the recalculation of the
        spectrum data and/or a generated name for the spectrum.
        """
        self._spectrum = None
        if self._use_generated_name:
            self._name = None

    def get_name(self) -> Optional[str]:
        if self._name is None and self._use_generated_name:
            self._name = self._generate_name()

        return self._name

    def set_name(self, name):
        self._name = name

    def set_use_generated_name(self, use_generated: bool) -> None:
        self._use_generated_name = use_generated
        if use_generated:
            self._name = None

    def use_generated_name(self) -> bool:
        return self._use_generated_name

    def _generate_name(self) -> str:
        """
        This helper function generates a name for this spectrum from its
        configuration details.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def get_source_name(self):
        filenames = self._dataset.get_filepaths()
        if filenames is not None and len(filenames) > 0:
            ds_name = os.path.basename(filenames[0])
        else:
            ds_name = "unknown"

        return ds_name

    def get_dataset(self):
        return self._dataset

    def get_avg_mode(self):
        return self._avg_mode

    def set_avg_mode(self, avg_mode):
        if avg_mode not in SpectrumAverageMode:
            raise ValueError("avg_mode must be a value from SpectrumAverageMode")

        self._avg_mode = avg_mode
        self._reset_internal_state()

    def num_bands(self) -> int:
        """Returns the number of spectral bands in the raster data."""
        return self._dataset.num_bands()

    def get_bad_bands(self):
        return self._dataset.get_bad_bands()

    def get_elem_type(self) -> np.dtype:
        """
        Returns the element-type of the spectrum.
        """
        return self._dataset.get_elem_type()

    def has_wavelengths(self) -> bool:
        """
        Returns True if this spectrum has wavelength units for all bands, False
        otherwise.
        """
        return self._dataset.has_wavelengths()

    def get_wavelengths(self, filter_bad_bands=False) -> List[u.Quantity]:
        """
        Returns a list of wavelength values corresponding to each band.  The
        individual values are astropy values-with-units.
        """
        b0: Dict = self._dataset.band_list()[0]
        if "wavelength" in b0:
            key = "wavelength"
        else:
            key = "index"
        bands = [b[key] for b in self._dataset.band_list()]
        if filter_bad_bands:
            bad_bands = self._dataset.get_bad_bands()
            bands = [bands[i] for i in range(len(bands)) if bad_bands[i]]
        return bands

    def get_wavelength_units(self) -> Optional[u.Unit]:
        """
        Gets the wavelength units.
        """
        return self.get_wavelengths()[0].unit

    def _calculate_spectrum(self):
        """
        This internal helper method computes and stores the spectrum data for
        this object, based on its current configuration.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def get_spectrum(self) -> np.ndarray:
        """
        Return the spectrum data as a 1D NumPy array.
        """
        if self._spectrum is None:
            self._calculate_spectrum()

        return self._spectrum

    def __eq__(self, other: "Spectrum") -> bool:
        return (
            self.get_spectrum() == other.get_spectrum()
            and self.get_elem_type() == other.get_elem_type()
            and self.get_wavelengths() == other.get_wavelengths()
            and self.get_wavelength_units() == other.get_wavelength_units()
        )


class SpectrumAtPoint(RasterDataSetSpectrum):
    """
    This class represents the spectrum at or around a point in a raster data
    set.  A rectangular area may be specified, and an average spectrum will be
    computed over that area.
    """

    def __init__(
        self,
        dataset: RasterDataSet,
        point: Tuple[int, int],
        area: Tuple[int, int] = (1, 1),
        avg_mode=SpectrumAverageMode.MEAN,
    ):
        super().__init__(dataset)

        self._point: Tuple[int, int] = point
        self._area: Tuple[int, int] = None
        self.set_area(area)
        self.set_avg_mode(avg_mode)

    def _generate_name(self) -> str:
        """
        This helper function generates a name for this spectrum from its
        configuration details.
        """

        if self._area == (1, 1):
            name = f"Spectrum at ({self._point[0]}, {self._point[1]})"

        else:
            name = (
                f"{AVG_MODE_NAMES[self._avg_mode]} of "
                + f"{self._area[0]}x{self._area[1]} "
                + f"area around ({self._point[0]}, {self._point[1]})"
            )

        return name

    def _calculate_spectrum(self):
        """
        This internal helper method computes and stores the spectrum data for
        this object, based on its current configuration.
        """
        (x, y) = self._point

        if self._area == (1, 1):
            try:
                self._spectrum = self._dataset.get_all_bands_at(x, y)
            except:
                self._spectrum = np.full((self._dataset.num_bands(),), np.nan)

        else:
            (width, height) = self._area
            left = x - (width - 1) / 2
            top = y - (height - 1) / 2
            rect = QRect(left, top, width, height)
            try:
                self._spectrum = calc_rect_spectrum(self._dataset, rect, mode=self._avg_mode)
            except:
                self._spectrum = np.full((width, height), np.nan)

    def get_point(self):
        return self._point

    def get_area(self):
        return self._area

    def set_area(self, area) -> None:
        if area[0] % 2 != 1 or area[1] % 2 != 1:
            raise ValueError(f"area values must be odd; got {area}")

        if self._area != area:
            self._area = area
            self._reset_internal_state()


class ROIAverageSpectrum(RasterDataSetSpectrum):
    """
    This class represents the average spectrum of a Region of Interest in a
    raster data set.
    """

    def __init__(
        self,
        dataset: RasterDataSet,
        roi: RegionOfInterest,
        avg_mode=SpectrumAverageMode.MEAN,
    ):
        super().__init__(dataset)

        self._roi: RegionOfInterest = roi
        self.set_avg_mode(avg_mode)

    def get_roi(self) -> RegionOfInterest:
        return self._roi

    def _generate_name(self) -> str:
        """
        This helper function generates a name for this spectrum from its
        configuration details.
        """
        return f"{AVG_MODE_NAMES[self._avg_mode]} of " + f'"{self._roi.get_name()}" Region of Interest'

    def _calculate_spectrum(self):
        """
        This internal helper method computes and stores the spectrum data for
        this object, based on its current configuration.
        """
        self._spectrum = calc_roi_spectrum(self._dataset, self._roi, self._avg_mode)
