from typing import Dict, List, Optional, Union, TYPE_CHECKING, Any, Tuple

from osgeo import gdal, gdal_array, osr

from sklearn.decomposition import PCA
import numpy as np
from astropy import units as u

from wiser.utils.numba_wrapper import numba_njit_wrapper, convert_to_float32_if_needed

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.spectrum import Spectrum

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QPlainTextEdit,
    QMessageBox,
)

ARRAY_NUMBA_THRESHOLD = 150000000  # 150 MB

# For easier typing in this module
Number = Union[int, float]


def numpy_dtype_to_gdal_export_types(elem_type: Union[np.dtype, type]) -> Tuple[int, np.dtype]:
    """
    Map an in-memory per-pixel dtype to a GDAL raster band type and a NumPy dtype
    for ``WriteArray`` / ``WriteRaster``.

    GDAL has no native boolean band type. Boolean cubes (for example SAM or SFF
    classification masks) are exported as ``GDT_Byte`` with values 0/1, so callers
    should cast band arrays to the returned NumPy dtype when it differs from the
    source element type.
    """
    et = np.dtype(elem_type)
    if np.issubdtype(et, np.bool_) or et == np.dtype(bool):
        return gdal.GDT_Byte, np.dtype(np.uint8)

    gdal_elem_type = gdal_array.NumericTypeCodeToGDALTypeCode(et)
    if gdal_elem_type is None:
        raise TypeError(f"Unsupported NumPy dtype for GDAL export: {et}")
    return gdal_elem_type, et


# ============================================================================
# OPERATIONS INVOLVING TYPED SPECTRAL VALUES


# Red:  700-635nm
RED_WAVELENGTH = 700 * u.nm

# Green:  560-520nm
GREEN_WAVELENGTH = 530 * u.nm

# Blue:  490-450nm
BLUE_WAVELENGTH = 470 * u.nm


# These are the string unit names used for band values by ENVI files, and their
# corresponding astropy.units representations.  All names are lowercase, so that
# it's easy to find the unit by converting the input text to lower case.
KNOWN_SPECTRAL_UNITS: Dict[str, u.Unit] = {
    "centimeters": u.cm,
    "meters": u.m,
    "micrometers": u.micrometer,
    "millimeters": u.millimeter,
    "microns": u.micron,
    "nanometers": u.nanometer,
    "cm": u.centimeter,
    "m": u.meter,
    "mm": u.millimeter,
    "nm": u.nanometer,
    "um": u.micrometer,
    "wavenumber": u.cm**-1,
    "angstroms": u.angstrom,
    "ghz": u.GHz,
    "mhz": u.MHz,
}


def get_spectral_unit_from_any(unit: Any) -> Optional[u.Unit]:
    if isinstance(unit, u.Unit):
        return unit
    elif isinstance(unit, str):
        return KNOWN_SPECTRAL_UNITS[unit.lower()]
    else:
        return None


def create_pca_metadata_widget(pca, dataset, parent=None) -> QWidget:
    """
    Create a QWidget that displays PCA metadata and has a 'Save To File' button.

    Args:
        pca: A fitted sklearn.decomposition.PCA instance.
        dataset: An object with a .get_name() method.
        parent: Optional parent QWidget.

    Returns:
        QWidget: The constructed widget.
    """

    class PcaMetadataWidget(QWidget):
        def __init__(self, pca_obj: PCA, dataset_obj: "RasterDataSet", parent=None):
            super().__init__(parent)
            self._pca = pca_obj
            self._dataset = dataset_obj

            self._text_edit = QPlainTextEdit(self)
            self._text_edit.setReadOnly(True)

            self._save_button = QPushButton("Save To File", self)
            self._save_button.clicked.connect(self._on_save_clicked)

            layout = QVBoxLayout(self)
            layout.addWidget(self._text_edit)
            layout.addWidget(self._save_button)
            self.setLayout(layout)

            self._text_edit.setPlainText(self._build_text())

        def _fmt_array(self, arr, indent="    "):
            # Nicely format numpy arrays with indentation
            arr = np.asarray(arr)
            # Use numpy's default truncation threshold (~1000 elements) so big
            # learned attributes like components_ — which is (N, N) for an
            # all-components fit — don't blow Qt's text layout with O(N²)
            # formatted floats.  Power users can still pickle ``pca`` directly.
            arr_str = np.array2string(
                arr,
                precision=4,
                suppress_small=True,
                max_line_width=120,
            )
            return "\n".join(indent + line for line in arr_str.splitlines())

        def _build_text(self) -> str:
            lines = []

            # Dataset name
            name = "Unknown"
            if hasattr(self._dataset, "get_name") and callable(self._dataset.get_name):
                name = self._dataset.get_name()

            lines.append(f"Dataset: {name}")
            lines.append("PCA Metadata")
            lines.append("=" * 60)
            lines.append("")

            # PCA init parameters (not learned attributes)
            lines.append("Parameters:")
            lines.append(f"  n_components: {getattr(self._pca, 'n_components', 'N/A')}")
            lines.append(f"  whiten: {getattr(self._pca, 'whiten', 'N/A')}")
            lines.append(f"  svd_solver: {getattr(self._pca, 'svd_solver', 'N/A')}")
            lines.append(f"  tol: {getattr(self._pca, 'tol', 'N/A')}")
            lines.append(f"  iterated_power: {getattr(self._pca, 'iterated_power', 'N/A')}")
            lines.append("")
            lines.append("Learned Attributes:")
            lines.append("-" * 60)

            # Helper to add an attribute if present
            def add_attr(name, label=None, is_array=False):
                if not hasattr(self._pca, name):
                    return
                value = getattr(self._pca, name)
                label = label or name
                if is_array:
                    lines.append(f"{label}:")
                    lines.append(self._fmt_array(value))
                else:
                    lines.append(f"{label}: {value}")
                lines.append("")

            # Common learned attributes after fit
            add_attr("n_components_", "n_components_")
            add_attr("n_features_in_", "n_features_in_")
            add_attr("mean_", "mean_", is_array=True)
            add_attr("components_", "components_", is_array=True)
            add_attr("explained_variance_", "explained_variance_", is_array=True)
            add_attr("explained_variance_ratio_", "explained_variance_ratio_", is_array=True)
            add_attr("singular_values_", "singular_values_", is_array=True)
            add_attr("noise_variance_", "noise_variance_")

            return "\n".join(lines)

        def _on_save_clicked(self):
            text = self._text_edit.toPlainText()
            if not text:
                QMessageBox.information(self, "Nothing to Save", "There is no text to save.")
                return

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save PCA Metadata",
                "",
                "Text Files (*.txt);;All Files (*)",
            )

            if not filename:
                return  # user cancelled

            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Saving File",
                    f"Could not save file:\n{e}",
                )

    return PcaMetadataWidget(pca, dataset, parent)


def finite_unmasked_row_mask(
    flattened_data: np.ndarray,
    flattened_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean ``[N]`` selecting usable rows of a ``[N][bands]`` spectra list.

    A row is usable iff every band is finite (no NaN/Inf) and — when a mask is
    supplied — every band is unmasked. A single bad feature invalidates the whole
    spectrum, so callers (PCA fit, MNF mean/covariance) drop the entire row.

    This is the single source of truth for "is this spectrum analysis-ready",
    shared by :func:`compute_PCA_on_image` and the pipeline's
    ``_flatten_valid_dataset_rows`` so PCA and MNF clean identically.
    """
    data = np.asarray(flattened_data)
    valid = np.all(np.isfinite(data), axis=1)
    if flattened_mask is not None:
        valid &= ~np.any(np.asarray(flattened_mask, dtype=bool), axis=1)
    return valid


def compute_PCA_on_image(
    image_arr: Union[np.ndarray, np.ma.masked_array],
    num_components: int,
    bad_bands: List[int] = None,
    data_ignore: Number = None,
) -> Tuple[Union[np.ndarray, np.ma.masked_array], PCA]:
    """
    This function handles all of the necessary cleaning needed to perform PCA
    on the spectra in an image_cube. This cleaning involves not including pixels with
    the data ignore value and not including bands that should be ignored. If there are
    any non-numeric values (like np.nan or +/-np.inf left in the array after cleaning,
    then this function errors).

    Args:
        image_arr (Union[np.ndarray, np.ma.masked_array]):
            A 3D array with dimensions [b][y][x]
        num_components (int):
            The number of components for PCA
        bad_bands (List[int]):
            An array where 1's mean keep the band, 0's mean get rid of it
        data_ignore (Number):
            The number the signifies a pixel should be ignored

    Returns:
        Union[np.ndarray, np.ma.masked_array]:
            The array after we have performmed PCA. It is returned in the format
            [y][x][b]
    """
    nbands = image_arr.shape[0]
    nrows = image_arr.shape[1]
    ncols = image_arr.shape[2]

    # Match each spectra with its location in the image
    ys = np.arange(nrows)
    xs = np.arange(ncols)

    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    # Shape [y][x][2]
    coords = np.stack((yy, xx), axis=2)

    # Remove the bad pixels from the image array
    image_arr = image_arr.transpose(1, 2, 0).copy(order="C")  # [b][y][x] --> [y][x][b]
    # [y][x][b] --> [y*x][b]
    image_arr: np.ndarray = image_arr.reshape((image_arr.shape[0] * image_arr.shape[1], image_arr.shape[2]))

    if bad_bands is not None:
        assert len(bad_bands) == nbands, "Length of bad_bands must match number of bands"
        bad_bands_bool = np.array(bad_bands, dtype=bool)
        image_arr = image_arr[:, bad_bands_bool]
    # [y][x][2] --> [y*x][2]
    coords = coords.reshape((coords.shape[0] * coords.shape[1], coords.shape[2]))

    # Drop any spectrum that is masked (nodata / bad-band sentinel) or holds a
    # non-finite value (NaN/Inf) in any kept band before fitting. sklearn's PCA
    # rejects NaN/Inf outright, and a single bad feature makes the whole spectrum
    # unusable.
    row_data = np.asarray(np.ma.getdata(image_arr))
    row_mask = np.ma.getmaskarray(image_arr) if isinstance(image_arr, np.ma.MaskedArray) else None
    valid_rows = finite_unmasked_row_mask(row_data, row_mask)
    image_arr = row_data[valid_rows, :]
    coords = coords[valid_rows, :]
    if image_arr.shape[0] == 0:
        raise ValueError("No valid spectra remain for PCA after removing masked/non-finite pixels")

    pca = PCA(n_components=num_components)

    # We expect oper_result to be given back to us in
    # [y*x][num_components] form
    oper_result = pca.fit_transform(image_arr)

    # Remove the bad bands from the spectra
    if data_ignore is None:
        data_ignore = np.nan

    return_arr = np.full((nrows, ncols, num_components), data_ignore, dtype=np.float32)
    return_arr[coords[:, 0], coords[:, 1], :] = oper_result
    masked_return_arr = np.ma.masked_values(return_arr, data_ignore)

    return masked_return_arr, pca


def build_band_info_from_wavelengths(
    wavelengths: List[u.Quantity],
) -> List[Dict[str, Any]]:
    band_info = []
    for i, wl in enumerate(wavelengths):
        band_info.append(
            {
                "index": i,
                # Match your existing format used by update_band_info (e.g., "472.02 nm")
                "description": f"{wl.value:.2f} {wl.unit.to_string()}",
                "wavelength": wl,
                "wavelength_str": str(wl.value),  # numeric value as string
                "wavelength_units": wl.unit.to_string(),  # unit name as string
            }
        )
    return band_info


# ============================================================================
# NETCDF WAVELENGTH EXTRACTION


# Variable-name fragments that suggest a variable holds wavelength data.
# All comparisons are done case-insensitively as a substring search.
_WAVELENGTH_VAR_SUBSTRINGS: frozenset = frozenset(
    {"wavelength", "wavelengths", "wvl", "wlen", "lambda", "bands"}
)


def parse_unit_from_string(s: Optional[str]) -> Optional[u.Unit]:
    """Parse an astropy unit from a plain string.

    Tries the astropy parser first, then falls back to a hand-written mapping
    that covers common spectral unit spellings.  Returns ``None`` for
    unitless / unrecognised strings.
    """
    if not s:
        return None
    t = s.strip().lower().replace("µ", "u")
    if t in {"unitless", "dimensionless", "1"}:
        return None
    try:
        return u.Unit(t)
    except Exception:
        pass
    mapping: Dict[str, u.Unit] = {
        "nm": u.nanometer,
        "nanometer": u.nanometer,
        "nanometers": u.nanometer,
        "um": u.micrometer,
        "micrometer": u.micrometer,
        "micrometers": u.micrometer,
        "mm": u.millimeter,
        "millimeter": u.millimeter,
        "millimeters": u.millimeter,
        "cm": u.centimeter,
        "centimeter": u.centimeter,
        "centimeters": u.centimeter,
        "m": u.meter,
        "meter": u.meter,
        "meters": u.meter,
        "angstrom": u.angstrom,
        "å": u.angstrom,
        "cm-1": u.cm**-1,
        "cm^-1": u.cm**-1,
        "1/cm": u.cm**-1,
        "wavenumber": u.cm**-1,
        "ghz": u.GHz,
        "mhz": u.MHz,
    }
    if t in mapping:
        return mapping[t]
    for key, unit in mapping.items():
        if key in t:
            return unit
    return None


def _is_wavelength_var_name(name: str) -> bool:
    """Return ``True`` if *name* contains any wavelength-related keyword."""
    lower = name.lower()
    return any(kw in lower for kw in _WAVELENGTH_VAR_SUBSTRINGS)


def _collect_wavelength_candidates(
    group,
) -> List[Tuple[np.ndarray, Optional[u.Unit]]]:
    """Recursively walk *group* and all sub-groups, returning candidates.

    A candidate is any variable whose name passes :func:`_is_wavelength_var_name`
    and whose data is a 1-D numeric array.  Each candidate is a
    ``(data_array, unit_or_None)`` tuple.

    Traversal order: variables in the current group first (in iteration
    order), then sub-groups depth-first.  This means shallower / earlier
    variables take priority when selection is applied later.
    """
    candidates: List[Tuple[np.ndarray, Optional[u.Unit]]] = []

    for var_name, var in group.variables.items():
        if not _is_wavelength_var_name(var_name):
            continue
        try:
            data = np.asarray(var[:])
            if data.ndim != 1 or not np.issubdtype(data.dtype, np.number):
                continue
        except Exception:
            continue
        unit = parse_unit_from_string(getattr(var, "units", None))
        candidates.append((data, unit))

    for sub_group in group.groups.values():
        candidates.extend(_collect_wavelength_candidates(sub_group))

    return candidates


def extract_netcdf_wavelengths(
    netcdf_dataset,
) -> Tuple[Optional[np.ndarray], Optional[u.Unit]]:
    """Search a ``netCDF4.Dataset`` for wavelength data at any nesting depth.

    All groups and sub-groups are visited recursively.  Variable names are
    matched case-insensitively against the substrings ``wavelength``,
    ``wavelengths``, ``wvl``, ``wlen``, ``lambda``, and ``bands``. Nasa
    netcdf products use CF conventions. While these conventions where
    a wavelength variable should be specified and contain units.

    Selection
    ---------
    1. Return the **first** candidate (in depth-first, declaration order) that
       has **both** a 1-D numeric data array *and* a recognised unit.
    2. If no candidate has both, return the first available data array paired
       with the first available unit — either may be ``None``.

    Returns
    -------
    ``(wavelengths, unit)`` where *wavelengths* is a :class:`numpy.ndarray`
    or ``None``, and *unit* is an :class:`astropy.units.Unit` or ``None``.
    """
    candidates = _collect_wavelength_candidates(netcdf_dataset)

    if not candidates:
        return None, None

    # Priority 1: first candidate with both data and unit
    for data, unit in candidates:
        if data is not None and unit is not None:
            return data, unit

    # Priority 2: independently pick the first data and the first unit
    first_data = next((d for d, _ in candidates if d is not None), None)
    first_unit = next((unit for _, unit in candidates if unit is not None), None)
    return first_data, first_unit


_GOOD_WAVELENGTH_VAR_SUBSTRINGS: frozenset = frozenset(
    {"good_wavelengths", "good_wavelength", "bbl", "bad_band_list"}
)


def _is_good_wavelength_var_name(name: str) -> bool:
    """Return ``True`` if *name* matches a good-wavelength variable."""
    lower = name.lower()
    return any(kw in lower for kw in _GOOD_WAVELENGTH_VAR_SUBSTRINGS)


def _collect_good_wavelength_candidates(group) -> List[np.ndarray]:
    """Recursively collect good-wavelength masks, walking *group* and all
    sub-groups depth-first.

    Each candidate is a 1-D ``int`` array where ``1`` means the band is good
    and ``0`` means bad.  Any element whose raw value equals the variable's
    ``_FillValue`` is forced to ``0`` before the array is appended.
    """
    candidates: List[np.ndarray] = []

    for var_name, var in group.variables.items():
        if not _is_good_wavelength_var_name(var_name):
            continue
        try:
            raw = np.asarray(var[:])
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number):
                continue
        except Exception:
            continue

        data = raw.astype(int)

        fill_value = getattr(var, "_FillValue", None)
        if fill_value is not None:
            try:
                data[raw == fill_value] = 0
            except Exception:
                pass

        candidates.append(data)

    for sub_group in group.groups.values():
        candidates.extend(_collect_good_wavelength_candidates(sub_group))

    return candidates


def extract_netcdf_bad_bands(netcdf_dataset) -> Optional[List[int]]:
    """Search a ``netCDF4.Dataset`` for a good-wavelength mask at any depth.

    All groups and sub-groups are visited recursively.  Variable names are
    matched case-insensitively against the substrings ``good_wavelength`` and
    ``good_wavelengths``.

    Values are interpreted as ``1`` = good band, ``0`` = bad band.  Any
    element whose raw value equals the variable's ``_FillValue`` is treated as
    a bad band (set to ``0``).

    The first matching array is returned as a :class:`numpy.ndarray` of
    ``numpy.bool_``.  Returns ``None`` if no matching variable is found.
    """
    candidates = _collect_good_wavelength_candidates(netcdf_dataset)
    if not candidates:
        return None
    return [int(v) for v in candidates[0]]


def get_netCDF_reflectance_path(file_path):
    """
    Checks for the presence of reflectance and reflectance uncertainty subdatasets.
    Returns the path to reflectance if available, otherwise falls back to reflectance uncertainty.
    """
    # Open the netCDF file with GDAL
    dataset = gdal.Open(file_path)

    # Get the list of subdatasets
    subdatasets = dataset.GetSubDatasets()

    # Check for reflectance and reflectance uncertainty
    for subdataset, _ in subdatasets:
        if "reflectance" in subdataset:
            return subdataset
        elif "reflectance_uncertainty" in subdataset:
            return subdataset
        elif "mask" in subdataset:
            return subdataset

    raise Exception("netCDF file type is not supported!")


def get_spectral_unit(unit_str: str) -> u.Unit:
    """
    Given a string representation of the units, this function returns an
    ``astropy.units.Unit`` object to represent the unit.
    """
    return KNOWN_SPECTRAL_UNITS[unit_str.lower()]


def spectral_unit_to_string(unit: u.Unit) -> str:
    for k, v in KNOWN_SPECTRAL_UNITS.items():
        if unit == v:
            return k

    return None


def make_spectral_value(value: Number, unit_str: str) -> u.Quantity:
    """
    Given a numeric value and a string representation of the units, this
    function returns an ``astropy.units.Quantity`` object to represent the
    value with units.
    """
    return value * get_spectral_unit(unit_str)


def convert_spectral(value: u.Quantity, to_unit: u.Unit) -> u.Quantity:
    """
    Convert a spectral value with units (e.g. a frequency or wavelength),
    to the specified units.
    """
    return value.to(to_unit, equivalencies=u.spectral())


def get_band_values(input_bands: List[u.Quantity], to_unit: Optional[u.Unit] = None) -> List[float]:
    """
    Given a list of band values represented as astropy.units.Quantity (values
    with units), this function will convert all quantities to a single unit, and
    then return a list of just the numeric values.

    The caller may specify what unit to convert all values to, using the to_unit
    argument.  If this is left as None, the unit of the first quantity in the
    list is used.
    """
    if to_unit is None:
        to_unit = input_bands[0].unit

    return [convert_spectral(v, to_unit).value for v in input_bands]


def convert_spectrum_wavelengths(spectrum: "Spectrum", to_unit: u.Unit) -> np.ndarray:
    """Return a spectrum's per-band wavelengths as a float array in ``to_unit``.

    Conversion goes through :func:`convert_spectral`, which applies spectral
    equivalencies, so wavelength, frequency, and wavenumber grids all convert
    correctly.  For a bare list of band wavelengths (e.g. a dataset's grid),
    call :func:`get_band_values` directly.

    Args:
        spectrum: The :class:`~wiser.raster.spectrum.Spectrum` to read
            wavelengths from.
        to_unit: The astropy unit to express the wavelengths in.

    Returns:
        A 1-D ``float64`` array of the per-band wavelength values in ``to_unit``.

    Raises:
        ValueError: If ``spectrum`` has no wavelengths.
        astropy.units.UnitConversionError: If the wavelengths cannot be
            converted to ``to_unit``.
    """
    if not spectrum.has_wavelengths():
        raise ValueError("Spectrum has no wavelengths to convert.")
    return np.asarray(get_band_values(spectrum.get_wavelengths(), to_unit), dtype=np.float64)


def set_band(arr: np.ndarray, band_index: int, value) -> None:
    """
    Sets the specified band (axis 2 index) of a 3D array or the entire array if it is 2D to a given value.
    """
    if arr.ndim == 2:
        arr[band_index, :] = value
    elif arr.ndim == 3:
        arr[band_index, :, :] = value
    else:
        raise TypeError(
            f"The passed in array should only have either 2 or 3 dimensions, but it has: {arr.ndim}"
        )


# ============================================================================
# FINDING SUITABLE BANDS IN RASTER DATA SETS


def find_band_near_wavelength(
    bands: List[Dict], wavelength: u.Quantity, max_distance: u.Quantity = 20 * u.nm
) -> Optional[int]:
    """
    Given a collection of bands and a wavelength, this function will try to find
    the band closest to the wavelength that is also within the maximum distance
    specified to the function.

    The index of the band in the list of bands is returned from the function.
    If no suitable band is found, the function returns None.
    """

    wavelengths = [b.get("wavelength") for b in bands]
    if None in wavelengths:
        raise ValueError("Not all bands specify a wavelength")

    return find_closest_wavelength(wavelengths, wavelength, max_distance)


def find_closest_wavelength(
    wavelengths: List[u.Quantity],
    input_wavelength: u.Quantity,
    max_distance: u.Quantity = None,
) -> Optional[int]:
    """
    Given a list of wavelengths and an input wavelength, this function returns
    the index of the wavelength closest to the input wavelength.  If no
    wavelength is within max_distance of the input then None is returned.
    """

    # Do the whole calculation in nm to keep things simple.
    if max_distance is None:
        max_distance = 20 * input_wavelength.unit.si
    input_value = convert_spectral(input_wavelength, u.nm).value
    max_dist_value = None
    if max_distance is not None:
        max_dist_value = convert_spectral(max_distance, u.nm).value

    values = [convert_spectral(v, u.nm).value for v in wavelengths]

    return find_closest_value(values, input_value, max_dist_value)


def find_closest_value(
    values: List[Number], input_value: Number, max_distance: Optional[Number] = None
) -> Optional[int]:
    """
    Given a list of numbers (ints and/or floats) and an input number, this
    function returns the index of the number closest to the input number.
    If no number is within max_distance of the input then None is returned.
    """
    best_index = None
    best_distance = None

    for index, value in enumerate(values):
        distance = abs(value - input_value)

        if max_distance is not None and distance > max_distance:
            continue

        if best_index is None or distance < best_distance:
            best_index = index
            best_distance = distance

    return best_index


# ============================================================================
# COMMON BAND-MATH OPERATIONS


def normalize_ndarray_python(array: np.ndarray, minval=None, maxval=None) -> Union[None, np.ndarray]:
    """
    Normalize the specified array, generating a new array to return to the
    caller.  The minimum and maximum values can be specified if already known,
    or if the caller wants to normalize to a different min/max than the array's
    actual min/max values.  NaN values are left unaffected.
    """
    dt = array.dtype
    if not (np.issubdtype(dt, np.integer) or np.issubdtype(dt, np.floating)):
        array = array.astype(np.float32)
    if isinstance(minval, (bool, np.bool_)):
        minval = float(minval)

    if isinstance(maxval, (bool, np.bool_)):
        maxval = float(maxval)
    if minval is None:
        minval = np.nanmin(array)

    if maxval is None:
        maxval = np.nanmax(array)

    if maxval == minval:
        return np.zeros_like(array, dtype=np.float32)

    return (array - minval) / (maxval - minval)


@numba_njit_wrapper(non_njit_func=normalize_ndarray_python)
def normalize_ndarray_numba(data: np.ndarray, minval: float, maxval: float) -> np.ndarray:
    """
    Normalize an array to the range [0, 1].
    """
    if maxval == minval:
        return np.zeros_like(data, dtype=np.float32)
    # Create an empty array with the same shape as `data` and dtype float32
    normalized = np.empty(data.shape, dtype=np.float32)

    # Total number of elements in the array
    total_elements = data.size

    # Iterate over each element in the flattened array
    for idx in range(total_elements):
        value = data.flat[idx]
        if np.isfinite(value):
            normalized.flat[idx] = (value - minval) / (maxval - minval)
        else:
            normalized.flat[idx] = 0.0  # Handle NaN or Inf

    return normalized


def normalize_ndarray(arr: np.ndarray, minval=None, maxval=None) -> Union[None, np.ndarray]:
    if arr.nbytes < ARRAY_NUMBA_THRESHOLD:
        return normalize_ndarray_python(array=arr, minval=minval, maxval=maxval)
    else:
        arr, minval, maxval = convert_to_float32_if_needed(arr, minval, maxval)
        return normalize_ndarray_numba(arr, minval, maxval)


def get_normalized_band(dataset, band_index):
    """
    Extracts the specified band of raster data, mapping all elements to the
    range of [0.0, 1.0].  Elements will be of type np.float32, unless the input
    data is already np.float64, in which case the elements are left as
    np.float64.
    """
    band_data = dataset.get_band_data(band_index)
    stats = dataset.get_band_stats(band_index)

    norm_data = (band_data - stats.get_min()) / (stats.get_max() - stats.get_min())

    if norm_data.dtype not in [np.float32, np.float64]:
        norm_data = norm_data.astype(np.float32)

    return norm_data


def get_normalized_band_using_stats(band_data: np.ndarray, stats):
    """
    Maps all elements in the band to the range of [0.0, 1.0].
    Elements will be of type np.float32, unless the input
    data is already np.float64, in which case the elements are left as
    np.float64.
    """
    if isinstance(band_data, np.ma.masked_array):
        band_data_mask = band_data.mask
        band_data = band_data.data
    norm_data = normalize_ndarray(band_data, stats.get_min(), stats.get_max())
    if isinstance(band_data, np.ma.masked_array):
        band_data = np.ma.masked_array(band_data, mask=band_data_mask)

    if norm_data.dtype not in [np.float32, np.float64]:
        norm_data = norm_data.astype(np.float32)

    return norm_data


def set_data_ignore_of_gdal_dataset(gdal_dataset: gdal.Dataset, source_dataset: "RasterDataSet"):
    nodata = source_dataset.get_data_ignore_value()
    if nodata is not None:
        # set the same nodata on every band
        for i in range(1, gdal_dataset.RasterCount + 1):
            gdal_dataset.GetRasterBand(i).SetNoDataValue(nodata)


def copy_metadata_to_gdal_dataset(gdal_dataset: gdal.Dataset, source_dataset: "RasterDataSet"):
    # Propagate wavelength names (band descriptions)
    band_info = source_dataset.band_list()  # returns dict of lists keyed by metadata names
    wle_names = band_info[0].get("wavelength_name")
    if wle_names:
        for i, band_info in enumerate(band_info):
            wle_name = band_info.get("wavelength_name")
            b = gdal_dataset.GetRasterBand(i + 1)
            b.SetDescription(wle_name)

    # Propagate data‑ignore (NoData) value
    nodata = source_dataset.get_data_ignore_value()
    if nodata is not None:
        # set the same nodata on every band
        for i in range(1, gdal_dataset.RasterCount + 1):
            gdal_dataset.GetRasterBand(i).SetNoDataValue(nodata)

    # Propagate default bands (for display)
    defaults = source_dataset.default_display_bands()
    if defaults:
        # store as comma‑separated string in metadata
        gdal_dataset.SetMetadataItem("DEFAULT_BANDS", ",".join(str(b) for b in defaults))

    # Propagate bad bands
    bad = source_dataset.get_bad_bands()  # list of ints
    if bad:
        gdal_dataset.SetMetadataItem("BAD_BANDS", ",".join(str(b) for b in bad))

    # Propagate wavelength units:
    wl_str = band_info[0].get("wavelength_str")  # list of astropy.Quantity
    if wl_str is not None:
        for i, q in enumerate(band_info):
            wl_str = band_info[i].get("wavelength_str")
            gdal_dataset.GetRasterBand(i + 1).SetMetadataItem("wavelength", wl_str)

    wl_units = band_info[0].get("wavelength_units")  # Should be an astropy.Unit
    if wl_units is not None:
        for i, q in enumerate(band_info):
            wl_units = band_info[i].get("wavelength_units")
            gdal_dataset.GetRasterBand(i + 1).SetMetadataItem("wavelength_units", str(wl_units))

    # Note: the caller owns the handle's lifetime. We intentionally do not flush
    # or close here so callers can keep stamping/reading before closing.


def get_bbox(gt, width, height):
    """Compute (minX, minY, maxX, maxY) of a raster given its GeoTransform."""
    xs, ys = [], []
    for px, py in ((0, 0), (width, 0), (0, height), (width, height)):
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def reproject_bbox(bbox, src_srs, dst_srs):
    """Reproject the 4 corners of bbox into dst_srs."""
    ct = osr.CoordinateTransformation(src_srs, dst_srs)
    corners = [
        (bbox[0], bbox[1]),
        (bbox[0], bbox[3]),
        (bbox[2], bbox[1]),
        (bbox[2], bbox[3]),
    ]
    pts = [ct.TransformPoint(x, y)[:2] for x, y in corners]
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)


def bboxes_intersect(b1, b2):
    """Return True if b1 and b2 (minX,minY,maxX,maxY) overlap."""
    return not (
        b1[2] < b2[0]  # b1.maxX < b2.minX
        or b1[0] > b2[2]  # b1.minX > b2.maxX
        or b1[3] < b2[1]  # b1.maxY < b2.minY
        or b1[1] > b2[3]  # b1.minY > b2.maxY
    )


def can_transform_between_srs(srs1: osr.SpatialReference, srs2: osr.SpatialReference):
    try:
        ct = osr.CoordinateTransformation(srs1, srs2)  # noqa: F841
        return True
    except BaseException:
        return False


def have_spatial_overlap(
    srs1: osr.SpatialReference,
    gt1: List[float],
    w1: int,
    h1: int,
    srs2: osr.SpatialReference,
    gt2: List[float],
    w2: int,
    h2: int,
):
    """
    Return True if two rasters (given by their OSR SpatialReference,
    GeoTransform, width & height) overlap in space.
    """
    # 1) compute each envelope
    bbox1 = get_bbox(gt1, w1, h1)
    bbox2 = get_bbox(gt2, w2, h2)

    # 2) reproject bbox2 into srs1 (if needed)
    if not srs1.IsSame(srs2):
        bbox2 = reproject_bbox(bbox2, srs2, srs1)

    # 3) test intersection
    return bboxes_intersect(bbox1, bbox2)
