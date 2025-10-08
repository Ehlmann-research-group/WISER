import os
import textwrap

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy import units as u


class EnviFileFormatError(Exception):
    """
    Represents an issue in an ENVI format data file, or the header file
    associated with the data file.
    """

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message


ENVI_UNIT_STRS: Dict[u.Unit, str] = {
    u.cm: "Centimeters",
    u.m: "Meters",
    u.micrometer: "Micrometers",
    u.millimeter: "Millimeters",
    u.nanometer: "Nanometers",
    u.cm**-1: "Wavenumber",
    u.angstrom: "Angstroms",
    u.GHz: "GHz",
    u.MHz: "MHz",
}


ENVI_UNIT_CAPITALIZATIONS: Dict[str, str] = {
    "centimeters": "Centimeters",
    "meters": "Meters",
    "micrometers": "Micrometers",
    "millimeters": "Millimeters",
    "nanometers": "Nanometers",
    "wavenumber": "Wavenumber",
    "angstroms": "Angstroms",
    "ghz": "GHz",
    "mhz": "MHz",
}


def astropy_unit_to_envi_str(astropy_unit: u.Unit) -> Optional[str]:
    """
    Map an astropy unit-value to an ENVI unit-string.
    """
    return ENVI_UNIT_STRS.get(astropy_unit)


def wiser_unitstr_to_envi_str(wiser_str: str) -> str:
    """
    Make sure to follow the string capitalization that ENVI specifies for its
    header files.
    """
    if wiser_str is None:
        return None

    s = wiser_str.lower()
    if s in ENVI_UNIT_CAPITALIZATIONS:
        return ENVI_UNIT_CAPITALIZATIONS[s]

    return wiser_str


def normalize_envi_header_property_name(name: str) -> str:
    """
    Normalize a property name from an ENVI header file by removing leading and
    trailing whitespace, and ensuring that words in the name are separated by a
    single space.
    """
    name = name.strip()
    parts = name.split(" ")
    return " ".join(parts)


def envi_multivalue_to_list(value: str, elemtype=float) -> List:
    """
    Convert a string of the form "{value, value, ...}" to a list of actual
    values.  The values are parsed using the function specified with the
    elemtype argument.  The list of values is returned.

    If the string doesn't start with "{" and end with "}" then a ValueError
    is raised.
    """

    if value[0] != "{" or value[-1] != "}":
        raise ValueError('Multivalue must be enclosed with "{" and "}"')

    parts = value[1:-1].split(",")
    return [elemtype(p.strip()) for p in parts]


def map_envi_type_to_numpy_type(envi_type: int, envi_endian: int) -> np.dtype:
    """
    Given an ENVI type number and endian value (0 = little-endian;
    1 = big-endian), this function returns the corresponding numpy.dtype
    value for reading the ENVI data.

    If the type number or the endian value is unrecognized then a ValueError
    with a suitable message will be raised.
    """

    type_map = {
        1: "B",  # 8-bit unsigned integer
        2: "i2",  # 16-bit signed integer
        3: "i4",  # 32-bit signed integer
        4: "f4",  # 32-bit single-precision floating-point
        5: "f8",  # 64-bit double-precision floating-point
        # TODO(donnie):  Not sure if this is c4 or c8
        6: "c8",  # Complex:  pair of single-precision floating-point
        9: "c16",  # Double-precision complex:  pair of double precision floating-point
        12: "u2",  # 16-bit unsigned integer
        13: "u4",  # 32-bit unsigned integer
        14: "i8",  # 64-bit signed integer
        15: "u8",  # 64-bit unsigned integer
    }

    if envi_type not in type_map:
        raise ValueError(f"Unrecognized ENVI type number {envi_type}")

    numpy_type_str = type_map[envi_type]

    if envi_type != 1:  # Bytes don't have an endianness
        if envi_endian == 0:
            numpy_type_str = "<" + numpy_type_str
        elif envi_endian == 1:
            numpy_type_str = ">" + numpy_type_str
        else:
            raise ValueError(f"Unrecognized ENVI endian value {envi_endian}")

    return np.dtype(numpy_type_str)


def load_envi_header(filename: str) -> Dict[str, Any]:
    """
    Loads the .hdr file for an ENVI format spectral file, and returns the
    metadata as a Python dictionary.  Keys in the dictionary are normalized ENVI
    header property names.

    The filename argument is expected to be the path and filename to the actual
    ENVI header file.  If the file cannot be opened then the error will be
    reported by an exception raised by the Python built-in open() function.

    File format errors are indicated by raising an EnviFileFormatError
    exception.
    """

    with open(filename) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    # First line should be "ENVI"
    if len(lines) == 0 or lines[0] != "ENVI":
        raise EnviFileFormatError('Line 1:  "ENVI" specifier is missing')

    # ===========================================================================
    # First, all metdata key/value pairs in the .hdr file are converted to
    # values in the metadata Python dictionary.

    metadata = {}

    # In the code, the first line of the file has line number 0.  We translate
    # this to 1-based line numbers in error messages.
    line_no = 0
    while True:
        line_no += 1
        if line_no >= len(lines):  # Have we consumed all lines?
            break

        line = lines[line_no]

        # If the line is empty, skip it.
        if len(line) == 0:
            continue

        # If the line is a comment, skip it.  (Note that the line's whitespace
        # was already stripped earlier.)
        if line[0] == ";":
            continue

        # Try to parse a "name = value" line, which may span multiple lines
        # if the value is wrapped in curly-braces {} .

        parts = line.split("=", maxsplit=1)
        if len(parts) != 2:
            raise EnviFileFormatError(f'Line {line_no+1}:  not of the form "name = value"')

        name = normalize_envi_header_property_name(parts[0])

        value = parts[1].strip()
        if value[-1] == "{":
            # The value is a multiline value.  Read in subsequent lines until
            # we see a line ending with "}".

            value_start_line = line_no
            while True:
                line_no += 1
                if line_no >= len(lines):  # Are we at EOF?
                    raise EnviFileFormatError(
                        f'Line {value_start_line+1}:  EOF before reading entire value of "{name}"'
                    )

                line = lines[line_no]
                value += line
                if line[-1] == "}":
                    break

        metadata[name] = value

    # ===========================================================================
    # Next, some metdata values are specific types (e.g. float/int), so we go
    # through and convert those entries' values to the expected type.

    # These attributes are integers, but might be written as a float.
    for k in [
        "samples",
        "lines",
        "bands",
        "header offset",
        "data type",
        "byte order",
        "y start",
    ]:
        if k in metadata:
            metadata[k] = int(float(metadata[k]))

    # These attributes are floats
    for k in ["data ignore value"]:
        if k in metadata:
            metadata[k] = float(metadata[k])

    # These attributes are lists of floats
    for k in ["wavelength", "correction factors", "smoothing factors", "fwhm"]:
        if k in metadata:
            metadata[k] = envi_multivalue_to_list(metadata[k])

    # This attribute is a list of integers
    if "bbl" in metadata:
        metadata["bbl"] = envi_multivalue_to_list(metadata["bbl"], int)

    # This attribute is a list of strings
    if "spectra names" in metadata:
        metadata["spectra names"] = envi_multivalue_to_list(metadata["spectra names"], str)

    # ===========================================================================
    # All done!

    return metadata


def load_envi_data(filename, header_filename=None, metadata=None, mmap=True):
    """
    Loads the specified ENVI data file from disk.  The filename is the path
    to the binary data file; the header file is assumed to be at the path
    filename + '.hdr'.

    The function returns a 2-tuple of this form:  (metadata, data).

    *   metadata is a dictionary of the header file's metadata, with some type
        coercion applied to the data.

    *   data is a 3D NumPy array shaped to be accessed with [x][y][band]
        coordinates.  (In ENVI parlance, this is [samples][lines][bands].)
        This is done with the data's underlying interleaving taken into account.

    Setting the mmap flag to True causes the data file to be memory-mapped;
    setting the flag to False causes it to be loaded entirely into memory.
    Memory-mapping the file will cause the function to return faster, because
    the data will be loaded incrementally as it is accessed, rather than
    needing to be loaded in its entirety up-front.
    """

    if header_filename is None:
        header_filename = filename + ".hdr"

    if metadata is None:
        # Load the metadata so that we know what we're in for.
        metadata = load_envi_header(header_filename)

    # Figure out how many elements to expect in the data

    bands = metadata["bands"]
    samples = metadata["samples"]
    lines = metadata["lines"]
    num_values = bands * samples * lines

    # Figure out the numpy element type from the ENVI header
    numpy_type = map_envi_type_to_numpy_type(metadata["data type"], metadata["byte order"])

    # Load the binary data itself.
    header_offset = metadata["header offset"]
    with open(filename, "rb") as f:
        if mmap:
            data = np.memmap(f, numpy_type, "r", offset=header_offset)
        else:
            data = np.fromfile(f, dtype=numpy_type, count=num_values, offset=header_offset)

    # Figure out the shape of the 3D array, based on the interleaving.

    interleave = metadata["interleave"]

    if interleave == "bsq":  # Band Sequential:  [bands, samples, lines]
        data.shape = (bands, samples, lines)
        data = np.moveaxis(data, 0, 2)

    elif interleave == "bip":  # Band-interleaved-by-pixel:  [samples, lines, bands]
        data.shape = (samples, lines, bands)
        # No moveaxis required.

    elif interleave == "bil":  # Band-interleaved-by-line:  [lines, bands, samples]
        data.shape = (lines, bands, samples)
        data = np.moveaxis(data, 1, 2)

    else:
        raise EnviFileFormatError(f'Unrecognized ENVI interleave "{interleave}"')

    return (metadata, data)


def find_envi_filenames(filename: str) -> Tuple[str, str]:
    """
    Given a filename that may be either the header file or the image file of an
    ENVI raster iamge, this function attempts to figure out both the header and
    image filenames based on common extensions and approaches.

    If the function is successful at identifying both the header and image
    filenames, it returns a tuple ``(hdr_filename, img_filename)``.

    A ``FileNotFoundError`` is raised if the specified ``filename`` doesn't
    exist, or if both the header and image filenames cannot be determined.
    """
    # Make sure that whatever we were handed, is actually a file.  Otherwise,
    # there's no point in going on.
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    header_filename = None
    data_filename = None

    if filename.endswith(".hdr"):
        # Looks like we were handed an ENVI header file.
        header_filename = filename

        # Try to determine the data filename from the header filename.

        nohdr = filename[:-4]
        candidates = [nohdr, nohdr + ".img", nohdr + ".sli"]
        for candidate in candidates:
            if os.path.isfile(candidate):
                data_filename = candidate
                break

        if data_filename is None:
            raise FileNotFoundError(
                "Couldn't determine ENVI data filename " + f"from header file {header_filename}"
            )

    else:
        # Assume we were handed an ENVI data file.
        data_filename = filename

        # Try to determine the header filename from the data filename.

        # print(f'filename = "{filename}"')
        candidates = [filename + ".hdr"]
        if filename.endswith(".img") or filename.endswith(".sli"):
            candidates.append(filename[:-4] + ".hdr")

        # print(f'considering candidates:  {candidates}')
        for candidate in candidates:
            if os.path.isfile(candidate):
                header_filename = candidate
                break

        if header_filename is None:
            raise FileNotFoundError(
                "Couldn't determine ENVI header filename " + f"from data file {data_filename}"
            )

    return (header_filename, data_filename)


def load_envi_file(filename, mmap=True):
    """
    Loads the specified ENVI file from disk.  The filename can be either the
    header file or the binary data file; this function will determine whether
    the argument is the header or the data file, and will look for other files
    in the directory to try to identify the argument's peer file.

    The function returns a 3-tuple of this form:  (file_list, metadata, data).

    *   file_list is the list of files that the function found for this ENVI
        file.  The header file is listed first, and the binary data file is
        listed second.

    *   metadata is a dictionary of the header file's metadata, with some type
        coercion applied to the data.

    *   data is a 3D NumPy array shaped to be accessed with [x][y][band]
        coordinates.  (In ENVI parlance, this is [samples][lines][bands].)
        This is done with the data's underlying interleaving taken into account.

    Setting the mmap flag to True causes the data file to be memory-mapped;
    setting the flag to False causes it to be loaded entirely into memory.
    Memory-mapping the file will cause the function to return faster, because
    the data will be loaded incrementally as it is accessed, rather than
    needing to be loaded in its entirety up-front.
    """

    # This function should throw if the header and data filenames cannot be
    # determined for some reason.
    (header_filename, data_filename) = find_envi_filenames(filename)
    assert header_filename is not None and data_filename is not None

    (metadata, data) = load_envi_data(data_filename, header_filename, mmap)

    return ([header_filename, data_filename], metadata, data)


def list_to_envi_multivalue(name: str, seq: List) -> str:
    """
    Convert a list (or other sequence) of data values to a nicely formatted ENVI
    "name = { value, value, ... }".  If the string version is too long to fit on
    a single line, it is text-wrapped across multiple lines.
    """
    # Generate a nice comma-separated and wrapped version of the list data.
    seq_str = ", ".join([str(v) for v in seq])
    if len(name) + len(seq_str) + 3 > 78:
        lines = textwrap.wrap(seq_str, 76, initial_indent="  ", subsequent_indent="  ")
        seq_str = "\n".join(lines)
        return f"{name} = {{\n{seq_str} }}\n"
    else:
        return f"{name} = {{ {seq_str} }}\n"


def write_envi_name_value_pair(f, name: str, value: Any) -> None:
    if value is None:
        return

    if isinstance(value, list) or isinstance(value, tuple):
        output_str = list_to_envi_multivalue(name, value)

    else:
        # Not an iterable.  Use the single-valued format.

        if isinstance(value, float) and value.is_integer():
            value = int(value)

        output_str = f"{name} = {value}\n"

    f.write(output_str)


def verify_required_config(metadata: Dict[str, Any], name: str):
    """
    Verify that the specified attribute-name has a non-None value in the ENVI
    metadata.  If no value is specified for the name (or if the value is None),
    then a ``ValueError`` is raised.
    """
    if metadata.get(name) is None:
        raise ValueError(f'ENVI metadata is missing required config "{name}"')


def write_envi_header(filename: str, metadata: Dict[str, Any]) -> None:
    known_keys = [
        "description",
        # Structural information about the data
        "samples",
        "lines",
        "bands",
        "header offset",
        "file type",
        "data type",
        "interleave",
        "sensor type",
        "byte order",
        "data ignore value",
        "y start",
        # Geographical information
        "map info",
        "projection info",
        "coordinate system string",
        # Band metadata
        "default bands",
        "band names",
        "spectra names",
        "wavelength units",
        "wavelength",
        "fwhm",
        "bbl",
        # Supporting files
        "crosstrack scatter file",
        "flat field file",
        "spectral scatter file",
        "correction factors",
        "wavelength file",
        "radiance version",
        "rcc file",
        "smoothing factors",
        "bad pixel map",
    ]

    # Verify that all required keys are present
    for name in [
        "samples",
        "lines",
        "bands",
        "file type",
        "data type",
        "interleave",
        "byte order",
    ]:
        verify_required_config(metadata, name)

    with open(filename, "w") as f:
        f.write("ENVI\n")

        all_keys = set(metadata.keys())

        # Go through the keys we know about, in the specified ordering.
        for name in known_keys:
            value = metadata.get(name)
            all_keys.discard(name)
            write_envi_name_value_pair(f, name, value)

        # Output all unrecognized keys at the end.
        for name in all_keys:
            value = metadata.get(name)
            write_envi_name_value_pair(f, name, value)
