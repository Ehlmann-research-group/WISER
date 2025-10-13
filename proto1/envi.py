import numpy


class EnviFileFormatError(Exception):
    """
    Represents an issue in an ENVI format data file, or the header file
    associated with the data file.
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


def normalize_name(name):
    """
    Normalize a name from an ENVI file by removing leading and trailing
    whitespace, and ensuring that all words in the name are separated by a
    single space.
    """
    name = name.strip()
    parts = name.split(" ")
    return " ".join(parts)


def multivalue_to_list(value, elemtype=float):
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


def map_envi_type_to_numpy_type(envi_type, envi_endian):
    """
    Given an ENVI type number and endian value (0 = little-endian;
    1 = big-endian), this function returns the corresponding numpy.dtype
    value for reading the ENVI data.

    If the type number is unrecognized then a KeyError will be raised.
    If the endian value is unrecognized then a ValueError will be raised.
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

    numpy_type_str = type_map[envi_type]

    if envi_type != 1:  # Bytes don't have an endianness
        if envi_endian == 0:
            numpy_type_str = "<" + numpy_type_str
        elif envi_endian == 1:
            numpy_type_str = ">" + numpy_type_str
        else:
            raise ValueError("Unrecognized ENVI endian value {envi_endian}")

    return numpy.dtype(numpy_type_str)


def load_envi_header(f):
    """
    Loads the .hdr file for an ENVI format multispectral file.

    File format errors are indicated by raising an EnviFileFormatError
    exception.
    """

    # First line should be "ENVI"
    line = f.readline().strip()
    if line != "ENVI":
        raise EnviFileFormatError('Line 1:  "ENVI" specifier is missing')

    metadata = {}

    line_no = 1
    while True:
        line_no += 1
        line = f.readline()
        if len(line) == 0:  # Are we at EOF?
            break

        # Remove leading and trailing whitespace.  If the line is empty,
        # skip it.
        line = line.strip()
        if len(line) == 0:
            continue

        # Try to parse a "name = value" line, which may span multiple lines
        # if the value is wrapped in curly-braces {} .

        parts = line.split("=", maxsplit=1)
        if len(parts) != 2:
            raise EnviFileFormatError(f'Line {line_no}:  not of the form "name = value"')

        name = normalize_name(parts[0])

        value = parts[1].strip()
        if value[-1] == "{":
            # The value is a multiline value.  Read in subsequent lines until
            # we see a line ending with "}".

            value_start_line = line_no
            while True:
                line_no += 1
                line = f.readline()
                if len(line) == 0:  # Are we at EOF?
                    raise EnviFileFormatError(
                        f'Line {value_start_line}:  EOF before reading entire value of "{name}"'
                    )

                line = line.strip()
                value += line
                if line[-1] == "}":
                    break

        metadata[name] = value

    # These attributes are integers
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
            metadata[k] = int(metadata[k])

    # These attributes are floats
    for k in ["data ignore value"]:
        if k in metadata:
            metadata[k] = float(metadata[k])

    # These attributes are lists of floats
    for k in ["wavelength", "correction factors", "smoothing factors", "fwhm"]:
        if k in metadata:
            metadata[k] = multivalue_to_list(metadata[k])

    # This attribute is a list of integers
    if "bbl" in metadata:
        metadata["bbl"] = multivalue_to_list(metadata["bbl"], int)

    return metadata


def load_envi_data(filename, header_filename=None, mmap=True):
    """
    Loads the specified ENVI data file from disk.  The filename is the path
    to the binary data file; the header file is assumed to be at the path
    filename + '.hdr'.

    Setting the mmap flag to True causes the data file to be memory-mapped;
    setting the flag to False causes it to be loaded entirely into memory.
    Memory-mapping the file will cause the function to return faster, because
    the data will be loaded incrementally as it is accessed, rather than
    needing to be loaded in its entirety up-front.
    """

    if header_filename is None:
        header_filename = filename + ".hdr"

    # Load the metadata so that we know what we're in for.
    with open(header_filename) as f:
        metadata = load_envi_header(f)

    # Figure out the numpy element type from the ENVI header
    numpy_type = map_envi_type_to_numpy_type(metadata["data type"], metadata["byte order"])

    # Load the binary data itself.
    header_offset = metadata["header offset"]
    with open(filename) as f:
        if mmap:
            data = numpy.memmap(f, numpy_type, "r", offset=header_offset)
        else:
            data = numpy.fromfile(f, numpy_type, offset=header_offset)

    # Figure out the shape of the 3D array, based on the interleaving.

    # TODO(donnie):  Need to provide additional abilities to retrieve spectra
    #     for (x, y) coordinates, etc.

    bands = metadata["bands"]
    samples = metadata["samples"]
    lines = metadata["lines"]

    interleave = metadata["interleave"]

    if interleave == "bsq":  # Band Sequential:  [bands, samples, lines]
        data.shape = (bands, samples, lines)
    elif interleave == "bip":  # Band-interleaved-by-pixel:  [samples, lines, bands]
        data.shape = (samples, lines, bands)
    elif interleave == "bil":  # Band-interleaved-by-line:  [lines, bands, samples]
        data.shape = (lines, bands, samples)
    else:
        raise EnviFileFormatError(f'Unrecognized ENVI interleave "{interleave}"')

    return (metadata, data)
