import os
import sys
import numpy as np
from osgeo import gdal
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")
from wiser.raster.loaders.envi import load_envi_header, write_envi_header
from enum import Enum

class Data_Interleave(Enum):
    BSQ = 0
    BIL = 1
    BIP = 2


def read_hdr_file(hdr_file):
    """
    Reads the .hdr file and extracts necessary metadata including multi-line wavelength information.
    Handles cases where wavelengths span multiple lines and filters out any invalid entries.
    """
    metadata = {}
    wavelengths = []
    in_wavelength_block = False  # Flag to track when we are inside the wavelength block
    
    with open(hdr_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect the start of the wavelength block
            if 'wavelength = {' in line.lower():
                in_wavelength_block = True
                line = line.replace('wavelength = {', '').strip()
            
            # If inside the wavelength block or just started it
            if in_wavelength_block:
                # Detect the end of the wavelength block on the same line or separate line
                if '}' in line:
                    line = line.replace('}', '').strip()
                    in_wavelength_block = False  # End of the wavelength block
                
                # Add the remaining wavelengths to the list
                if line:
                    wavelengths.extend(line.split(','))  # Add the line's wavelengths to the list
            
            # Process the rest of the metadata normally
            elif '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower()
                value = value.strip()
                
                # Handle cases where the value contains a list inside curly braces
                if '{' in value and '}' in value:
                    value = value.replace('{', '').replace('}', '').split(',')
                    value = [v.strip() for v in value]
                
                # Store key-value pairs in metadata
                metadata[key] = value

    # Clean up and convert the wavelength values to floats, filtering out empty entries
    wavelengths = [float(w.strip()) for w in wavelengths if w.strip()]

    return metadata, wavelengths

def write_hdr_file(hdr_file, metadata, bands, new_wavelengths):
    """
    Writes the updated metadata to a new .hdr file, including all fields and formatting the wavelength properly.
    Ensures alignment of numbers and commas using spaces for better readability.
    """
    # Prepare the HDR content starting with the ENVI header
    hdr_content = "ENVI\n"

    # Write all other metadata
    for key in metadata:
        value = metadata[key]
        if isinstance(value, list):
            hdr_content += f"{key} = {{{', '.join(map(str, value))}}}\n"
        else:
            hdr_content += f"{key} = {value}\n"

    # # Add the updated wavelength information, properly formatted
    # hdr_content += "wavelength = {\n"

    # # Determine the maximum number of digits before the decimal for proper alignment
    # max_integer_digits = max(len(str(int(w))) for w in new_wavelengths)

    # # Define the spacing between numbers, but only add as much as needed
    # for i in range(0, len(new_wavelengths), 6):
    #     wavelengths_line = new_wavelengths[i:i + 6]

    #     formatted_line = []
    #     for w in wavelengths_line:
    #         current_digits = len(str(int(w)))
    #         # Adjust the spacing to keep the numbers aligned, but without adding extra spaces
    #         padding = " " * (max_integer_digits - current_digits + 1)  # Only 1 space between numbers

    #         # Format the number with appropriate padding
    #         formatted_line.append(f"{padding}{w:.6f}")

    #     # Join the formatted numbers with commas and ensure correct spacing
    #     hdr_content += " " + ', '.join(formatted_line)

    #     # If this is the last line, close with a brace instead of a comma
    #     if i + 6 >= len(new_wavelengths):
    #         hdr_content += "}\n"
    #     else:
    #         hdr_content += ",\n"

    # Write the .hdr file
    with open(hdr_file, 'w') as f:
        f.write(hdr_content)

def multiply_band_dimension(image_cube, factor, metadata):
    """
    Stacks each 2D slice in the L dimension by repeating it 'factor' times.
    """
    bands = metadata['bands']
    samples = metadata['samples']
    lines = metadata['lines']

    interleave = metadata['interleave']
    print('interleave: ', interleave)
    if interleave == 'bsq':    # Band Sequential:  [bands, samples, lines]
        image_cube.shape = (bands, samples, lines)
    elif interleave == 'bip':  # Band-interleaved-by-pixel:  [samples, lines, bands]
        image_cube.shape = (samples, lines, bands)
        print("image_cube.shape before", image_cube.shape)
        image_cube = np.moveaxis(image_cube, 2, 0)
        print("image_cube.shape after", image_cube.shape)
    elif interleave == 'bil':  # Band-interleaved-by-line:  [lines, bands, samples]
        image_cube.shape = (lines, bands, samples)
        print(interleave)
        print("image_cube.shape before", image_cube.shape)
        image_cube = np.reshape(image_cube, (bands, samples, lines))
        print("image_cube.shape after", image_cube.shape)
    else:
        raise ValueError(f'Unrecognized ENVI interleave "{interleave}"')
    
    return np.repeat(image_cube, factor, axis=0)

def assert_slices_equal(image_cube):
    bands, samples, lines = image_cube.shape
    print('ASSERTING')
    for i in range(0, bands - 1, 2):  # Go through every other band
        assert np.array_equal(image_cube[i, :, :], image_cube[i + 1, :, :]), f"Slice {i} is not equal to slice {i + 1}"

def multiply_l_dimension(image_cube, factor):
    return np.repeat(image_cube, factor, axis=0)


def expand_wavelengths(wavelengths, factor):
    """
    Expands the wavelength list by adding intermediate wavelengths.
    The difference between consecutive wavelengths is divided by 'factor' and 
    used to generate intermediate values.
    """
    expanded_wavelengths = []
    incrememnt = 0.0
    for i in range(len(wavelengths) - 1):
        current_wavelength = wavelengths[i]
        next_wavelength = wavelengths[i + 1]
        increment = (next_wavelength - current_wavelength) / factor
        
        # Add the current wavelength and the intermediate wavelengths
        for j in range(factor):
            expanded_wavelengths.append(current_wavelength + j * increment)
    
    # Add the last wavelength to the expanded list
    current_wavelength = wavelengths[-1]
    for j in range(factor):
        expanded_wavelengths.append(current_wavelength + j * increment)
    
    return expanded_wavelengths

def expand_arr(arr, factor):
    """
    Expands the Nx1 arr to be factor*Nx1 where each entry is just copied to be next to itself
    factor times
    """
    return np.repeat(arr, factor, axis=0).tolist()

def expand_metadata_arrays(metadata, ignore, factor):
    """
    Loops through metadata dictionary, applies expand_arr to list entries except those in 'ignore',
    and updates metadata with the modified lists.
    """
    for key, value in metadata.items():
        if key not in ignore and isinstance(value, list):
            metadata[key] = expand_arr(value, factor)
    
    return metadata

def main(hdr_file, factor):
    # # Read the .hdr file and get metadata and wavelength
    # metadata, wavelengths = read_hdr_file(hdr_file)
    # print("metadata: ", metadata)
    # print("wavelengths: ", wavelengths)
    # print("wavelengths len: ", len(wavelengths))

    metadata = load_envi_header(hdr_file)
    # print("metadata: ", metadata)
    # print("Wavelengths: ", metadata['wavelength'])
    wavelengths = metadata['wavelength']
    interleave = metadata['interleave']

    expand_metadata_arrays(metadata, ["wavelength"], factor)
    # print("metadata after: ", metadata)
    if wavelengths is None:
        raise ValueError("Wavelength information not found in the .hdr file.")

    # Get file name and corresponding image cube file
    base_name = os.path.splitext(hdr_file)[0]
    image_cube_file = base_name  # Assuming no extension for the data file

    # Open the image cube using GDAL
    dataset = gdal.Open(image_cube_file)
    if not dataset:
        raise ValueError(f"Cannot open the file {image_cube_file}")

    # Read image cube into a NumPy array
    image_cube = dataset.ReadAsArray()

    # Multiply the L dimension by stacking each slice N times
    image_cube_expanded = multiply_band_dimension(image_cube, factor, metadata)
    # assert_slices_equal(image_cube_expanded)
    # Now all the data is in BSQ
    samples = image_cube_expanded.shape[1]
    lines = image_cube_expanded.shape[2]
    depth = image_cube_expanded.shape[0]
    # image_cube_expanded = multiply_l_dimension(image_cube, factor)
    tr = 50
    # print('image_cube: \n ', image_cube[tr:tr+3,tr:tr+3,tr:tr+3])
    # print('image_cube_expanded: \n ', image_cube_expanded[tr:tr+3*factor,tr:tr+3,tr:tr+3])
    # Expand the wavelength list by repeating each wavelength N times
    new_wavelengths = expand_wavelengths(wavelengths, factor)

    # Get metadata for writing the new file
    driver = gdal.GetDriverByName('ENVI')
    out_file = f"{base_name}_increased_bands_by_{factor}"

    # Create a new output dataset
    out_dataset = driver.Create(out_file, xsize=lines,
                                ysize=samples, bands=depth,
                                eType=gdal.GDT_Float32)

    # Write each band to the output file
    for i in range(image_cube_expanded.shape[0]):
        out_band = out_dataset.GetRasterBand(i + 1)
        out_band.WriteArray(image_cube_expanded[i, :, :])

    # Update metadata to put it into bsq
    metadata['samples'] = samples = image_cube_expanded.shape[1]
    metadata['lines'] = lines = image_cube_expanded.shape[2]
    metadata['bands'] = str(image_cube_expanded.shape[0])

    # Update number of bands
    metadata['wavelength'] = new_wavelengths  # Update wavelength information
    
    # Adjust metadata based on the output dataset's properties
    metadata['interleave'] = 'bsq'  # ENVI default is often 'bil', but update this if necessary
    # metadata['header offset'] = '0'  # Default header offset; change if necessary
    
    # Write the new .hdr file with updated metadata and wavelengths
    out_hdr_file = f"{out_file}.hdr"
    write_hdr_file(out_hdr_file, metadata, image_cube_expanded.shape[0], new_wavelengths)
    # write_envi_header(out_hdr_file, metadata)

    # Flush and close the datasets
    out_dataset.FlushCache()
    del out_dataset

    print(f"Output written to {out_file} and {out_hdr_file}")

if __name__ == "__main__":
    hdr_file = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"
    factor = 5
    main(hdr_file, factor)
