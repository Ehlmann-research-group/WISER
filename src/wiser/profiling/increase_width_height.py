import os
import sys
import numpy as np
from osgeo import gdal
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")
from wiser.raster.loaders.envi import load_envi_header


def read_hdr_file(hdr_file):
    """
    Reads the .hdr file and extracts necessary metadata including multi-line wavelength information.
    """
    metadata = {}
    wavelengths = []
    in_wavelength_block = False  # Flag to track when we are inside the wavelength block
    
    with open(hdr_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if 'wavelength = {' in line.lower():
                in_wavelength_block = True
                line = line.replace('wavelength = {', '').strip()
            
            if in_wavelength_block:
                if '}' in line:
                    line = line.replace('}', '').strip()
                    in_wavelength_block = False
                
                if line:
                    wavelengths.extend(line.split(','))
            
            elif '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if '{' in value and '}' in value:
                    value = value.replace('{', '').replace('}', '').split(',')
                    value = [v.strip() for v in value]
                
                metadata[key] = value

    wavelengths = [float(w.strip()) for w in wavelengths if w.strip()]

    return metadata, wavelengths

def write_hdr_file(hdr_file, metadata, bands, new_wavelengths):
    essential_keys = ['samples', 'lines', 'bands', 'header offset', 'file type', 'data type',
                      'interleave', 'byte order', 'wavelength units', 'map info', 
                      'coordinate system string', 'y start']
    
    hdr_content = "ENVI\n"
    
    for key in metadata:
        value = metadata[key]
        if isinstance(value, list):
            hdr_content += f"{key} = {{{', '.join(map(str, value))}}}\n"
        else:
            hdr_content += f"{key} = {value}\n"
    
    hdr_content += "wavelength = {\n"
    hdr_content += ', '.join(map(str, new_wavelengths)) + "\n}\n"
    
    with open(hdr_file, 'w') as f:
        f.write(hdr_content)

# def multiply_w_h_dimensions(image_cube, factor):
#     """
#     Multiplies the W and H dimensions of the image cube by repeating each slice 'factor' times along both axes.
#     """
#     # Expand along the width (axis=2)
#     expanded_width = np.repeat(image_cube, factor, axis=2)
    
#     # Expand along the height (axis=1)
#     expanded_w_h = np.repeat(expanded_width, factor, axis=1)
    
#     return expanded_w_h

def multiply_w_h_dimensions(image_cube, factor, bands_list = None):
    """
    Multiplies the W and H dimensions of the specified slices of the image cube (based on bands_list) by repeating each
    slice 'factor' times along both axes.
    
    Parameters:
        image_cube (numpy.ndarray): The 3D image cube (bands x height x width).
        factor (int): The factor by which to multiply the width and height dimensions.
        bands_list (list of int): List of band indices to select slices from the image cube.
    
    Returns:
        numpy.ndarray: The expanded image cube with the selected bands and expanded width and height dimensions.
    """
    # Slice the image cube to select only the bands in bands_list
    if bands_list is not None:
        selected_bands = image_cube[bands_list, :, :]
    else:
        selected_bands = image_cube
    
    # Expand along the width (axis=2)
    expanded_width = np.repeat(selected_bands, factor, axis=2)
    
    # Expand along the height (axis=1)
    expanded_w_h = np.repeat(expanded_width, factor, axis=1)
    
    return expanded_w_h

def expand_wavelengths(wavelengths, factor):
    expanded_wavelengths = []
    
    for i in range(len(wavelengths) - 1):
        current_wavelength = wavelengths[i]
        next_wavelength = wavelengths[i + 1]
        increment = (next_wavelength - current_wavelength) / factor
        
        for j in range(factor):
            expanded_wavelengths.append(current_wavelength + j * increment)
    
    expanded_wavelengths.append(wavelengths[-1])
    
    return expanded_wavelengths

def main(hdr_file, factor, bands_list = None):
    # # Read the .hdr file and get metadata and wavelength
    # metadata, wavelengths = read_hdr_file(hdr_file)
    # print("metadata: ", metadata)
    # print("wavelengths: ", wavelengths)
    # print("wavelengths len: ", len(wavelengths))
    
    print("load_envi_header: ", load_envi_header(hdr_file))
    metadata = load_envi_header(hdr_file)
    print("Wavelengths: ", metadata['wavelength'])
    raw_wavelength = np.array(metadata['wavelength'])
    wavelengths = raw_wavelength if bands_list is None else raw_wavelength[bands_list]
    print("New Wavelengths: ", wavelengths)

    if wavelengths is None:
        raise ValueError("Wavelength information not found in the .hdr file.")

    base_name = os.path.splitext(hdr_file)[0]
    image_cube_file = base_name

    dataset = gdal.Open(image_cube_file)
    if not dataset:
        raise ValueError(f"Cannot open the file {image_cube_file}")

    image_cube = dataset.ReadAsArray()

    # Multiply the W and H dimensions
    image_cube_expanded = multiply_w_h_dimensions(image_cube, factor, bands_list)

    # Expand the wavelength list
    new_wavelengths = wavelengths

    driver = gdal.GetDriverByName('ENVI')
    out_file = f"{base_name}_expanded_lines_and_samples_{factor}"
    print(f"Num bands: {image_cube_expanded.shape[0]}")
    print("About to write")
    # Create the new output dataset
    out_dataset = driver.Create(out_file, xsize=image_cube_expanded.shape[2],
                                ysize=image_cube_expanded.shape[1], bands=image_cube_expanded.shape[0],
                                eType=gdal.GDT_Float32)

    # Write each band to the output file
    for i in range(image_cube_expanded.shape[0]):
        print(f"i: {i}")
        out_band = out_dataset.GetRasterBand(i + 1)
        out_band.WriteArray(image_cube_expanded[i, :, :])
    out_dataset.FlushCache()
    del out_dataset

    metadata['samples'] = str(image_cube_expanded.shape[2])  # Update width
    metadata['lines'] = str(image_cube_expanded.shape[1])  # Update height
    metadata['bands'] = str(image_cube_expanded.shape[0])  # Update number of bands
    metadata['wavelength'] = new_wavelengths
    
    metadata['interleave'] = 'bsq'
    metadata['header offset'] = '0'

    out_hdr_file = f"{out_file}.hdr"
    write_hdr_file(out_hdr_file, metadata, image_cube_expanded.shape[0], new_wavelengths)


    print(f"Output written to {out_file} and {out_hdr_file}")

if __name__ == "__main__":
    hdr_file = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"
    # hdr_file = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\C5705B-00003Z-01_2018_07_28_14_18_38_VNIRcalib.hdr"
    factor = 20
    main(hdr_file, factor, [1, 2, 3])