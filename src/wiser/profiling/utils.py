import os
import numpy as np
from osgeo import gdal

def multiply_bands(image_cube, factor):
    """
    Stacks the image cube along the band dimension by repeating it 'factor' times.
    """
    return np.concatenate([image_cube] * factor, axis=0)

def main(hdr_file, factor):
    # # Read the .hdr file
    # metadata = read_hdr_file(hdr_file)

    # Get file name and corresponding image cube file
    base_name = os.path.splitext(hdr_file)[0]
    image_cube_file = base_name  # Assuming no extension for the data file

    # Open the image cube using GDAL
    dataset = gdal.Open(image_cube_file)
    if not dataset:
        raise ValueError(f"Cannot open the file {image_cube_file}")

    # Read image cube into a NumPy array
    image_cube = dataset.ReadAsArray()

    print("image_cube.shape ", image_cube.shape)

    # Multiply the bands by stacking the cube along the bands dimension
    image_cube_stacked = multiply_bands(image_cube, factor)

    print("image_cube_stacked.shape ", image_cube_stacked.shape)

    # Get metadata for writing the new file
    fileformat = 'ENVI'
    driver = gdal.GetDriverByName(fileformat)
    metadata = driver.GetMetadata()
    if metadata.get(gdal.DCAP_CREATE) == "YES":
        print("Driver {} supports Create() method.".format(fileformat))
    out_file = f"{base_name}_multiplied{factor}"
    print('metadata: ', dataset.GetMetadata())

    # Create a new output dataset
    out_dataset = driver.Create(out_file, xsize=image_cube_stacked.shape[2],
                                ysize=image_cube_stacked.shape[1], bands=image_cube_stacked.shape[0],
                                eType=gdal.GDT_Float32)

    # Write each band to the output file
    for i in range(image_cube_stacked.shape[0]):
        out_band = out_dataset.GetRasterBand(i + 1)
        out_band.WriteArray(image_cube_stacked[i, :, :])

    # Flush and close the datasets
    out_dataset.FlushCache()
    del out_dataset

    print(f"Output written to {out_file}")

if __name__ == "__main__":
    hdr_file = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"  # Replace with the path to your .hdr file
    factor = 3  # Replace with the desired multiplication factor
    main(hdr_file, factor)
