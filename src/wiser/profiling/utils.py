from osgeo import gdal, osr
import numpy as np
from netCDF4 import Dataset

# Create a sample NumPy array
array = np.array([[[0.  , 0.  , 0.  , 0.  ],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5 , 0.5 , 0.5 , 0.5 ],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.  , 1.  , 1.  , 1.  ]],

                [[0.  , 0.  , 0.  , 0.  ],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5 , 0.5 , 0.5 , 0.5 ],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.  , 1.  , 1.  , 1.  ]],

                [[0.  , 0.  , 0.  , 0.  ],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5 , 0.5 , 0.5 , 0.5 ],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.  , 1.  , 1.  , 1.  ]]])

def save_to_gtiff(array: np.ndarray, output_path="output.tiff"):

    # Define the driver for GeoTIFF
    driver = gdal.GetDriverByName("GTiff")

    # Create the dataset (raster) with 1 band
    # Specify the file name, data type, and dimensions
    bands, height, width = array.shape
    dataset = driver.Create(output_path, width, height, bands, gdal.GDT_Float32)

    if dataset is None:
        raise RuntimeError("Could not create output file.")

    # # Optionally, set geotransform and projection
    # # Example geotransform: (top-left x, pixel width, 0, top-left y, 0, pixel height)
    # geotransform = (0, 1, 0, 0, 0, -1)
    # dataset.SetGeoTransform(geotransform)

    # # Example spatial reference (WGS84)
    # spatial_ref = osr.SpatialReference()
    # spatial_ref.SetWellKnownGeogCS("WGS84")
    # dataset.SetProjection(spatial_ref.ExportToWkt())

    # Write the NumPy array to the first band (bands are 1-indexed in GDAL)
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(array[i, :, :])
        # Optionally, set nodata value
        band.SetNoDataValue(-9999)
        # Flush and close the dataset
        band.FlushCache()
    dataset.FlushCache()
    dataset = None  # Close the file

    print(f"Saved GeoTIFF to {output_path}")

def save_to_netcdf(array, output_filename):
    """
    Create a netCDF file from a 3D NumPy array using the netCDF4 Python package.
    This will produce multiple data variables, ensuring that 'reflectance' is 
    visible as a subdataset when opened with GDAL.

    Parameters
    ----------
    array : numpy.ndarray
        A 3D NumPy array of shape (bands, height, width).
    output_filename : str
        The path where the netCDF file will be saved.
    """
    if array.ndim != 3:
        raise ValueError("Input array must be a 3D array of shape (bands, height, width).")

    bands, height, width = array.shape

    # Create a new netCDF file
    ds = Dataset(output_filename, 'w', format='NETCDF4')

    # Create dimensions
    ds.createDimension('bands', bands)
    ds.createDimension('y', height)
    ds.createDimension('x', width)

    # Create the reflectance variable
    reflectance_var = ds.createVariable('reflectance', array.dtype, ('bands', 'y', 'x'))
    reflectance_var[:] = array

    # Create a dummy variable so that GDAL sees multiple data variables.
    # This ensures that each variable will appear as a subdataset.
    dummy_data = np.zeros_like(array)
    dummy_var = ds.createVariable('dummy', array.dtype, ('bands', 'y', 'x'))
    dummy_var[:] = dummy_data

    # Close the dataset
    ds.close()



if __name__ == "__main__":
    gtiff_out = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Small_Data\\output.tiff"
    save_to_gtiff(array, output_path=gtiff_out)

    netcdf_out = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Small_Data\\output.nc"
    # save_to_netcdf(array, netcdf_out)

    save_to_netcdf(array, netcdf_out)