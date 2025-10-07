import numpy as np
from osgeo import gdal
from netCDF4 import Dataset

def create_raster(array, filename, driver_name, wavelength_units='nm'):
    """
    Create a GDAL dataset from a 3D NumPy array using the specified driver.
    The array shape is assumed to be (bands, rows, cols).
    """
    # Parse array dimensions
    bands, rows, cols = array.shape

    # Select the driver
    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"GDAL driver '{driver_name}' is not available.")

    # Create the dataset
    # eType=gdal.GDT_Float32 for floating-point data
    dataset = driver.Create(filename, cols, rows, bands, gdal.GDT_Float32)
    if dataset is None:
        raise RuntimeError(f"Could not create dataset with driver '{driver_name}'.")

    # Write each band to the dataset
    for b in range(bands):
        band_data = array[b, :, :]
        out_band = dataset.GetRasterBand(b + 1)
        out_band.WriteArray(band_data)
        # Optionally set NoData value
        out_band.SetNoDataValue(-9999)
        out_band.FlushCache()

    # Optionally set geotransform/projection here if needed
    # e.g., dataset.SetGeoTransform([...])
    #       dataset.SetProjection(srs.ExportToWkt())

    # Close the dataset
    dataset.FlushCache()
    dataset = None

def create_netcdf_from_3d_array(array, output_filename):
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

# 1. Create a 10x11x12 NumPy array
#    Interpreted as 10 bands, each of shape 11x12
bands = 5
rows = 40
cols = 60
data = np.arange(bands * rows * cols, dtype=np.float32).reshape((bands, rows, cols))
wavelength_units = "nm"

def main():

    # Write to ENVI (.hdr)
    # Note that GDAL's ENVI driver will produce "test_envi.hdr" and "test_envi"
    # as the actual binary data (no extension). 
    filepath = f'test_datasets/envi_{bands}_{rows}_{cols}'
    create_raster(data, filepath, "ENVI")

    # Write to GeoTIFF (.tiff)
    filepath = f'test_datasets/gtiff_{bands}_{rows}_{cols}.tiff'
    create_raster(data, "test_datasets/gtiff.tiff", "GTiff")

    # Write to NetCDF 
    # Note that this currently doesn't work.
    # create_netcdf_from_3d_array(data, "test_datasets/netcdf.nc")

if __name__ == "__main__":
    main()