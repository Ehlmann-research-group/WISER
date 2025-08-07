import numpy as np
import time
import os
from osgeo import gdal

from wiser.plugins.types import BatchProcessingPlugin, BatchProcessingInputType, BatchProcessingOutputType

class AverageBandsPlugin(BatchProcessingPlugin):
    def __init__(self):
        super().__init__()

    def get_ordered_input_types(self):
        return [BatchProcessingInputType.IMAGE_BAND]
    
    def get_ordered_output_types(self):
        return [BatchProcessingOutputType.IMAGE_BAND]
    
    def process(self, *args):
        # args[0] is the input filepath
        in_path = args[0]
        # 1) open for reading
        ds = gdal.Open(in_path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"GDAL failed to open '{in_path}'")

        # 2) read all bands into a 3D array (bands, rows, cols)
        bands = []
        for i in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(i)
            arr = band.ReadAsArray()
            bands.append(arr)
        stack = np.stack(bands, axis=0)

        # 3) compute per-pixel average across bands
        avg = stack.mean(axis=0)

        # 4) build output path
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_{self.__class__.__name__}{ext}"

        # 5) create output dataset with the same geospatial metadata
        driver = gdal.GetDriverByName(ds.GetDriver().ShortName)
        out_ds = driver.Create(
            out_path,
            ds.RasterXSize,
            ds.RasterYSize,
            1,                          # one band
            ds.GetRasterBand(1).DataType
        )
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())

        # write the averaged array
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(avg)
        out_band.FlushCache()

        # 6) clean up
        ds = None
        out_ds = None

        time.sleep(10)

        # return the path to your new single-band file
        return (out_path,)
