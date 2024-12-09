import logging
import os

from typing import Any, Dict, List

import numpy as np

from osgeo import gdal

from .dataset import RasterDataSet
from .dataset_impl import (RasterDataImpl, ENVI_GDALRasterDataImpl,
    GTiff_GDALRasterDataImpl, NumPyRasterDataImpl, NetCDF_GDALRasterDataImpl,
    JP2_GDALRasterDataImpl, FITS_GDALRasterDataImpl, PDS3_GDALRasterDataImpl)



logger = logging.getLogger(__name__)


class RasterDataLoader:
    '''
    A loader for loading 2D raster data-sets from the local filesystem, using
    GDAL (Geospatial Data Abstraction Library) for reading the data.
    '''

    def __init__(self):
        # File formats that we recognize.
        self._formats = {
            'ENVI': ENVI_GDALRasterDataImpl,
            'GTiff': GTiff_GDALRasterDataImpl,
            'NetCDF': NetCDF_GDALRasterDataImpl,
            'JP2': JP2_GDALRasterDataImpl,
            'FITS': FITS_GDALRasterDataImpl,
            'PDS3': PDS3_GDALRasterDataImpl,
        }

        # This is a counter so we can generate names for unnamed datasets.
        self._unnamed_datasets: int = 0


    def load_from_file(self, path, data_cache = None):
        '''
        Load a raster data-set from the specified path.  Returns a
        :class:`RasterDataSet` object.
        '''

        # Iterate through all supported formats, and try to use each one to
        # load the raster data.
        impl_list = None
        for (driver_name, impl_type) in self._formats.items():
            try:
                impl_list = impl_type.try_load_file(path)

            except Exception as e:
                print(f"Exception: \n {e}")
                logger.debug(f'Couldn\'t load file {path} with driver ' +
                             f'{driver_name} and implementation {impl_type}.', e)

        if impl_list is None:
            raise Exception(f'Couldn\'t load file {path}:  unsupported format')

        datasets = []
        for impl in impl_list:
            ds = RasterDataSet(impl, data_cache)
            files = ds.get_filepaths()
            if files:
                name = os.path.basename(files[0])
            else:
                name = os.path.basename(path)
            subdataset_name = ds.get_subdataset_name()
            if subdataset_name is not None:
                name += ":" + subdataset_name.split(":")[-1]

            ds.set_name(name)
            datasets.append(ds)

        return datasets


    def get_save_filenames(self, path: str, format: str = 'ENVI') -> List[str]:
        if format == 'ENVI':
            return ENVI_GDALRasterDataImpl.get_save_filenames(path)
        else:
            raise ValueError(f'Unsupported format "{format}"')


    def save_dataset_as(self, dataset: RasterDataSet, path: str, format: str,
            config: Dict[str, Any]) -> ENVI_GDALRasterDataImpl:
        if format == 'ENVI':
            return ENVI_GDALRasterDataImpl.save_dataset_as(dataset, path, config)
        else:
            raise ValueError(f'Unsupported format "{format}"')


    def dataset_from_numpy_array(self, arr: np.ndarray, cache) -> RasterDataSet:
        '''
        Given a NumPy ndarray, this function returns a RasterDataSet object
        that uses the array for its raster data.  The input ndarray must have
        three dimensions; they are interpreted as
        [spatial_y][spatial_x][spectral].

        Raises a ValueError if the input array doesn't have 3 dimensions.
        '''

        if len(arr.shape) != 3:
            raise ValueError('NumPy array must have 3 dimensions')

        impl = NumPyRasterDataImpl(arr)
        return RasterDataSet(impl, cache)

    def dataset_from_gdal_dataset(self, dataset: gdal.Dataset, cache) -> RasterDataSet:
        impl = ENVI_GDALRasterDataImpl(dataset)
        return RasterDataSet(impl, cache)

    # TODO(donnie):  Not presently needed - can instantiate a NumPyArraySpectrum
    #     object from a NumPy array...
    # def spectrum_from_numpy_array(self, arr: np.ndarray) -> Spectrum:
    #     return None
