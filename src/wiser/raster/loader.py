import logging
import os

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from osgeo import gdal, gdalconst, gdal_array

from .dataset import RasterDataSet
from .dataset_impl import (RasterDataImpl, ENVI_GDALRasterDataImpl,
    GTiff_GDALRasterDataImpl, NumPyRasterDataImpl)

from .spectrum import Spectrum


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
        }

        # This is a counter so we can generate names for unnamed datasets.
        self._unnamed_datasets: int = 0


    def load_from_file(self, path):
        '''
        Load a raster data-set from the specified path.  Returns a
        :class:`RasterDataSet` object.
        '''

        # Iterate through all supported formats, and try to use each one to
        # load the raster data.
        impl = None
        for (driver_name, impl_type) in self._formats.items():
            try:
                print("loader load_from_file start")
                impl = impl_type.try_load_file(path)
                print("loader load_from_file end")

            except Exception as e:
                logger.debug(f'Couldn\'t load file {path} with driver ' +
                             f'{driver_name} and implementation {impl_type}.', e)

        if impl is None:
            raise Exception(f'Couldn\'t load file {path}:  unsupported format')

        ds = RasterDataSet(impl)
        files = ds.get_filepaths()
        if files:
            name = os.path.basename(files[0])
        else:
            name = os.path.basename(path)
        ds.set_name(name)

        return ds


    def get_save_filenames(self, path: str, format: str = 'ENVI') -> List[str]:
        if format == 'ENVI':
            return ENVI_GDALRasterDataImpl.get_save_filenames(path)
        else:
            raise ValueError(f'Unsupported format "{format}"')


    def save_dataset_as(self, dataset: RasterDataSet, path: str, format: str,
            config: Dict[str, Any]) -> None:
        if format == 'ENVI':
            return ENVI_GDALRasterDataImpl.save_dataset_as(dataset, path, config)
        else:
            raise ValueError(f'Unsupported format "{format}"')


    def dataset_from_numpy_array(self, arr: np.ndarray) -> RasterDataSet:
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
        return RasterDataSet(impl)

    def dataset_from_gdal_dataset(self, dataset: gdal.Dataset) -> RasterDataSet:
        impl = ENVI_GDALRasterDataImpl(dataset)
        return RasterDataSet(impl)

    # TODO(donnie):  Not presently needed - can instantiate a NumPyArraySpectrum
    #     object from a NumPy array...
    # def spectrum_from_numpy_array(self, arr: np.ndarray) -> Spectrum:
    #     return None
