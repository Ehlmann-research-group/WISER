import os

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from osgeo import gdal, gdalconst, gdal_array

from .dataset import RasterDataSet
from .dataset_impl import RasterDataImpl, ENVI_GDALRasterDataImpl, NumPyRasterDataImpl

from .spectrum import Spectrum


class RasterDataLoader:
    '''
    A loader for loading 2D raster data-sets from the local filesystem, using
    GDAL (Geospatial Data Abstraction Library) for reading the data.
    '''

    def __init__(self):
        # This is a counter so we can generate names for unnamed datasets.
        self._unnamed_datasets: int = 0


    def _name_dataset(self, dataset: RasterDataSet) -> None:
        files = dataset.get_filepaths()
        if files:
            name = os.path.basename(files[0])
        else:
            self._unnamed_datasets += 1
            name = f'unnamed {self._unnamed_datasets}'

        dataset.set_name(name)


    def load_from_file(self, path):
        '''
        Load a raster data-set from the specified path.  Returns a
        :class:`RasterDataSet` object.
        '''

        # TODO(donnie):  For now, assume we have a file path.
        # TODO(donnie):  Use urllib.parse.urlparse(urlstring) to parse URLs.

        # ENVI files:  GDAL doesn't like dealing with the ".hdr" files, so if we
        # are given a ".hdr" file, try to find the corresponding data file.
        if path.endswith('.hdr'):
            s = path[:-4]
            if os.path.isfile(s):
                path = s
            else:
                s = s + '.img'
                if os.path.isfile(s):
                    path = s
                else:
                    raise ValueError('Can\'t find raster file corresponding ' +
                                     f'to ENVI header file {path}')

        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        gdal_dataset = gdal.OpenEx(path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['ENVI', 'GTiff', 'PDS', 'PDS4'])

        impl = None
        driver_name = gdal_dataset.GetDriver().ShortName
        if driver_name == 'ENVI':
            impl = ENVI_GDALRasterDataImpl(gdal_dataset)
        else:
            raise ValueError(f'Unsupported format "{driver_name}"')

        ds = RasterDataSet(impl)
        self._name_dataset(ds)
        return ds


    def get_save_filenames(self, path: str, format: str = 'ENVI') -> List[str]:
        if format == 'ENVI':
            return ENVI_GDALRasterDataImpl.get_save_filenames(path)
        else:
            raise ValueError(f'Unsupported format "{format}"')


    def save_dataset_as(self, dataset: RasterDataSet, path: str, format: str) -> None:
        if format == 'ENVI':
            return ENVI_GDALRasterDataImpl.save_dataset_as(dataset, path)
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
        ds = RasterDataSet(impl)
        self._name_dataset(ds)

        return ds


    def spectrum_from_numpy_array(self, arr: np.ndarray) -> Spectrum:
        return None
