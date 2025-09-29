import logging
import os

from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

from osgeo import gdal

from .dataset import RasterDataSet
from .dataset_impl import (RasterDataImpl, ENVI_GDALRasterDataImpl,
    GTiff_GDALRasterDataImpl, NumPyRasterDataImpl, NetCDF_GDALRasterDataImpl,
    ASC_GDALRasterDataImpl, JP2_GDALRasterDataImpl, PDS3_GDALRasterDataImpl,
    PDS4_GDALRasterDataImpl, GDALRasterDataImpl, JP2_GDAL_PDR_RasterDataImpl
    )

from wiser.gui.fits_loading_dialog import FitsDatasetLoadingDialog

from PySide2.QtWidgets import QDialog

if TYPE_CHECKING:
    from wiser.raster.dataset import DataCache

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
            'ASCII': ASC_GDALRasterDataImpl,
            'JP2': JP2_GDALRasterDataImpl,
            'PDS3': PDS3_GDALRasterDataImpl,
            'PDS4': PDS4_GDALRasterDataImpl,
        }
    
        # What to do when loading in each file format
        self._format_loaders = {
            ENVI_GDALRasterDataImpl: self.load_normal_dataset, 
            GTiff_GDALRasterDataImpl: self.load_normal_dataset, 
            NetCDF_GDALRasterDataImpl: self.load_normal_dataset,
            ASC_GDALRasterDataImpl: self.load_normal_dataset,
            JP2_GDALRasterDataImpl: self.load_normal_dataset,
            PDS3_GDALRasterDataImpl: self.load_normal_dataset,
            PDS4_GDALRasterDataImpl: self.load_normal_dataset,
            GDALRasterDataImpl: self.load_normal_dataset,
        }

        # This is a counter so we can generate names for unnamed datasets.
        self._unnamed_datasets: int = 0


    def load_normal_dataset(self, impl: RasterDataImpl, data_cache: 'DataCache') -> List[RasterDataSet]:
        '''
        The normal way to load in a dataset
        '''

        # This returns a list because load_FITS_dataset could possibly return a list
        return [RasterDataSet(impl, data_cache)]
    
    def load_FITS_dataset(self, impl: RasterDataImpl, data_cache: 'DataCache') -> List[RasterDataSet]:
        # We should show the Fits dialog which should return to us
        self._fits_dialog = FitsDatasetLoadingDialog(impl, data_cache)
        result = self._fits_dialog.exec()
    
        if result == QDialog.Accepted:
           return self._fits_dialog.return_datasets
        return []


    def load_from_file(self, path, subdataset_name = '', data_cache = None, interactive = True) -> List[RasterDataSet]:
        '''
        Load a raster data-set from the specified path.  Returns a
        list of :class:`RasterDataSet` object.
        '''

        # Iterate through all supported formats, and try to use each one to
        # load the raster data.
        impl_list = None
        for (driver_name, impl_type) in self._formats.items():
            try:
                if subdataset_name:
                    print(f"using subdataset name: {subdataset_name}")
                    impl_list = impl_type.try_load_file(path, subdataset_name=subdataset_name, interactive=interactive)
                else:
                    print(f"not using subdataset name")
                    impl_list = impl_type.try_load_file(path, interactive=interactive)
            except Exception as e:
                logger.debug(f'Couldn\'t load file {path} with driver ' +
                             f'{driver_name} and implementation {impl_type}.', e)
                
        # Try luck with gdal
        try:
            if impl_list is None:
                impl_list = GDALRasterDataImpl.try_load_file(path, interactive=interactive)
        except Exception as e:
            logger.debug(f'Couldn\'t load file {path} with driver ' +
                            f'{driver_name} and implementation {impl_type}.', e)

        if impl_list is None:
            raise Exception(f'Couldn\'t load file {path}:  unsupported format')

        # Used if a dataset contains multiple subdatasets and we want to load all of them
        outer_datasets = []
        for impl in impl_list:
            func = self._format_loaders[type(impl)]
            datasets = func(impl, data_cache)
            for ds in datasets:
                files = ds.get_filepaths()
                if files:
                    name = os.path.basename(files[0])
                else:
                    name = os.path.basename(path)
                subdataset_name = ds.get_subdataset_name()
                if subdataset_name is not None:
                    name += ":" + subdataset_name.split(":")[-1]

                ds.set_name(name)
                outer_datasets.append(ds)

        return outer_datasets


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


    def dataset_from_numpy_array(self, arr: np.ndarray, cache: 'DataCache') -> RasterDataSet:
        '''
        Given a NumPy ndarray, this function returns a RasterDataSet object
        that uses the array for its raster data.  The input ndarray must have
        three dimensions; they are interpreted as
        [spectral][spatial_y][spatial_x].

        Raises a ValueError if the input array doesn't have 3 dimensions.
        '''

        if len(arr.shape) != 3:
            raise ValueError('NumPy array must have 3 dimensions')

        impl = NumPyRasterDataImpl(arr)
        return RasterDataSet(impl, cache)

    def dataset_from_gdal_dataset(self, dataset: gdal.Dataset, cache: 'DataCache') -> RasterDataSet:
        impl = ENVI_GDALRasterDataImpl(dataset)
        return RasterDataSet(impl, cache)

    # TODO(donnie):  Not presently needed - can instantiate a NumPyArraySpectrum
    #     object from a NumPy array...
    # def spectrum_from_numpy_array(self, arr: np.ndarray) -> Spectrum:
    #     return None
