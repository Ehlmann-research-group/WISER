from typing import List

import numpy as np

from wiser.raster.stretch import StretchBase

class RasterViewMetaData:

    def __init__(self, ds_id: int = None, display_bands: List[int] = None, \
               stretches: List[StretchBase] = None, img_data: np.ndarray = None, \
                colormap = None):
        self._ds_id = ds_id
        self._display_bands = display_bands
        self._stretches = stretches
        self._img_data = img_data
        self._colormap = colormap
    
    def is_fully_initialized(self) -> bool:
        return self._ds_id is not None and self._display_bands is not None and self._stretches is not None

    def get_ds_id(self):
        return self._ds_id
    
    def get_image_data(self):
        return self._img_data

    def set_image_data(self, image_data):
        self._img_data = image_data

    def __str__(self):
        return (f"RasterViewMetaData(ds_id={self._ds_id}, "
                f"display_bands={self._display_bands}, "
                f"stretches={self._stretches})")

    def __eq__(self, other):
        if isinstance(other, RasterViewMetaData):
            return (self._ds_id == other._ds_id and
                    self._display_bands == other._display_bands and
                    self._stretches == other._stretches and 
                    self._colormap == other._colormap)
        return False