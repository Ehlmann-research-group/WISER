

# We should have all calls to load_from_file go through here

# We app state should interface with this too

# The rest of the application should be able to query data from here. If that data doesn't exist and 
# we have space, we should just add in the data

# If we don't have space we will have to kick something out based on an eviction policy


'''
Context Pane
Zoom Pane
Main View all are Raster Panes

Raster Panes have multiple raster views, can set which one is there by ds-id

Raster views are fed raster datasets from the raster pane 

- Actually displaying the data requires: stretching, normalization, clipping, & *255.0
'''
from typing import OrderedDict as OrderedDictType, Union
from collections import OrderedDict

import numpy as np

class DataCache():

    # Initialize as a singleton (do this later)

    # Member varialbes should be a rendering cache dictionary and a computation cache dictionary

    # Keys for rendering cache are made from 

    # Keys from computation cache:
    #   - image cube keys: Image cube, dataset id
    #   - image band: RasterDataset, image band, is it normalized

    # Ability to look up an image band by band number and dataset

    # Ability to look up an image cube by dataset id

    # 

    # Default is to use 1GB (1,000,000) for rendering and 2GB (2,000,000)for computation size
    def __init__(self, render_memory_capacity=1000000, computation_memory_capacity=2000000):
        self._render_cache: OrderedDictType[int: Union[np.ndarray, np.ma.masked_array]] = OrderedDict()
        self._computation_cache: OrderedDictType[int: Union[np.ndarray, np.ma.masked_array]] = OrderedDict()
        self._render_capacity = render_memory_capacity
        self._render_size = 0
        self._computation_capacity = computation_memory_capacity
        self._computation_size = 0
    
    def _evict_computation_cache(self):
        while self._computation_size > self._computation_capacity:
            if not self._computation_cache:
                break
            key, value = self._computation_cache.popitem(last=False)
            self._computation_size -= value.nbytes
    
    def _add_computation_item(self, key: int, value: Union[np.ndarray, np.ma.masked_array]):
        self._computation_cache[key] = value
        self._computation_size += value.nbytes

    def remove_image_band(self, band_index: int, dataset):
        key = hash(band_index, dataset)
        if self._computation_cache[key] is not None:
            self._computation_size -= self._computation_cache[key].nbytes
            del self._computation_cache[key]

    def get_image_band(self, band_index: int, dataset):
        key = hash((band_index, dataset))

        if key in self._computation_cache:
            return self._computation_cache[key]
        else:
            data_size = dataset.get_band_memory_size()
            if data_size > self._computation_capacity:
                raise RuntimeError(f'Size of data exceeds cache size: {data_size} > {self._computation_capacity}')
            if self._computation_size + data_size > self._computation_capacity:
            # We want to retrieve the band from the dictionary after we have evicted enough items
            # to house this one
                self._evict_computation_cache()
            data = dataset.get_band_data(band_index)
            self._add_computation_item(key, data)
            return data
    
    def remove_image_cube(self, dataset):
        key = hash((dataset))
        self._computation_size -= self._computation_cache[key].nbytes
        del self._computation_cache[key]
    
    def get_image_cube(self, dataset):
        key = hash((dataset))

        if key in self._computation_cache:
            return self._computation_cache[key]
        else:
            data_size = dataset.get_memory_size()
            if data_size > self._computation_capacity:
                raise RuntimeError(f'Size of data exceeds cache size: {data_size} > {self._computation_capacity}')
            if self._computation_size + data_size > self._computation_capacity:
                self._evict_computation_cache()
            data = dataset.get_image_data()
            self._add_computation_item(key, data)
            return data


    

