from typing import OrderedDict as OrderedDictType, Union, Tuple, Dict, List
from collections import OrderedDict

import numpy as np

import abc

class Cache:

    # 3,000,000,000 is 3GB
    def __init__(self, capacity: int = 3000000000):
        self._capacity = capacity
        self._size = 0
        self._cache: OrderedDictType[int: Union[np.ndarray, np.ma.masked_array]] = OrderedDict()
        self._key_lookup_table: Dict[int: List[int]] = {}
    
    def _evict(self):
        while self._size > self._capacity:
            if not self._cache:
                break
            key, value = self._cache.popitem(last=False)
            self._size -= value.nbytes

    def clear_cache(self):
        for key in self._cache:
            self._size -= self._cache[key].nbytes
            del self._cache[key]
        assert np.isclose(self._size, 0.0) 

    def remove_cache_item(self, key: int):
        if key in self._cache:
            self._size -= self._cache[key].nbytes
            del self._cache[key]

    def add_cache_item(self, key: int, value: Union[np.ndarray, np.ma.masked_array]):
        data_size = value.nbytes
        if data_size > self._capacity:
            raise RuntimeError(f'Size of data exceeds cache size: {data_size} > {self._capacity}')
        if self._size + data_size > self._capacity:
            self._evict()
        self._cache[key] = value
        self._size += value.nbytes

    def get_cache_key(self, *args):
        raise NotImplementedError(f'get_cache_key is not implemented for {type(self)}')
    
    def get_cache_item(self, key: int):
        if key in self._cache:
            return self._cache[key]
        return None
    
    def in_cache(self, key: int):
        return key in self._cache
    
    def get_partial_key(self, dataset):
        raise NotImplementedError(f'get_partial_key is not implemented for {type(self)}')
    
    def lookup_keys(self, partial_key: int):
        return self._key_lookup_table[partial_key]
    
    def clear_keys_from_partial(self, partial_key: int):
        keys = self.lookup_keys(partial_key)
        for key in keys:
            self.remove_cache_item(key)

class RenderCache(Cache):
    
    def __init__(self, capacity: int = 3000000000):
        super().__init__(capacity)
        self._key_lookup_table: Dict[int: List[int]] = {}

    def get_cache_key(self, dataset, band_tuple: Tuple[int], stretches):
        '''
        Creates and returns the cache key based on the above entries. It also
        stores the cache key with the dataset so that we can look up all the keys
        that correspond to the dataset. This makes it easier to delete things when 
        we delete a dataset.
        '''
        partial_key = self.get_partial_key(dataset)
        cache_key = hash((dataset, *band_tuple, *stretches))
        if partial_key not in self._key_lookup_table:
            self._key_lookup_table[partial_key] = []
        self._key_lookup_table[partial_key].append(cache_key)
        return cache_key
    
    def get_partial_key(self, dataset):
        return hash((dataset))


class ComputationCache(Cache):

    def __init__(self, capacity: int = 7000000000):
        super().__init__(capacity)

    def get_cache_key(self, dataset, band_index: int = -1):
        partial_key = self.get_partial_key(dataset)
        cache_key = hash((dataset, band_index))
        if partial_key not in self._key_lookup_table:
            self._key_lookup_table[partial_key] = []
        self._key_lookup_table[partial_key].append(cache_key)
        return hash((dataset, band_index))
    
    def get_partial_key(self, dataset):
        return hash((dataset))

class HistogramCache(Cache):
    def __init__(self, capacity: int = 100000000):
        '''
        Initialize with capacity of 100 MB
        '''
        super().__init__(capacity)

    def _evict(self):
        while self._size > self._capacity:
            if not self._cache:
                break
            key, values = self._cache.popitem(last=False)
            for value in values:
                self._size -= value.nbytes

    def clear_cache(self):
        for key in self._cache:
            values = self._cache[key]
            for value in values:
                self._size -= value.nbytes
            del self._cache[key]
        assert np.isclose(self._size, 0.0) 

    def remove_cache_item(self, key: int):
        if key in self._cache:
            values = self._cache[key]
            for value in values:
                self._size -= value.nbytes
            del self._cache[key]

    def add_cache_item(self, key: int, values: Tuple[Union[np.ndarray, np.ma.masked_array]]):
        '''
        The first value in values should be the histogram bins and the second should be
        the histogram edges
        '''
        data_size = 0
        for value in values:
            data_size += value.nbytes
        if data_size > self._capacity:
            raise RuntimeError(f'Size of data exceeds cache size: {data_size} > {self._capacity}')
        if self._size + data_size > self._capacity:
            self._evict()
        self._cache[key] = values
        for value in values:
            self._size += value.nbytes
    
    def get_cache_key(self, dataset,band_index: int, stretch_type, conditioner_type):
        partial_key = self.get_partial_key(dataset)
        cache_key = hash((dataset, band_index, stretch_type, conditioner_type))
        if partial_key not in self._key_lookup_table:
            self._key_lookup_table[partial_key] = []
        self._key_lookup_table[partial_key].append(cache_key)
        return cache_key
    
    def get_partial_key(self, dataset):
        return hash((dataset))

class DataCache():

    def __init__(self, render_capacity:int = 3000000000, computation_capacity: int=7000000000):
        self._render_cache = RenderCache(render_capacity)
        self._computation_cache = ComputationCache(computation_capacity)
        self._histogram_cache = HistogramCache()
    
    def get_render_cache(self):
        return self._render_cache
    
    def get_computation_cache(self):
        return self._computation_cache

    def get_histogram_cache(self):
        return self._histogram_cache
