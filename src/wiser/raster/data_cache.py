from typing import OrderedDict as OrderedDictType, Union, Tuple, Dict, List
from collections import OrderedDict

import numpy as np

class Cache:
    """
    A generic cache implementation using an ordered dictionary to store key-value pairs with size management.

    Attributes:
        _capacity (int): The maximum allowed size of the cache in bytes.
        _size (int): The current size of the cache in bytes.
        _cache (OrderedDictType[int, Union[np.ndarray, np.ma.masked_array]]): 
            An ordered dictionary storing cached items with their keys.
        _key_lookup_table (Dict[int, List[int]]): 
            A dictionary mapping partial keys to lists of cache keys for efficient lookup and eviction.
    """

    def __init__(self, capacity: int = 3000000000):
        self._capacity = capacity
        self._size = 0
        self._cache: OrderedDictType[int: Union[np.ndarray, np.ma.masked_array]] = OrderedDict()
        self._key_lookup_table: Dict[int: List[int]] = {}
    
    def _evict(self):
        """
        Evicts items from the cache until the current size is within the capacity.

        This method removes the least recently added items first (FIFO order) and updates the current size.
        """
        while self._size > self._capacity:
            if not self._cache:
                break
            key, value = self._cache.popitem(last=False)
            self._size -= value.nbytes

    def clear_cache(self):
        """
        Clears all items from the cache and resets the current size to zero.

        Raises:
            AssertionError: If the cache size does not reduce to zero after clearing.
        """
        for key in self._cache:
            self._size -= self._cache[key].nbytes
            del self._cache[key]
        assert np.isclose(self._size, 0.0) 

    def remove_cache_item(self, key: int):
        """
        Removes a specific item from the cache by its key.

        Args:
            key (int): The key of the item to be removed.
        """
        if key in self._cache:
            self._size -= self._cache[key].nbytes
            del self._cache[key]

    def add_cache_item(self, key: int, value: Union[np.ndarray, np.ma.masked_array]):
        """
        Adds a new item to the cache. Evicts existing items if necessary to maintain capacity.

        Args:
            key (int): The key associated with the value.
            value (Union[np.ndarray, np.ma.masked_array]): The data to be cached.
        """
        data_size = value.nbytes
        if data_size > self._capacity:
            print(f'Size of data exceeds cache size: {data_size} > {self._capacity}')
            return
        if self._size + data_size > self._capacity:
            self._evict()
        self._cache[key] = value
        self._size += value.nbytes

    def get_cache_key(self, *args):
        """
        Generates a unique cache key based on input arguments.

        This method should be implemented by subclasses to provide specific key generation logic.

        Raises:
            NotImplementedError: Always raised to indicate that the method needs to be overridden.
        """
        raise NotImplementedError(f'get_cache_key is not implemented for {type(self)}')
    
    def get_cache_item(self, key: int):
        """
        Retrieves an item from the cache by its key.

        Args:
            key (int): The key of the item to retrieve.

        Returns:
            Union[np.ndarray, np.ma.masked_array, None]: The cached data if the key exists, otherwise None.
        """
        if key in self._cache:
            return self._cache[key]
        return None
    
    def in_cache(self, key: int):
        return key in self._cache
    
    def get_partial_key(self, dataset):
        """
        Generates a partial key based on the dataset.

        This method should be implemented by subclasses to provide specific partial key generation logic.

        Args:
            dataset: The dataset for which to generate the partial key.

        Raises:
            NotImplementedError: Always raised to indicate that the method needs to be overridden.
        """
        raise NotImplementedError(f'get_partial_key is not implemented for {type(self)}')
    
    def lookup_keys(self, partial_key: int):
        """
        Retrieves all cache keys associated with a given partial key.

        Args:
            partial_key (int): The partial key to look up.

        Returns:
            List[int]: A list of cache keys associated with the partial key.
        """
        return self._key_lookup_table[partial_key]
    
    def clear_keys_from_partial(self, partial_key: int):
        """
        Removes all cache items associated with a specific partial key.

        Args:
            partial_key (int): The partial key whose associated cache items are to be removed.
        """
        keys = self.lookup_keys(partial_key)
        for key in keys:
            self.remove_cache_item(key)

class RenderCache(Cache):
    """
    A cache specialized for rendering data, extending the generic Cache class.

    This cache manages rendered datasets, allowing efficient storage and retrieval based on dataset,
    band tuples, and stretches.
    """
    
    def __init__(self, capacity: int = 3000000000):
        super().__init__(capacity)
        self._key_lookup_table: Dict[int: List[int]] = {}

    def get_cache_key(self, dataset, band_tuple: Tuple[int], stretches):
        """
        Creates and returns a unique cache key based on the dataset, band tuple, and stretches.

        It also stores the cache key with the partial key generated from the dataset in order to 
        facilitate efficient lookup and eviction when the dataset is deleted.

        Args:
            dataset: The dataset associated with the cache item.
            band_tuple (Tuple[int]): A tuple of band indices.
            stretches: The stretch parameters applied to the data.

        Returns:
            int: A unique hash representing the cache key.
        """
        partial_key = self.get_partial_key(dataset)
        cache_key = hash((dataset, *band_tuple, *stretches))
        if partial_key not in self._key_lookup_table:
            self._key_lookup_table[partial_key] = []
        self._key_lookup_table[partial_key].append(cache_key)
        return cache_key
    
    def get_partial_key(self, dataset):
        return hash((dataset))


class ComputationCache(Cache):
    """
    A cache specialized for computational results, extending the generic Cache class.

    This cache manages results of computations on datasets and bands, allowing efficient storage and retrieval
    based on dataset and band index.
    """

    def __init__(self, capacity: int = 7000000000):
        super().__init__(capacity)

    def get_cache_key(self, dataset, band_index: int = -1):
        """
        Creates and returns a unique cache key based on the dataset and band index.

        Args:
            dataset: The dataset associated with the computation.
            band_index (int, optional): The index of the band. Defaults to -1.

        Returns:
            int: A unique hash representing the cache key.
        """
        partial_key = self.get_partial_key(dataset)
        cache_key = hash((dataset, band_index))
        if partial_key not in self._key_lookup_table:
            self._key_lookup_table[partial_key] = []
        self._key_lookup_table[partial_key].append(cache_key)
        return hash((dataset, band_index))
    
    def get_partial_key(self, dataset):
        return hash((dataset))

class HistogramCache(Cache):
    """
    A cache specialized for histogram data, extending the generic Cache class.

    This cache manages histogram bins and edges, allowing efficient storage and retrieval
    based on dataset, band index, stretch type, and conditioner type.

    This cache can be much smaller than the other cache's because histogram data is small.
    """
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
            print(f'Size of data exceeds histogram cache size: {data_size} > {self._capacity}')
            return
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

    def __init__(self, render_capacity:int = 10000000000, computation_capacity: int=10000000000):
        self._render_cache = RenderCache(render_capacity)
        self._computation_cache = ComputationCache(computation_capacity)
        self._histogram_cache = HistogramCache()
    
    def get_render_cache(self):
        return self._render_cache
    
    def get_computation_cache(self):
        return self._computation_cache

    def get_histogram_cache(self):
        return self._histogram_cache
