from wiser.raster import RasterDataSet

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState


class WISERControl:
    '''
    This class exposes certain parts of the WISER application's state, and
    provides operations that can be performed on the WISER application,
    allowing plugins to interact with the WISER application programmatically.
    '''


    def __init__(self, app_state: 'ApplicationState'):
        # Try to keep this private, so that plugin writers and others know they
        # really shouldn't muck around with it!
        self.__app_state: 'ApplicationState' = app_state


    def add_dataset(self, dataset: RasterDataSet) -> None:
        '''
        Add the specified ``RasterDataSet`` object to the data-sets being
        displayed by WISER.

        The dataset itself will be assigned a new unique ID by WISER, accessible
        via the :method:`RasterDataSet.get_id` method.
        '''
        self.__app_state.add_dataset(dataset)


    def dataset_from_numpy_array(self, arr: np.ndarray) -> RasterDataSet:
        '''
        Convert a NumPy array into a ``RasterDataSet`` object.  This operation
        doesn't cause WISER to display the dataset object; to do so, call the
        :method:`add_dataset` method with the new object.

        Note that the returned ``RasterDataSet`` will not have any metadata for
        its bands.
        '''
        return self.__app_state.get_loader().dataset_from_numpy_array(arr)
